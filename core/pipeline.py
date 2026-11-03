"""
Shared run pipeline for the interval abstractions (``RunIMDP.py``, ``Run2IMDP.py``).

Both entrypoints follow the same recipe: configure the run, obtain an abstraction (either
built from scratch or restored from a checkpoint), solve it, and turn the resulting value
function and policy into simulations and figures. Only the abstraction itself and the solver
call differ between them, so everything else lives here and the abstraction step is passed
in as a callback.
"""

import copy
import csv
import datetime
import logging
import os
import pickle
import random
import re
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

import benchmarks
from core.abstraction.model import attach_region_data
from core.abstraction.partition import DensePartition
from core.jax_config import configure_jax
from core.utils import add_file_handler, configure_logging

logger = logging.getLogger(__name__)

# Repository root, i.e. the directory holding the entrypoint scripts (this module lives in core/).
ROOT_DIR = Path(__file__).resolve().parent.parent

# Arguments that determine the abstraction (the batch sizes too, as the random keys are drawn per batch)
ABSTRACTION_SETTINGS = ('coarse_factor', 'num_samples', 'frs_num_samples', 'alpha', 'pac_beta', 'seed',
                        'batch_size', 'frs_batch_size')


def configure_run(args):
    '''
    Configure logging, JAX, and the random seeds for this run.

    :param args: Argument namespace, extended in place with the JAX key and the run paths.
    :return: Timestamp string identifying the run.
    '''

    configure_logging(args.log_level)
    configure_jax(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # Set current working directory
    args.cwd = os.fspath(ROOT_DIR)
    args.root_dir = ROOT_DIR

    # Key results of the run, exported to summary.csv by write_summary
    args.summary = {}

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def record_synthesis(args, start, lower, upper):
    '''
    Store the synthesis time and the bounds at the initial state in the run summary.

    :param args: Argument namespace.
    :param start: Time at which the synthesis started.
    :param lower: Lower bound on the satisfaction probability in the initial state.
    :param upper: Upper bound on the satisfaction probability in the initial state (None if unknown).
    '''

    args.summary['time_synthesis'] = time.time() - start
    args.summary['lower_bound'] = float(lower)
    args.summary['upper_bound'] = None if upper is None else float(upper)
    logger.info('<<< Synthesis took %.3f sec. >>>\n', args.summary['time_synthesis'])


def write_summary(args, model):
    '''
    Export the run summary (settings, runtimes, model size, and results) to summary.csv.

    :param args: Argument namespace.
    :param model: Benchmark model.
    '''

    rows = {
        'model': args.model,
        'abstraction': args.summary.get('abstraction'),
        'specification': getattr(model, 'specification', None),
        'noise': model.noise['type'],  # the model's own noise, which need not follow --noise_distr
        'solver': args.solver,
        'horizon': 'none' if args.horizon is None else args.horizon,
        'checkpoint': args.load_checkpoint,
    }
    # The abstraction settings are those of the run that built the abstraction (see prepare_abstraction)
    for key in ABSTRACTION_SETTINGS + ('time_abstraction', 'time_synthesis', 'num_states', 'num_intervals',
                                        'lower_bound', 'upper_bound', 'empirical_satprob'):
        rows[key] = args.summary.get(key)

    csv_path = args.output_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['metric', 'value'])
        writer.writerows(rows.items())
    logger.info('Run summary saved to %s', csv_path)


def _abstraction_settings(args):
    '''Values of ABSTRACTION_SETTINGS in an argument namespace (None if absent, e.g. in an older checkpoint).'''

    return {key: getattr(args, key, None) for key in ABSTRACTION_SETTINGS}


def _open_output_dir(args, stamp):
    '''
    Create the timestamped output directory for this run and start logging into it.

    :param args: Argument namespace, extended in place with the output directory.
    :param stamp: Run timestamp.
    :return: Path of the created directory.
    '''

    run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.summary['abstraction']}_{args.model}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = run_output_dir
    add_file_handler(run_output_dir, stamp)

    return run_output_dir


def save_checkpoint(args, key, model, partition, abstraction):
    '''
    Write the model, partition, and abstraction to checkpoint.pkl in the run output directory.

    :param args: Argument namespace.
    :param key: Key under which the abstraction is stored ('imdp' or '2imdp').
    :param model: Benchmark model.
    :param partition: Partition of the continuous state space.
    :param abstraction: Abstraction object to store.
    '''

    # Strip JAX runtime objects that can't be pickled
    args_to_save = copy.copy(args)
    del args_to_save.rvi_device
    del args_to_save.jax_key

    ckpt_path = args.output_dir / 'checkpoint.pkl'
    logger.info('Saving checkpoint to %s', ckpt_path)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'model': model, 'partition': partition, key: abstraction, 'args': args_to_save}, f)
    logger.info('Checkpoint saved.\n')


def save_safety_results(args, s0, lower, upper, policy):
    """Save time-indexed values and global-action policy; index h is remaining horizon."""
    np.savez_compressed(
        args.output_dir / 'safety_synthesis.npz',
        V_lower=lower, V_upper=upper, policy=policy,
        initial_state=int(s0), horizon=int(args.horizon),
    )
    logger.info('Safety horizon=%d, initial state=%d: lower=%.6f, upper=%.6f',
                args.horizon, s0, lower[args.horizon, s0], upper[args.horizon, s0])


def is_safety(formula):
    """Recognize the finite-horizon safety specification G(safe)."""
    return bool(re.fullmatch(r"G\(safe\)", re.sub(r"\s+", "", formula)))


def safe_state_mask(partition, safe_set, *, optimistic=False):
    """Conservatively label cells fully contained in a union of safe boxes."""
    boxes = np.asarray(safe_set, dtype=float)
    lo = np.asarray(partition.regions['lower_bounds'])
    hi = np.asarray(partition.regions['upper_bounds'])
    d = lo.shape[1]
    if boxes.ndim == 2 and boxes.shape == (2, d):
        boxes = boxes[None, :, :]
    if boxes.ndim != 3 or boxes.shape[1:] != (2, d) or not len(boxes):
        raise ValueError('safe_set must have shape (n_boxes, 2, dimension)')
    if np.any(boxes[:, 0, :] >= boxes[:, 1, :]):
        raise ValueError('safe_set boxes must have positive side lengths')
    mask = np.zeros(len(lo), dtype=bool)
    for a, b in boxes:
        if optimistic:
            mask |= np.all(lo < b, axis=1) & np.all(hi > a, axis=1)
        else:
            mask |= np.all(lo >= a, axis=1) & np.all(hi <= b, axis=1)
    return mask


def is_reach_avoid(formula):
    """
    Check whether the specification is exactly

        (!unsafe) U goal

    allowing whitespace and optional parentheses
    """

    # Remove whitespace
    formula = re.sub(r"\s+", "", formula)

    # Accepted syntactic representations
    reach_avoid_formulas = {
        "!unsafeUgoal",
        "(!unsafe)Ugoal",
        "!unsafeU(goal)",
        "(!unsafe)U(goal)",
    }

    return formula in reach_avoid_formulas


def is_scLTL(formula):
    # Check whether the specification is co-safe

    check_result = subprocess.run(
        [
            "ltlfilt",
            "-f", formula,
            "--guarantee"
        ],
        capture_output=True,
        text=True,
        check=False
    )

    return bool(check_result.stdout.strip())


def prepare_abstraction(args, stamp, name, build):
    '''
    Obtain the abstraction for this run, either from a checkpoint or by building it from scratch.

    :param args: Argument namespace.
    :param stamp: Run timestamp.
    :param name: Name of the abstraction ('IMDP' or '2IMDP'); its lowercase form is the
        key under which the abstraction is stored in the checkpoint.
    :param build: Callback (args, model, partition, s_init) -> abstraction, invoked when the
        abstraction is generated from scratch.
    :return: Tuple of (model, partition, abstraction).
    '''

    key = name.lower()
    args.summary['abstraction'] = name

    if args.load_checkpoint:
        # --- Load abstraction from checkpoint ---
        ckpt_path = Path(args.load_checkpoint)
        logger.info('Loading checkpoint from %s', ckpt_path)
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)

        if key not in ckpt:
            raise KeyError(
                f"Checkpoint '{ckpt_path}' holds no '{key}' abstraction (keys: {sorted(ckpt)}). "
                f"Checkpoints are specific to the entrypoint that wrote them."
            )

        model = ckpt['model']
        set_horizon(args, model)
        partition = ckpt['partition']
        abstraction = attach_region_data(ckpt[key], model, partition)
        # Abstraction settings, time and size from the run that built the checkpoint
        args.summary.update(_abstraction_settings(ckpt['args']))
        args.summary.update(getattr(ckpt['args'], 'summary', {}))

        # Name the output folder after the checkpointed model rather than --model, which is
        # typically left unset when a checkpoint is replayed.
        args.model = ckpt['args'].model

        run_output_dir = _open_output_dir(args, stamp)
        logger.info('Run %s | model=%s (from checkpoint)', stamp, args.model)
        logger.info('Output directory: %s', run_output_dir)
        logger.info('\n=== %s loaded from checkpoint: %s ===', name, ckpt_path)

        return model, partition, abstraction

    # --- Build abstraction from scratch ---
    run_output_dir = _open_output_dir(args, stamp)
    logger.info('Run %s | model=%s | noise=%s | batch=%d | frs_batch=%d',
                stamp, args.model, args.noise_distr, args.batch_size, args.frs_batch_size)
    logger.info('Output directory: %s', run_output_dir)

    logger.info('\n=== Generate %s from scratch ===', name)
    logger.debug('Arguments: %s', vars(args))

    t = time.time()
    model = benchmarks.create_model(args)
    logger.info('<<< Generating model took %.3f sec. >>>\n', time.time() - t)

    formula = getattr(model, 'specification', None)
    if formula is None:
        raise ValueError('The benchmark does not define a specification.')
    if is_safety(formula):
        if getattr(model, 'safe_set', None) is None:
            raise ValueError('G(safe) requires model.safe_set.')
    elif is_reach_avoid(formula):
        regions = getattr(model, 'regions', {})
        labels = getattr(model, 'labels', {})
        aps = getattr(model, 'atomic_propositions', ())
        if not {'r_goal', 'r_unsafe'} <= set(regions):
            raise ValueError('Reach-avoid requires r_goal and r_unsafe regions.')
        if not {'goal', 'unsafe'} <= set(aps):
            raise ValueError('Reach-avoid requires goal and unsafe propositions.')
        if 'goal' not in labels.get('r_goal', ()) or 'unsafe' not in labels.get('r_unsafe', ()):
            raise ValueError('Invalid reach-avoid region labels.')
    elif not is_scLTL(formula):
        raise ValueError('Specification must be reach-avoid, scLTL, or G(safe).')
    set_horizon(args, model)

    t = time.time()

    # Create partition of the continuous state space into convex polytope
    partition = DensePartition(model=model, float_dtype=args.floatprecision)

    s_init, s_init_exists = partition.x2state(model.x0)
    if not s_init_exists:
        raise ValueError(f"Initial state x0={model.x0} is not in the partition.")

    # The callback holds the only reference to its intermediate action data, which is freed
    # when it returns (this matters: on the larger benchmarks it dominates peak memory).
    abstraction = build(args, model, partition, s_init)

    logger.info('Initial state x0=%s → state index %d\n', model.x0, s_init)
    args.summary.update(_abstraction_settings(args))
    args.summary['time_abstraction'] = time.time() - t
    args.summary['num_states'] = abstraction.nr_states  # including the absorbing state
    logger.info('<<< Generating %s abstraction took %.3f sec. >>>\n', name, args.summary['time_abstraction'])

    if args.save_checkpoint:
        save_checkpoint(args, key, model, partition, abstraction)

    return model, partition, abstraction


def extract_policy_inputs(args, partition, policy, nr_states, *, time_indexed=False):
    """
    Map action indices to continuous control inputs.

    Supports:
        Reach-avoid:                policy[s]
        Finite-horizon reach-avoid: policy[h, s]
        sc-LTL:                     policy[s, z]

    Parameters
    ----------
    args : Namespace
        Argument namespace.

    partition : object
        Partition of the continuous state space.

    policy : np.ndarray
        Global action indices. Shape:
            (nr_states,) for reach-avoid
            (H+1, nr_states) for finite-horizon reach-avoid, where h is the number of remaining steps
            (nr_states, n_dfa_states) for sc-LTL

    nr_states : int
        Number of IMDP states, including the absorbing state.

    time_indexed : bool
        Whether policy is a finite-horizon reach-avoid policy, which has the same number of
        dimensions as an sc-LTL policy.

    Returns
    -------
    np.ndarray
        Continuous control inputs.

        Reach-avoid:
            (nr_states - 1, control_dim)

        Finite-horizon reach-avoid:
            (H+1, nr_states - 1, control_dim)

        sc-LTL:
            (nr_states - 1, n_dfa_states, control_dim)

        States without an enabled action contain NaN.
    """

    if time_indexed:
        return np.stack([extract_policy_inputs(args, partition, p, nr_states) for p in policy])

    float_dtype = args.floatprecision

    policy = np.asarray(policy)

    actions_np = np.asarray(partition.regions["actions"])

    n_states = nr_states - 1

    # Broadcast the shared (1, n_actions, control_dim) action grid across states.
    actions_np = np.broadcast_to(
        actions_np,
        (n_states, *actions_np.shape[1:])
    )

    control_dim = actions_np.shape[2]
    n_actions = actions_np.shape[1]

    # --------------------------------------------------
    # Reach-avoid policy: sigma(s)
    # --------------------------------------------------

    if policy.ndim == 1:

        if policy.shape != (nr_states,):
            raise ValueError("Invalid reach-avoid policy shape")

        policy_inputs = np.full(
            (n_states, control_dim),
            np.nan,
            dtype=float_dtype
        )

        valid = (
            (policy[:n_states] >= 0)
            & (policy[:n_states] < n_actions)
        )

        state_ids = np.flatnonzero(valid)

        policy_inputs[state_ids] = actions_np[
            state_ids,
            policy[state_ids]
        ]

    # --------------------------------------------------
    # sc-LTL policy: sigma(s,z)
    # --------------------------------------------------

    elif policy.ndim == 2:

        if policy.shape[0] != nr_states:
            raise ValueError("Invalid sc-LTL policy shape")

        n_dfa_states = policy.shape[1]

        policy_inputs = np.full(
            (n_states, n_dfa_states, control_dim),
            np.nan,
            dtype=float_dtype
        )

        valid = (
            (policy[:n_states] >= 0)
            & (policy[:n_states] < n_actions)
        )

        state_ids, dfa_ids = np.where(valid)

        policy_inputs[state_ids, dfa_ids] = actions_np[
            state_ids,
            policy[state_ids, dfa_ids]
        ]

    else:
        raise ValueError(
            "Policy must have shape (n_states,) "
            "or (n_states, n_dfa_states)"
        )

    return policy_inputs

def _guard(description, failures, fn, *fn_args, **fn_kwargs):
    '''
    Run a single post-processing step, recording a failure instead of propagating it.

    :param description: Human-readable name of the step, used in the error log.
    :param failures: List that failed step descriptions are appended to.
    :param fn: Callable implementing the step.
    :return: Return value of the step, or None if it failed.
    '''

    try:
        return fn(*fn_args, **fn_kwargs)
    except Exception:
        logger.exception('Post-processing step failed: %s', description)
        failures.append(description)

        # Drop any half-drawn figure so it can't leak into the next step.
        import matplotlib.pyplot as plt
        plt.close('all')

        return None


def report(args, stamp, model, partition, V, policy, policy_inputs, iterations=1000, num_traces=10,
           sim_horizon=1000):
    '''
    Validate the synthesised policy by simulation and write the figures for this run.

    Every step is best-effort: policy synthesis is the expensive part of a run and its result
    is only persisted when --save_checkpoint is set, so a failure while validating or plotting
    is logged and the remaining steps still run rather than discarding the run.

    :param args: Argument namespace.
    :param stamp: Run timestamp.
    :param model: Benchmark model.
    :param partition: Partition of the continuous state space.
    :param V: Value function over the abstraction states; with a horizon H, of shape (H+1, S+1),
        where index h is the number of remaining steps.
    :param policy: Action index per state; with a horizon, indexed by remaining steps like V.
    :param policy_inputs: Control input per state, from extract_policy_inputs; with a horizon,
        indexed by remaining steps like V.
    :param iterations: Number of Monte Carlo simulations to run.
    :param num_traces: Number of simulated traces to draw in the trace plots.
    :param sim_horizon: Number of simulated steps without a horizon; with a horizon H, exactly H.
    :return: List of descriptions of the steps that failed (empty if all succeeded).
    '''

    from core.plotting.heatmap import heatmap
    from core.plotting.trajectories import plot_trajectories, plot_trajectories_3d
    from core.validate.simulate import MonteCarloSim

    failures = []

    # With a horizon H, the simulator indexes by elapsed steps k = H - h, and the heatmaps show
    # the values and controls with all H steps remaining
    sim_policy, sim_inputs = policy, policy_inputs
    if np.ndim(policy) == 2:
        sim_policy, sim_inputs, sim_horizon = policy[::-1], policy_inputs[::-1], len(policy) - 1
        V, policy_inputs = V[-1], policy_inputs[-1]

    sim = _guard('Monte Carlo simulation', failures, MonteCarloSim,
                 model, partition, sim_policy, sim_inputs, model.x0,
                 verbose=False, iterations=iterations, sim_horizon=sim_horizon)
    if sim is not None:
        logger.info('Empirical satisfaction probability: %s', sim.results['satprob'])
        args.summary['empirical_satprob'] = float(sim.results['satprob'])
    _guard('run summary', failures, write_summary, args, model)

    _guard('satisfaction probability heatmap', failures, heatmap,
           args, stamp, idx_show=model.plot_dimensions,
           partition=partition, results=V, filename='satprob')

    _guard('policy heatmap', failures, heatmap,
           args, stamp, idx_show=model.plot_dimensions,
           partition=partition, results=policy_inputs[:, 0], filename='policy')

    if sim is None:
        # Everything below draws simulated traces, which we do not have.
        logger.error('%d post-processing step(s) failed: %s', len(failures), ', '.join(failures))
        return failures

    traces = sim.results['traces']

    _guard('trajectory plot', failures, plot_trajectories,
           args, stamp, model.plot_dimensions, partition, model, traces,
           num_trajectories=num_traces, add_unsafe_box=False)

    # Models that are natural to show in 3D (e.g. the 6D drone) declare the three dimensions to use.
    dims_3d = getattr(model, 'plot_dimensions_3d', None)
    if dims_3d is not None:
        logger.info('Plotting trajectories in 3D (dimensions %s)...', list(dims_3d))
        _guard('3D trajectory plot', failures, plot_trajectories_3d,
               args, stamp, dims_3d, partition, model, traces,
               num_trajectories=num_traces, filename='trajectories_3d')

    # Models that can animate a trajectory declare which state dimensions the animation needs.
    gif_dims = list(getattr(model, 'gif_dimensions', []) or [])
    if gif_dims and hasattr(model, 'plot_trajectory_gif'):
        logger.info('Plotting %s trajectory gif...', args.model)
        states = np.array(traces[0]['x'])
        # A single dimension is passed as a flat array, matching the animation signatures.
        trajectory = states[:, gif_dims[0]] if len(gif_dims) == 1 else states[:, gif_dims]
        _guard('trajectory GIF', failures, model.plot_trajectory_gif, trajectory,
               filename=str(args.output_dir / f'{args.model.lower()}_{stamp}.gif'))

    if failures:
        logger.error('%d post-processing step(s) failed: %s', len(failures), ', '.join(failures))

    return failures


def finish(args):
    '''
    Log the closing banner for the run.

    :param args: Argument namespace.
    '''

    logger.info('\n=== Run finished. Output directory: %s ===', args.output_dir)
    logger.info('====================\n')


def extract_product_policy_inputs(args, partition, policy, nr_states):
    """Map global action indices policy[s,z] to physical inputs u[s,z,:]."""
    policy = np.asarray(policy, dtype=np.int32)
    if policy.ndim != 2 or policy.shape[0] != nr_states:
        raise ValueError('Expected policy shape (nr_states, n_dfa_states)')
    actions = np.asarray(partition.regions['actions'])
    if actions.shape[0] == 1:
        actions = np.broadcast_to(actions, (nr_states - 1, *actions.shape[1:]))
    if actions.shape[0] != nr_states - 1:
        raise ValueError('Partition action count does not match physical states')
    result = np.full((*policy.shape, actions.shape[-1]), np.nan, dtype=args.floatprecision)
    valid = policy[:-1] >= 0
    ss, zz = np.nonzero(valid)
    if np.any(policy[ss, zz] >= actions.shape[1]):
        raise ValueError('Policy references an action outside the partition action grid')
    result[ss, zz] = actions[ss, policy[ss, zz]]
    return result


def draw_regions_of_interest(ax, model, dimensions=None):
    """Overlay transparent region boundaries on an existing 2D heatmap axes."""
    from matplotlib.patches import Rectangle

    if dimensions is None:
        dimensions = model.plot_dimensions
    if len(dimensions) != 2:
        raise ValueError('Region overlays require exactly two plot dimensions')
    d0, d1 = map(int, dimensions)
    for name, boxes in getattr(model, 'regions', {}).items():
        boxes = np.asarray(boxes, dtype=float)
        if boxes.size == 0:
            continue
        if boxes.ndim != 3 or boxes.shape[1] != 2 or boxes.shape[2] <= max(d0, d1):
            raise ValueError(f'Invalid geometry for region {name}')
        for lower, upper in boxes:
            ax.add_patch(Rectangle(
                (lower[d0], lower[d1]),
                upper[d0] - lower[d0],
                upper[d1] - lower[d1],
                facecolor='none', edgecolor='black', linewidth=1.8,
                linestyle='--', zorder=20,
                transform=getattr(ax, 'roi_physical_transform', ax.transData),
            ))

def report_scLTL(args, stamp, model, partition, V_lower, policy, dfa, labels,
                  policy_inputs, *, iterations=1000, num_traces=10, sim_horizon=1000):
    """Validate product policy and plot the initial-label DFA slice.

    Heatmaps display values and controls for z = delta(z0,L(s)) at each s.
    They are not plots of the full product-state policy.
    """
    from core.plotting.heatmap import heatmap
    from core.plotting.trajectories import plot_trajectories, plot_trajectories_3d
    from core.validate.simulate import MonteCarloSim_scLTL

    failures = []
    labels = np.asarray(labels, dtype=np.int64)
    initial_z = np.asarray(dfa.transitions, dtype=np.int32)[int(dfa.initial_state), labels[:partition.size]]
    physical_s = np.arange(partition.size)
    V_initial = np.asarray(V_lower)[physical_s, initial_z]
    inputs_initial = np.asarray(policy_inputs)[physical_s, initial_z]

    sim = _guard('sc-LTL Monte Carlo simulation', failures, MonteCarloSim_scLTL,
                 model, partition, policy, policy_inputs, model.x0, dfa, labels,
                 iterations=iterations, sim_horizon=sim_horizon, verbose=False)
    if sim is not None:
        logger.info('Empirical sc-LTL satisfaction probability: %.6f', sim.results['satprob'])
        np.savez_compressed(args.output_dir / 'scLTL_monte_carlo.npz',
                            satisfied=sim.results['satisfied'],
                            satprob=sim.results['satprob'])
        args.summary['empirical_satprob'] = float(sim.results['satprob'])
    _guard('run summary', failures, write_summary, args, model)

    _guard('sc-LTL initial-state satisfaction heatmap', failures, heatmap,
           args, stamp, idx_show=model.plot_dimensions,
           partition=partition, results=V_initial, filename='scLTL_satprob_initial',
           overlay=lambda ax: draw_regions_of_interest(ax, model, model.plot_dimensions))
    _guard('sc-LTL initial-state control heatmap', failures, heatmap,
           args, stamp, idx_show=model.plot_dimensions,
           partition=partition, results=inputs_initial[:, 0], filename='scLTL_policy_initial')

    if sim is not None:
        traces = sim.results['traces']
        _guard('sc-LTL trajectory plot', failures, plot_trajectories,
               args, stamp, model.plot_dimensions, partition, model, traces,
               num_trajectories=num_traces, add_unsafe_box=False,
               labeled_regions=True)
        dims_3d = getattr(model, 'plot_dimensions_3d', None)
        if dims_3d is not None:
            _guard('sc-LTL 3D trajectory plot', failures, plot_trajectories_3d,
                   args, stamp, dims_3d, partition, model, traces,
                   num_trajectories=num_traces, filename='scLTL_trajectories_3d',
                   labeled_regions=True)
    return failures

def _is_horizon(horizon):
    """Whether horizon is a nonnegative integer (booleans excluded)."""
    return (isinstance(horizon, (int, np.integer)) and not isinstance(horizon, (bool, np.bool_))
            and horizon >= 0)


def set_horizon(args, model):
    """Take the horizon exclusively from the benchmark: model.horizon, or None (unbounded) if unset.

    G(safe) requires a horizon and reach-avoid may have one; scLTL is only solved unbounded.
    """
    formula = getattr(model, 'specification', None) or ''
    if is_safety(formula):
        return set_safety_horizon(args, model)
    horizon = getattr(model, 'horizon', None)
    if horizon is not None:
        if not is_reach_avoid(formula):
            raise ValueError('model.horizon is only supported for reach-avoid and G(safe).')
        if not _is_horizon(horizon):
            raise ValueError('model.horizon must be a nonnegative integer.')
        horizon = int(horizon)
    args.horizon = horizon
    return horizon


def set_safety_horizon(args, model):
    """Take the finite safety horizon exclusively from the benchmark."""
    horizon = getattr(model, 'horizon', None)
    if not _is_horizon(horizon):
        raise ValueError('G(safe) requires a nonnegative integer model.horizon.')
    if getattr(model, 'safe_set', None) is None:
        raise ValueError('G(safe) requires model.safe_set.')
    args.horizon = int(horizon)  # compatibility with the existing safety solvers
    return args.horizon


def report_safety(args, stamp, model, partition, V_lower, V_upper, policy,
                  *, iterations=1000, num_traces=10):
    """Save safety value heatmaps, MC trajectories and MC outcome plots."""
    import matplotlib.pyplot as plt
    from core.plotting.heatmap import heatmap
    from core.plotting.utils import PNG_DPI

    failures = []
    H = set_safety_horizon(args, model)
    boxes = np.asarray(model.safe_set, dtype=float)
    if boxes.ndim == 2:
        boxes = boxes[None, :, :]
    n_states = len(partition.regions['idxs'])
    actions = np.broadcast_to(
        np.asarray(partition.regions['actions']),
        (n_states, *np.asarray(partition.regions['actions']).shape[1:]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Sample trajectories under the time-dependent policy sigma[remaining_steps, state].
    noise = np.asarray(model.noise.sample(size=(iterations, H))) if H else None
    successes = np.zeros(iterations, dtype=bool)
    traces = []
    for run in range(iterations):
        x = np.asarray(model.x0, dtype=float).copy()
        trajectory = [x.copy()]
        success = True
        for k in range(H + 1):
            if not any(np.all(x >= lo) and np.all(x <= hi) for lo, hi in boxes):
                success = False
                break
            state, exists = partition.x2state(x)
            if not exists:
                success = False
                break
            if k == H:
                break
            a = int(policy[H - k, int(state)])
            if a < 0:
                success = False
                break
            x = np.asarray(model.step(x, actions[int(state), a], noise[run, k]))
            trajectory.append(x.copy())
        successes[run] = success
        if run < num_traces:
            traces.append((np.asarray(trajectory), success))

    probability = float(successes.mean())
    np.savez_compressed(
        args.output_dir / 'safety_monte_carlo.npz',
        successes=successes, satprob=probability, horizon=H,
        traces=np.array([t for t, _ in traces], dtype=object),
    )
    logger.info('Finite-horizon safety empirical probability (%d runs): %.6f',
                iterations, probability)
    args.summary['empirical_satprob'] = probability
    _guard('run summary', failures, write_summary, args, model)

    # Same heatmap routine, timestamp and output directory as other benchmarks.
    for filename, values in (
        ('safety_lower', V_lower[H]),
        ('safety_upper', V_upper[H]),
    ):
        _guard(filename + ' heatmap', failures, heatmap,
               args, stamp, idx_show=model.plot_dimensions,
               partition=partition, results=values, filename=filename)

    def save_figure(fig, filename):
        for extension in ('png', 'pdf'):
            fig.savefig(args.output_dir / f'{filename}_{stamp}.{extension}',
                        bbox_inches='tight', dpi=PNG_DPI)
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(['Safe', 'Unsafe'], [int(successes.sum()), int((~successes).sum())])
    ax.set_ylabel('Monte Carlo runs')
    ax.set_title(f'Safety over {H} transitions: {probability:.3f}')
    save_figure(fig, 'safety_mc_outcomes')

    dims = np.atleast_1d(np.asarray(model.plot_dimensions, dtype=int))
    if len(dims) in (1, 2) and traces:
        fig, ax = plt.subplots()
        for trajectory, success in traces:
            color = 'tab:blue' if success else 'tab:red'
            if len(dims) == 1:
                ax.plot(np.arange(len(trajectory)), trajectory[:, dims[0]],
                        color=color, alpha=0.65)
            else:
                ax.plot(trajectory[:, dims[0]], trajectory[:, dims[1]],
                        color=color, alpha=0.65)
        if len(dims) == 1:
            ax.set(xlabel='Time step', ylabel=f'x[{dims[0]}]')
            for lo, hi in boxes:
                ax.axhspan(lo[dims[0]], hi[dims[0]], color='green', alpha=0.08)
        else:
            from matplotlib.patches import Rectangle
            ax.set(xlabel=f'x[{dims[0]}]', ylabel=f'x[{dims[1]}]')
            for lo, hi in boxes:
                xy = lo[dims]
                wh = hi[dims] - xy
                ax.add_patch(Rectangle(xy, *wh, fill=False,
                                       edgecolor='green', linewidth=1.5))
        ax.set_title('Safety Monte Carlo trajectories (blue: safe, red: unsafe)')
        save_figure(fig, 'safety_mc_trajectories')
    if failures:
        logger.error('Safety plotting failures: %s', ', '.join(failures))
    return probability
