"""
Shared run pipeline for the abstraction entrypoints (``RunIMDP.py``, ``RunSVMDP.py``).

Both entrypoints follow the same recipe: configure the run, obtain an abstraction (either
built from scratch or restored from a checkpoint), solve it, and turn the resulting value
function and policy into simulations and figures. Only the abstraction itself and the solver
call differ between them, so everything else lives here and the abstraction step is passed
in as a callback.
"""

import copy
import datetime
import logging
import os
import pickle
import random
import time
from pathlib import Path

import jax
import numpy as np

import benchmarks
from core.abstraction.partition import DensePartition
from core.jax_config import configure_jax
from core.utils import add_file_handler, configure_logging

import re
import subprocess
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# Repository root, i.e. the directory holding the entrypoint scripts (this module lives in core/).
ROOT_DIR = Path(__file__).resolve().parent.parent


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

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _open_output_dir(args, stamp):
    '''
    Create the timestamped output directory for this run and start logging into it.

    :param args: Argument namespace, extended in place with the output directory.
    :param stamp: Run timestamp.
    :return: Path of the created directory.
    '''

    run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = run_output_dir
    add_file_handler(run_output_dir, stamp)

    return run_output_dir


def save_checkpoint(args, key, model, partition, abstraction):
    '''
    Write the model, partition, and abstraction to checkpoint.pkl in the run output directory.

    :param args: Argument namespace.
    :param key: Key under which the abstraction is stored ('imdp' or 'svmdp').
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
    :param name: Name of the abstraction ('IMDP' or 'SVMDP'); its lowercase form is the
        key under which the abstraction is stored in the checkpoint.
    :param build: Callback (args, model, partition, s_init) -> abstraction, invoked when the
        abstraction is generated from scratch.
    :return: Tuple of (model, partition, abstraction).
    '''

    key = name.lower()

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
        partition = ckpt['partition']
        abstraction = ckpt[key]

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

    # Check that the specification is scLTL
    formula = getattr(model, "specification", None)
    regions = getattr(model, "regions", None)
    labels = getattr(model, "labels", None)
    atomic_propositions = getattr(model, "atomic_propositions", None)
    
    if formula is None:
        raise ValueError(
            "The benchmark does not define a specification."
        )
    else:
        reach_avoid = is_reach_avoid(formula)
        if reach_avoid & ("r_goal" not in regions or "r_unsafe" not in regions):
            raise ValueError("If the specification is reach-avoid, model.regions must contain 'r_goal' "
                    "and 'r_unsafe'.")
        if  "goal" not in atomic_propositions or "unsafe" not in atomic_propositions:
            raise ValueError("If the specification is reach-avoid, model.atomic_propositions must contain " \
                    "'goal' and 'unsafe'.")
        if "r_goal" not in labels or "r_unsafe" not in labels:
            raise ValueError("If the specification is reach-avoid, model.labels must contain " \
                                "'goal' and 'unsafe'.")
        if ("goal" not in labels.get("r_goal", set()) or "unsafe" not in labels.get("r_unsafe", set())):
            raise ValueError(
                "self.labels must map 'r_goal' to {'goal'} and 'r_unsafe' to {'unsafe'}."
            )
        scLTL = is_scLTL(formula)
        if not (reach_avoid | scLTL):
            raise ValueError(
                "ERROR: invalid specification. The specification must be" \
                " reach-avoid or co-safe (scLTL). "
                )

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
    logger.info('<<< Generating %s abstraction took %.3f sec. >>>\n', name, time.time() - t)

    if args.save_checkpoint:
        save_checkpoint(args, key, model, partition, abstraction)

    return model, partition, abstraction


def extract_policy_inputs(args, partition, policy, nr_states):
    """
    Map action indices to continuous control inputs.

    Supports:
        Reach-avoid: policy[s]
        sc-LTL:      policy[s, z]

    Parameters
    ----------
    args : Namespace
        Argument namespace.

    partition : object
        Partition of the continuous state space.

    policy : np.ndarray
        Global action indices. Shape:
            (nr_states,) for reach-avoid
            (nr_states, n_dfa_states) for sc-LTL

    nr_states : int
        Number of IMDP states, including the absorbing state.

    Returns
    -------
    np.ndarray
        Continuous control inputs.

        Reach-avoid:
            (nr_states - 1, control_dim)

        sc-LTL:
            (nr_states - 1, n_dfa_states, control_dim)

        States without an enabled action contain NaN.
    """

    float_dtype = args.floatprecision

    policy = np.asarray(policy)

    actions_np = np.asarray(partition.regions["actions"])

    n_states = nr_states - 1

    # DensePartition stores actions as
    # (1, n_actions, control_dim).
    # SparsePartition stores them as
    # (n_states, n_actions, control_dim).
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


def report(args, stamp, model, partition, V, policy, policy_inputs, iterations=100, num_traces=10):
    '''
    Validate the synthesised policy by simulation and write the figures for this run.

    Every step is best-effort: policy synthesis is the expensive part of a run and its result
    is only persisted when --save_checkpoint is set, so a failure while validating or plotting
    is logged and the remaining steps still run rather than discarding the run.

    :param args: Argument namespace.
    :param stamp: Run timestamp.
    :param model: Benchmark model.
    :param partition: Partition of the continuous state space.
    :param V: Value function over the abstraction states.
    :param policy: Action index per state.
    :param policy_inputs: Control input per state, from extract_policy_inputs.
    :param iterations: Number of Monte Carlo simulations to run.
    :param num_traces: Number of simulated traces to draw in the trace plots.
    :return: List of descriptions of the steps that failed (empty if all succeeded).
    '''

    from core.plotting.heatmap import heatmap
    from core.plotting.traces import plot_traces, plot_traces_3d
    from core.validate.simulate import MonteCarloSim

    failures = []

    sim = _guard('Monte Carlo simulation', failures, MonteCarloSim,
                 model, partition, policy, policy_inputs, model.x0,
                 verbose=False, iterations=iterations)
    if sim is not None:
        logger.info('Empirical satisfaction probability: %s', sim.results['satprob'])

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

    _guard('trace plot', failures, plot_traces,
           args, stamp, model.plot_dimensions, partition, model, traces,
           line=False, num_traces=num_traces, add_unsafe_box=False)

    # Models that are natural to show in 3D (e.g. the 6D drone) declare the three dimensions to use.
    dims_3d = getattr(model, 'plot_dimensions_3d', None)
    if dims_3d is not None:
        logger.info('Plotting traces in 3D (dimensions %s)...', list(dims_3d))
        _guard('3D trace plot', failures, plot_traces_3d,
               args, stamp, dims_3d, partition, model, traces,
               num_traces=num_traces, filename='traces_3d')

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
                  policy_inputs, *, iterations=100, num_traces=10, sim_horizon=1000):
    """Validate product policy and plot the initial-label DFA slice.

    Heatmaps display values and controls for z = delta(z0,L(s)) at each s.
    They are not plots of the full product-state policy.
    """
    from core.plotting.heatmap import heatmap
    from core.plotting.traces import plot_traces, plot_traces_3d
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

    _guard('sc-LTL initial-state satisfaction heatmap', failures, heatmap,
           args, stamp, idx_show=model.plot_dimensions,
           partition=partition, results=V_initial, filename='scLTL_satprob_initial',
           overlay=lambda ax: draw_regions_of_interest(ax, model, model.plot_dimensions))
    _guard('sc-LTL initial-state control heatmap', failures, heatmap,
           args, stamp, idx_show=model.plot_dimensions,
           partition=partition, results=inputs_initial[:, 0], filename='scLTL_policy_initial')

    if sim is not None:
        traces = sim.results['traces']
        _guard('sc-LTL trace plot', failures, plot_traces,
               args, stamp, model.plot_dimensions, partition, model, traces,
               line=False, num_traces=num_traces, add_unsafe_box=False,
               labeled_regions=True)
        dims_3d = getattr(model, 'plot_dimensions_3d', None)
        if dims_3d is not None:
            _guard('sc-LTL 3D trace plot', failures, plot_traces_3d,
                   args, stamp, dims_3d, partition, model, traces,
                   num_traces=num_traces, filename='scLTL_traces_3d',
                   labeled_regions=True)
    return failures