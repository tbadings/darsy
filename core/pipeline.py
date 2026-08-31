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

    t = time.time()

    # Create partition of the continuous state space into convex polytope
    partition = DensePartition(model=model)

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
    '''
    Map the action index chosen for each state to the concrete control input.

    :param args: Argument namespace.
    :param partition: Partition of the continuous state space.
    :param policy: Action index per state, with -1 marking states without an enabled action.
    :param nr_states: Number of abstraction states, including the absorbing state.
    :return: Control input per state (shape: [nr_states - 1, action_dim]); states without an
        enabled action hold NaN. The final abstraction state is absorbing and has no actions,
        so it is excluded.
    '''

    float_dtype = getattr(args, 'floatprecision', np.float32)

    actions_np = np.array(partition.regions['actions'])
    # DensePartition stores (1, num_actions, action_dim); broadcast to all states.
    if actions_np.shape[0] == 1:
        actions_np = np.broadcast_to(actions_np, (nr_states - 1, *actions_np.shape[1:]))

    policy_inputs = np.full(
        (nr_states - 1, actions_np.shape[2]), fill_value=float('nan'), dtype=float_dtype
    )
    mask = policy[:-1] >= 0
    policy_inputs[mask] = actions_np[mask, policy[:-1][mask]]

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
