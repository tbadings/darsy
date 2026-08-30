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
from core.abstraction.imdp.probability_intervals import compute_probability_intervals
from core.abstraction.imdp.forward_reachability import RectangularForward
from core.abstraction.imdp.imdp import IMDP
from core.abstraction.imdp.rvi_jax import RVI_JAX
from core.abstraction.imdp.rvi_storm import RVI_STORM
from core.options import parse_arguments
from core.abstraction.partition import DensePartition
from core.jax_config import configure_jax
from core.utils import configure_logging, add_file_handler

if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    configure_jax(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.load_checkpoint:
        # --- Load IMDP from checkpoint ---
        ckpt_path = Path(args.load_checkpoint)
        logger.info('Loading checkpoint from %s', ckpt_path)
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        model = ckpt['model']
        partition = ckpt['partition']
        imdp = ckpt['imdp']
        args.model = ckpt['args'].model

        run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = run_output_dir
        add_file_handler(run_output_dir, stamp)
        logger.info('Run %s | model=%s (from checkpoint)', stamp, args.model)
        logger.info('Output directory: %s', run_output_dir)

        logger.info('\n=== IMDP loaded from checkpoint: %s ===', ckpt_path)
    else:
        t = time.time()

        # --- Build IMDP from scratch ---
        run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = run_output_dir
        add_file_handler(run_output_dir, stamp)
        logger.info('Run %s | model=%s | noise=%s | batch=%d', stamp, args.model, args.noise_distr, args.batch_size)
        logger.info('Output directory: %s', run_output_dir)

        logger.info('\n=== Generate IMDP from scratch ===')
        logger.debug('Arguments: %s', vars(args))

        model = benchmarks.create_model(args)

        logger.info('<<< Generating model took %.3f sec. >>>\n', time.time() - t)
        t = time.time()

        # Create partition of the continuous state space into convex polytope
        partition = DensePartition(model=model)

        s_init, s_init_exists = partition.x2state(model.x0)
        if not s_init_exists:
            raise ValueError(f"Initial state x0={model.x0} is not in the partition.")

        # Create actions based on forward reachable sets
        actions = RectangularForward(args=args, partition=partition, model=model)

        # Transition probability intervals for every state-action pair.
        P_full, S_id, A_id, P_absorbing = compute_probability_intervals(
            args=args,
            model=model,
            partition=partition,
            actions=actions,
            vectorized=True,
            debug_state=s_init,
        )

        imdp = IMDP(
            partition=partition,
            states=np.array(partition.regions['idxs']),
            x0=model.x0,
            goal_regions=np.array(partition.goal['bools']),
            critical_regions=np.array(partition.critical['bools']),
            P_full=P_full,
            S_id=S_id,
            A_id=A_id,
            P_absorbing=P_absorbing,
        )

        del actions

        logger.info('Initial state x0=%s → state index %d\n', model.x0, s_init)
        logger.info('<<< Generating IMDP abstraction took %.3f sec. >>>\n', time.time() - t)

        if args.save_checkpoint:
            # Save checkpoint (strip JAX runtime objects that can't be pickled)
            args_to_save = copy.copy(args)
            del args_to_save.rvi_device
            del args_to_save.jax_key
            ckpt_path = args.output_dir / 'checkpoint.pkl'
            logger.info('Saving checkpoint to %s', ckpt_path)
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'model': model, 'partition': partition, 'imdp': imdp, 'args': args_to_save}, f)
            logger.info('Checkpoint saved.\n')

    # %% Run dynamic programming on the IMDP

    logger.info('=== IMDP policy synthesis (solver=%s) ===', args.solver)
    t = time.time()
    if args.solver == 'jax':
        with jax.default_device(args.rvi_device):
            V, policy = RVI_JAX(
                args=args,
                imdp=imdp,
                s0=partition.x2state(model.x0)[0],
                max_iterations=10000,
                epsilon=1e-6,
                RND_SWEEPS=True,
                BATCH_SIZE=10000,
                policy_iteration=args.policy_iteration,
            )
    else:
        V, policy = RVI_STORM(
            args=args,
            imdp=imdp,
        )
    logger.info('<<< IMDP policy synthesis done (took %.3f sec.) >>>\n', time.time() - t)

    s0 = partition.x2state(model.x0)[0]
    logger.info('Value in initial state s0=%d: %.6f\n', s0, V[s0])

    # %% Extract policy inputs

    float_dtype = getattr(args, 'floatprecision', np.float32)
    # Define concrete policy (but exclude final IMDP state, which is absorbing and has no actions)
    actions_np = np.array(partition.regions['actions'])
    # DensePartition stores (1, num_actions, action_dim); broadcast to all states.
    if actions_np.shape[0] == 1:
        actions_np = np.broadcast_to(actions_np, (imdp.nr_states - 1, *actions_np.shape[1:]))
    policy_inputs = np.full(
        (imdp.nr_states - 1, actions_np.shape[2]), fill_value=float('nan'), dtype=float_dtype
    )
    mask = policy[:-1] >= 0
    policy_inputs[mask] = actions_np[mask, policy[:-1][mask]]

    # %% Simulations and plots

    from core.validate.simulate import MonteCarloSim
    from core.plotting.traces import plot_traces
    from core.plotting.heatmap import heatmap
    from core.plotting.traces import plot_traces_3d

    sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=100)
    logger.info('Empirical satisfaction probability: %s', sim.results['satprob'])

    heatmap(
        args, stamp, idx_show=model.plot_dimensions,
        partition=partition, results=V, filename='heatmap_satprob',
    )
    heatmap(
        args, stamp, idx_show=model.plot_dimensions,
        partition=partition, results=policy_inputs[:, 0], filename='policy',
    )
    plot_traces(
        args, stamp, model.plot_dimensions, partition, model,
        sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,
    )

    if args.model.startswith('Drone6D'):
        print('Plot Drone6D traces in 3D...')
        plot_traces_3d(
            args, stamp, [0, 2, 4], partition, model,
            sim.results['traces'], num_traces=10, filename="traces_3d",
        )

    if args.model == 'Pendulum':
        print('Plot Pendulum gif...')
        model.plot_trajectory_gif(
            np.array(sim.results['traces'][0]['x'])[:, 0],
            filename=str(args.output_dir / f'pendulum_{stamp}.gif'),
        )

    if args.model == 'MountainCar':
        print('Plot MountainCar gif...')
        model.plot_trajectory_gif(
            np.array(sim.results['traces'][0]['x'])[:, 0],
            filename=str(args.output_dir / f'mountaincar_{stamp}.gif'),
        )

    if args.model == 'CartPole':
        print('Plot CartPole gif...')
        model.plot_trajectory_gif(
            np.array(sim.results['traces'][0]['x'])[:, [0, 2]],
            filename=str(args.output_dir / f'cartpole_{stamp}.gif'),
        )

    logger.info('\n=== Run finished. Output directory: %s ===', run_output_dir)
    logger.info('====================\n')