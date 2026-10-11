import logging
import time

import jax
import numpy as np

from core.abstraction.imdp.forward_reachability import RectangularForward
from core.abstraction.imdp.imdp import IMDP
from core.abstraction.imdp.probability_intervals import compute_probability_intervals
from core.abstraction.imdp.rvi_jax import RVI_JAX
from core.abstraction.imdp.rvi_storm import RVI_STORM
from core.abstraction.partition import CoarsePartition
from core.options import parse_arguments
from core.pipeline import configure_run, extract_policy_inputs, finish, prepare_abstraction, report

logger = logging.getLogger(__name__)


def build_imdp(args, model, partition, s_init, coarse_factor=1, use_coarse=False):
    '''
    Build the IMDP abstraction of the given model.

    :param args: Argument namespace.
    :param model: Benchmark model.
    :param partition: Partition of the continuous state space.
    :param s_init: State index of the initial continuous state (used for debug output).
    :param coarse_factor: default is 1 to not affect standard IMDP.
    :param use_coarse: True to compute the second layer (2IMDP).
    :return: IMDP abstraction.
    '''

    # Create actions based on forward reachable sets
    actions = RectangularForward(args=args, partition=partition, model=model, coarse_factor=coarse_factor)

    # Coarse partition grouping fine cells into blocks
    coarse_partition = CoarsePartition(partition, coarse_factor)

    # Transition probability intervals for every state-action pair.
    P_full, S_id, A_id, P_absorbing, Q_id, P_coarse = compute_probability_intervals(
        args=args,
        model=model,
        partition=partition,
        actions=actions,
        coarse_partition=coarse_partition,
        debug_state=s_init,
        use_coarse=use_coarse,
    )

    return IMDP(
        partition=partition,
        states=np.array(partition.regions['idxs']),
        x0=model.x0,
        goal_regions=np.array(partition.goal['bools']),
        critical_regions=np.array(partition.critical['bools']),
        P_full=P_full,
        S_id=S_id,
        A_id=A_id,
        P_absorbing=P_absorbing,
        coarse_partition=coarse_partition,
        Q_id=Q_id,
        P_coarse=P_coarse,
    )


def solve_imdp(args, imdp, s0):
    '''
    Synthesize a robust policy on the IMDP with the selected solver backend.

    :param args: Argument namespace.
    :param imdp: IMDP abstraction.
    :param s0: Initial state index.
    :return: Tuple of (values, policy).
    '''

    if args.solver == 'jax':
        with jax.default_device(args.rvi_device):
            return RVI_JAX(
                args=args,
                imdp=imdp,
                s0=s0,
                max_iterations=10000,
                epsilon=1e-6,
                RND_SWEEPS=True,
                BATCH_SIZE=10000,
                policy_iteration=args.policy_iteration,
            )

    return RVI_STORM(args=args, imdp=imdp)


if __name__ == '__main__':
    args = parse_arguments()
    args.coarse_factor = 1
    stamp = configure_run(args)

    model, partition, imdp = prepare_abstraction(args, stamp, 'IMDP', build_imdp)

    # %% Run dynamic programming on the IMDP

    logger.info('=== IMDP policy synthesis (solver=%s) ===', args.solver)
    t = time.time()
    s0 = partition.x2state(model.x0)[0]
    V, policy = solve_imdp(args, imdp, s0)
    logger.info('<<< IMDP policy synthesis done (took %.3f sec.) >>>\n', time.time() - t)

    logger.info('Value in initial state s0=%d: %.6f\n', s0, V[s0])

    # %% Simulations and plots

    policy_inputs = extract_policy_inputs(args, partition, policy, imdp.nr_states)
    report(args, stamp, model, partition, V, policy, policy_inputs)

    finish(args)
