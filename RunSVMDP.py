import logging
import time

import jax
import numpy as np

from core.abstraction.svmdp.dynprog import SVMDP_DP
from core.abstraction.svmdp.forward_reachability import RectangularForward
from core.abstraction.svmdp.successor_ids import make_box_to_ids
from core.abstraction.svmdp.svmdp import SVMDP
from core.options import parse_arguments
from core.pipeline import configure_run, extract_policy_inputs, finish, prepare_abstraction, report

logger = logging.getLogger(__name__)


def build_svmdp(args, model, partition, s_init):
    '''
    Build the SVMDP abstraction of the given model.

    :param args: Argument namespace.
    :param model: Benchmark model.
    :param partition: Partition of the continuous state space.
    :param s_init: State index of the initial continuous state (unused; kept for the shared
        abstraction-callback signature).
    :return: SVMDP abstraction.
    '''

    # Compute forward reachable sets and noise-shifted successor cell IDs.
    actions = RectangularForward(args=args, partition=partition, model=model)

    # All partition states have all actions enabled (rectangular partition),
    # so a single shared list of action ids describes every state.
    states = np.array(partition.regions['idxs'])
    A_id = list(range(actions.num_actions))

    # Recompose successor IDs on the fly in the DP from the compact boxes (frs_idx_lb/frs_idx_ub)
    # rather than materialising the [S, A, nc, prod(max_span)] ID array (tens of GB for 3-D models).
    box_to_ids = make_box_to_ids(max_span=actions.max_slice, wrap=model.wrap, partition=partition)

    return SVMDP(
        partition=partition,
        states=states,
        x0=model.x0,
        goal_regions=np.array(partition.goal['bools']),
        critical_regions=np.array(partition.critical['bools']),
        P_full=actions.frs_noise_probs,
        S_idx_lb=actions.frs_idx_lb,
        S_idx_ub=actions.frs_idx_ub,
        box_to_ids=box_to_ids,
        A_id=A_id,
        P_absorbing=model.noise.partition['remainder'],
    )


def solve_svmdp(args, svmdp, s0):
    '''
    Synthesize a robust policy on the SVMDP.

    :param args: Argument namespace.
    :param svmdp: SVMDP abstraction.
    :param s0: Initial state index.
    :return: Tuple of (values, policy).
    '''

    with jax.default_device(args.rvi_device):
        return SVMDP_DP(
            args=args,
            svmdp=svmdp,
            s0=s0,
            max_iterations=10000,
            epsilon=1e-6,
            RND_SWEEPS=True,
            BATCH_SIZE=1000,
            policy_iteration=args.policy_iteration,
            prune_states=False,
        )


if __name__ == '__main__':
    args = parse_arguments()
    stamp = configure_run(args)

    if args.solver != 'jax':
        # --solver is shared with RunIMDP.py, but the SVMDP abstraction has only the JAX backend.
        logger.warning("solver='%s' is not available for the SVMDP abstraction; using the JAX solver instead.",
                       args.solver)

    model, partition, svmdp = prepare_abstraction(args, stamp, 'SVMDP', build_svmdp)

    # %% Run value iteration on the SVMDP

    logger.info('=== SVMDP policy synthesis ===')
    t = time.time()
    s0 = partition.x2state(model.x0)[0]
    V, policy = solve_svmdp(args, svmdp, s0)
    logger.info('<<< SVMDP policy synthesis done (took %.3f sec.) >>>\n', time.time() - t)

    logger.info('Value in initial state s0=%d: %.6f\n', s0, V[s0])

    # %% Simulations and plots

    policy_inputs = extract_policy_inputs(args, partition, policy, svmdp.nr_states)
    report(args, stamp, model, partition, V, policy, policy_inputs)

    finish(args)
