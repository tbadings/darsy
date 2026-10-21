"""Build a two-layer interval MDP and synthesize a robust reachability policy.

Run with the same benchmark flags as RunIMDP.py, using --solver jax.
"""
import logging
import time

import jax
import numpy as np

from core.abstraction.imdp.matlab_2layer_adapter import (
    InitialCondition, SolverOpts, UserParam)
from core.abstraction.imdp.robust_vi_2layer import (
    preprocess_2imdp, robust_max_reach_probability_2layer_jax)
from core.options import parse_arguments
from core.pipeline import configure_run, extract_policy_inputs, finish, prepare_abstraction, report
from RunIMDP import build_imdp
# from export_2imdp_matlab import save_2imdp_matlab

logger = logging.getLogger(__name__)


def build_2imdp(args, model, partition, s_init):
    """Reuse the shared fine/coarse interval abstraction built by RunIMDP."""
    return build_imdp(args, model, partition, s_init, coarse_factor=args.coarse_factor, use_coarse=True)


def solve_2imdp(args, imdp, s0):
    if args.solver != 'jax':
        raise ValueError('The 2I-MDP solver currently supports only --solver jax.')
    if args.policy_iteration:
        logger.warning('2I-MDP solver uses value iteration; --policy_iteration is ignored.')

    ns = imdp.nr_states
    enabled = np.array([s for s in imdp.states
                        if not imdp.goal_regions[s] and not imdp.critical_regions[s]
                        and len(imdp.A_id.get(int(s), [])) > 0], dtype=np.int32)
    target = np.flatnonzero(imdp.goal_regions).astype(np.int32)
    user_param = UserParam(max_iter=10000, Ns=ns-1,
                           Na=max((len(imdp.A_id.get(int(s), [])) for s in enabled), default=0),
                           S_target=target, S_not_terminal=enabled)
    lower = np.zeros(ns, dtype=np.float32)
    lower[target] = 1.0
    initial = InitialCondition(V_lower=lower, V_upper=lower.copy(),
                               sigma=np.zeros(ns, dtype=np.int32))
    data_prep = preprocess_2imdp(imdp, enabled, user_param.Na)
    with jax.default_device(args.rvi_device):
        V_lower, V_upper, local_policy = robust_max_reach_probability_2layer_jax(
            user_param, initial, None, SolverOpts(tol_VI=1e-6),
            data_prep=data_prep)

    # The solver returns local action indices, while the run pipeline expects
    # global action labels (indices into partition.regions['actions']).
    policy = np.full(ns, -1, dtype=np.int32)
    for s in enabled:
        policy[s] = imdp.A_id[int(s)][int(local_policy[s])]
    return V_lower, V_upper, policy


if __name__ == '__main__':
    args = parse_arguments()

    print("Batch size:", args.batch_size)

    stamp = configure_run(args)
    model, partition, abstraction = prepare_abstraction(args, stamp, '2IMDP', build_2imdp)

    # Export to MATLAB (debug only)
    # save_2imdp_matlab(
    #     abstraction,
    #     filename="results/results.mat",
    #     max_iter=10000,
    #     tol_VI=1e-6,
    #     )

    logger.info('=== 2I-MDP policy synthesis (solver=jax) ===')
    start = time.time()
    s0 = partition.x2state(model.x0)[0]
    V_lower, V_upper, policy = solve_2imdp(args, abstraction, s0)
    logger.info('<<< 2I-MDP synthesis done (took %.3f sec.) >>>', time.time()-start)
    logger.info('Initial state s0=%d: lower=%.6f, upper=%.6f',
                s0, V_lower[s0], V_upper[s0])

    policy_inputs = extract_policy_inputs(args, partition, policy, abstraction.nr_states)
    report(args, stamp, model, partition, V_lower, policy, policy_inputs)
    finish(args)