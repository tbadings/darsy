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
from core.abstraction.imdp.robust_vi_2layer_dfa import (
    encode_imdp_labels, robust_scLTL_synthesis_2layer_jax, initial_product_state)
from core.automaton.dfa import scLTL_to_DFA
from core.options import parse_arguments
from core.pipeline import (
    configure_run,
    extract_policy_inputs,
    finish,
    prepare_abstraction,
    report,
    report_scLTL,
    is_reach_avoid,
    is_scLTL,
)

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


def attach_region_data(imdp, model, partition):
    """Store geometric region memberships and labels, independently of a DFA."""
    imdp.states_of_interest = {
        r: np.asarray(mask, dtype=bool).copy()
        for r, mask in partition.states_of_interest.items()
    }
    imdp.partial_region_cells = {
        r: np.asarray(mask, dtype=bool).copy()
        for r, mask in partition.partial_region_cells.items()
    }
    imdp.region_labels = {r: set(labels) for r, labels in partition.region_labels.items()}
    imdp.atomic_propositions = list(partition.atomic_propositions)
    return imdp


def solve_2imdp_scLTL(args, imdp, s0, formula):
    """Return (lower, upper, global-action policy, dfa, initial z)."""
    if args.solver != 'jax':
        raise ValueError('sc-LTL 2IMDP synthesis requires --solver jax')
    dfa = scLTL_to_DFA(formula)
    labels = encode_imdp_labels(imdp, dfa)
    enabled = np.asarray([
        int(s) for s in imdp.states
        if len(imdp.A_id.get(int(s), [])) > 0
    ], dtype=np.int32)
    max_actions = max((len(imdp.A_id[int(s)]) for s in enabled), default=0)
    data_prep = preprocess_2imdp(imdp, enabled, max_actions)
    with jax.default_device(args.rvi_device):
        lower, upper, local_policy = robust_scLTL_synthesis_2layer_jax(
            data_prep, dfa, labels, epsilon=1e-6, max_iter=10000
        )
    policy = np.full_like(local_policy, -1)
    for s in enabled:
        for z in range(policy.shape[1]):
            a = int(local_policy[s, z])
            if a >= 0:
                policy[s, z] = int(imdp.A_id[int(s)][a])
    _, z0 = initial_product_state(s0, dfa, labels)
    return lower, upper, policy, dfa, z0, labels

if __name__ == '__main__':

    args = parse_arguments()

    print("Batch size:", args.batch_size)

    # --------------------------------------------------
    # Configure run and construct abstraction
    # --------------------------------------------------

    stamp = configure_run(args)

    model, partition, abstraction = prepare_abstraction(
        args,
        stamp,
        '2IMDP',
        build_2imdp
    )

    # Attach region information, including when loading
    # an abstraction from a checkpoint.
    attach_region_data(abstraction, model, partition)

    # --------------------------------------------------
    # Export 2IMDP to MATLAB
    # --------------------------------------------------

    # Export to MATLAB (debug only)
    # save_2imdp_matlab(
    #     abstraction,
    #     filename="results/results_2imdp.mat",
    #     max_iter=10000,
    #     tol_VI=1e-6,
    #     )

    # --------------------------------------------------
    # Initial abstract state
    # --------------------------------------------------

    s0, exists = partition.x2state(model.x0)

    if not exists:
        raise ValueError(
            "Initial continuous state is outside the partition."
        )

    # --------------------------------------------------
    # Determine synthesis method
    # --------------------------------------------------

    formula = getattr(model, "specification", None)

    if formula is None:
        raise ValueError(
            "The benchmark does not define a specification."
        )

    reach_avoid = is_reach_avoid(formula)

    logger.info(
        "=== 2IMDP synthesis: %s ===",
        "Reach-avoid" if reach_avoid else "sc-LTL"
    )

    start = time.time()

    # ==================================================
    # CASE 1: REACH-AVOID SYNTHESIS
    # ==================================================

    if reach_avoid:

        logger.info(
            "Reach-avoid specification detected. "
            "Using standard robust value iteration."
        )

        V_lower, V_upper, policy = solve_2imdp(
            args,
            abstraction,
            s0
        )

        logger.info(
            "Initial state s0=%d: lower=%.6f, upper=%.6f",
            s0,
            V_lower[s0],
            V_upper[s0]
        )

        # Convert action indices to continuous inputs
        policy_inputs = extract_policy_inputs(
            args,
            partition,
            policy,
            abstraction.nr_states
        )

        # Save synthesis results
        np.savez_compressed(
            args.output_dir / "reach_avoid_synthesis.npz",
            V_lower=V_lower,
            V_upper=V_upper,
            policy=policy,
            initial_state=s0
        )

        # Monte Carlo simulation and plotting
        report(
            args,
            stamp,
            model,
            partition,
            V_lower,
            policy,
            policy_inputs
        )

    # ==================================================
    # CASE 2: sc-LTL SYNTHESIS
    # ==================================================

    else:

        logger.info(
            "General sc-LTL specification detected. "
            "Constructing DFA and product 2IMDP."
        )

        # Construct DFA and synthesize product policy
        V_lower, V_upper, policy, dfa, z0, labels = (
            solve_2imdp_scLTL(
                args,
                abstraction,
                s0,
                formula
            )
        )

        logger.info(
            "Initial product state (s,z)=(%d,%d): "
            "lower=%.6f, upper=%.6f",
            s0,
            z0,
            V_lower[s0, z0],
            V_upper[s0, z0]
        )

        # Save synthesis results
        np.savez_compressed(
            args.output_dir / "product_synthesis.npz",
            V_lower=V_lower,
            V_upper=V_upper,
            policy=policy,
            initial_state=s0,
            initial_dfa_state=z0,
            dfa_transitions=dfa.transitions,
            dfa_accepting=dfa.accepting_states,
            atomic_propositions=np.asarray(
                dfa.atomic_propositions
            )
        )

        # Compute the actual control imputs implemented by the product policy
        policy_inputs = extract_policy_inputs(
            args,
            partition,
            policy,
            abstraction.nr_states
        )
        

        # Monte Carlo simulation and plotting
        report_scLTL(
            args=args,
            stamp=stamp,
            model=model,
            partition=partition,
            V_lower=V_lower,
            policy=policy,
            dfa=dfa,
            labels=labels,
            policy_inputs=policy_inputs
        )

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    logger.info(
        "<<< 2IMDP synthesis completed (took %.3f sec.) >>>",
        time.time() - start
    )

    finish(args)