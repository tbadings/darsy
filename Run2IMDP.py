"""Build a two-layer interval MDP and synthesize a robust reachability policy.

Run with the same benchmark flags as RunIMDP.py, using --solver jax.
"""
import logging
import time

import jax
import numpy as np

from core.abstraction.imdp.robust_dp import (
    encode_imdp_labels, initial_product_state, solve_reach_avoid, solve_safety, solve_scLTL)
from core.automaton.dfa import scLTL_to_DFA
from core.options import parse_arguments
from core.pipeline import (
    configure_run,
    extract_policy_inputs,
    finish,
    prepare_abstraction,
    record_synthesis,
    report,
    report_scLTL,
    is_reach_avoid,
    is_safety,
    safe_state_mask,
    report_safety,
    save_safety_results,
)

from RunIMDP import build_imdp

logger = logging.getLogger(__name__)


def build_2imdp(args, model, partition, s_init):
    """Reuse the shared fine/coarse interval abstraction built by RunIMDP."""
    return build_imdp(args, model, partition, s_init, coarse_factor=args.coarse_factor, use_coarse=True)


def solve_2imdp(args, imdp, s0):
    """Return lower and upper values, and a global-action policy, indexed by remaining steps with a horizon."""
    if args.solver != 'jax':
        raise ValueError('The 2I-MDP solver currently supports only --solver jax.')
    with jax.default_device(args.rvi_device):
        return solve_reach_avoid(
            imdp, two_layer=True, initial_state=s0, horizon=args.horizon,
            policy_iteration=args.policy_iteration, satprob=args.satprob,
            float_dtype=args.floatprecision)


def solve_2imdp_scLTL(args, imdp, s0, formula):
    """Return (lower, upper, global-action policy, dfa, initial z)."""
    if args.solver != 'jax':
        raise ValueError('sc-LTL 2IMDP synthesis requires --solver jax')
    dfa = scLTL_to_DFA(formula)
    labels = encode_imdp_labels(imdp, dfa)
    with jax.default_device(args.rvi_device):
        lower, upper, policy = solve_scLTL(
            imdp, dfa, labels, two_layer=True, initial_state=s0,
            policy_iteration=args.policy_iteration, satprob=args.satprob,
            float_dtype=args.floatprecision)
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
    safety = is_safety(formula)

    logger.info(
        '=== 2IMDP %s synthesis (solver=%s) ===',
        'safety' if safety else ('reach-avoid' if reach_avoid else 'scLTL'),
        args.solver
    )

    start = time.time()

    # ==================================================
    # CASE 1: REACH-AVOID SYNTHESIS
    # ==================================================

    if safety:
        if args.solver != 'jax':
            raise ValueError('Finite-horizon safety currently requires --solver jax')
        mask = safe_state_mask(partition, model.safe_set)
        upper_mask = safe_state_mask(partition, model.safe_set, optimistic=True)
        with jax.default_device(args.rvi_device):
            V_lower, V_upper, policy = solve_safety(abstraction, mask, upper_mask, args.horizon,
                                                    two_layer=True, float_dtype=args.floatprecision)
        record_synthesis(args, start, V_lower[args.horizon, s0], V_upper[args.horizon, s0])
        save_safety_results(args, s0, V_lower, V_upper, policy)
        report_safety(args, stamp, model, partition, V_lower, V_upper, policy)

    elif reach_avoid:

        logger.info(
            "Reach-avoid specification detected. "
            "Using standard robust value iteration."
        )

        V_lower, V_upper, policy = solve_2imdp(
            args,
            abstraction,
            s0
        )

        # With a horizon, report the values with all steps remaining
        at = s0 if args.horizon is None else (args.horizon, s0)
        record_synthesis(args, start, V_lower[at], V_upper[at])

        logger.info(
            "Initial state s0=%d: lower=%.6f, upper=%.6f",
            s0,
            V_lower[at],
            V_upper[at]
        )

        # Convert action indices to continuous inputs
        policy_inputs = extract_policy_inputs(
            args,
            partition,
            policy,
            abstraction.nr_states,
            time_indexed=args.horizon is not None
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
        record_synthesis(args, start, V_lower[s0, z0], V_upper[s0, z0])

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

    finish(args)