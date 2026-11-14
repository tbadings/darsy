"""Build a two-layer interval MDP and synthesize a robust reachability policy.

Run with the same benchmark flags as RunIMDP.py.
"""
import logging
import time
from argparse import Namespace
from typing import cast

import jax
import numpy as np

from core.abstraction.imdp.imdp import IMDP
from core.abstraction.imdp.results import ScltlSynthesisResult, SynthesisResult
from core.abstraction.imdp.robust_dp import (
    encode_imdp_labels, initial_product_state, solve_reach_avoid, solve_safety, solve_scltl)
from core.automaton.dfa import scltl_to_dfa
from core.abstraction.partition import DensePartition
from core.interfaces import Benchmark, SafetyBenchmark
from core.options import parse_arguments
from core.pipeline import (
    configure_run,
    extract_policy_inputs,
    finish,
    prepare_abstraction,
    record_synthesis,
    report,
    report_scltl,
    is_reach_avoid,
    is_safety,
    safe_state_mask,
    report_safety,
    save_safety_results,
)

from RunIMDP import build_imdp

logger = logging.getLogger(__name__)


def build_2imdp(args: Namespace, model: Benchmark, partition: DensePartition, s_init: int) -> IMDP:
    """Reuse the shared fine/coarse interval abstraction built by RunIMDP."""
    return build_imdp(args, model, partition, s_init, coarse_factor=args.coarse_factor, use_coarse=True)


def solve_2imdp(args: Namespace, imdp: IMDP, s0: int) -> SynthesisResult:
    """Solve reach-avoid using both interval layers and the configured horizon."""
    with jax.default_device(args.rvi_device):
        return solve_reach_avoid(
            imdp, two_layer=True, initial_state=s0, horizon=args.horizon,
            float_dtype=args.floatprecision)


def solve_2imdp_scltl(args: Namespace, imdp: IMDP, s0: int, formula: str) -> ScltlSynthesisResult:
    """Translate with Spot and solve unbounded scLTL using both interval layers."""
    dfa = scltl_to_dfa(formula)
    labels = encode_imdp_labels(imdp, dfa)
    with jax.default_device(args.rvi_device):
        result = solve_scltl(
            imdp, dfa, labels, two_layer=True, initial_state=s0,
            float_dtype=args.floatprecision)
    _, z0 = initial_product_state(s0, dfa, labels)
    return ScltlSynthesisResult(
        lower=result.lower, upper=result.upper, policy=result.policy,
        dfa=dfa, initial_dfa_state=z0, labels=labels,
    )

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
        '=== 2IMDP %s synthesis ===',
        'safety' if safety else ('reach-avoid' if reach_avoid else 'scLTL'),
    )
    start = time.time()

    # ==================================================
    # CASE 1: REACH-AVOID SYNTHESIS
    # ==================================================
    if safety:
        safety_model = cast(SafetyBenchmark, model)
        mask = safe_state_mask(partition, safety_model.safe_set)
        upper_mask = safe_state_mask(partition, safety_model.safe_set, optimistic=True)
        with jax.default_device(args.rvi_device):
            result = solve_safety(abstraction, mask, upper_mask, args.horizon,
                                  two_layer=True, float_dtype=args.floatprecision)
        record_synthesis(args, start, result.lower[args.horizon, s0], result.upper[args.horizon, s0])
        save_safety_results(args, s0, result.lower, result.upper, result.policy)
        report_safety(args, stamp, safety_model, partition, result.lower, result.upper, result.policy)
    elif reach_avoid:
        logger.info(
            "Reach-avoid specification detected. "
            "Using standard robust value iteration."
        )
        result = solve_2imdp(
            args,
            abstraction,
            s0
        )

        # With a horizon, report the values with all steps remaining
        at = s0 if args.horizon is None else (args.horizon, s0)
        record_synthesis(args, start, result.lower[at], result.upper[at])
        logger.info(
            "Initial state s0=%d: lower=%.6f, upper=%.6f",
            s0,
            result.lower[at],
            result.upper[at]
        )

        # Convert action indices to continuous inputs
        policy_inputs = extract_policy_inputs(
            args,
            partition,
            result.policy,
            abstraction.nr_states,
            time_indexed=args.horizon is not None
        )

        # Save synthesis results
        np.savez_compressed(
            args.output_dir / "reach_avoid_synthesis.npz",
            V_lower=result.lower,
            V_upper=result.upper,
            policy=result.policy,
            initial_state=s0
        )

        # Monte Carlo simulation and plotting
        report(
            args,
            stamp,
            model,
            partition,
            result.lower,
            result.policy,
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
        product = (
            solve_2imdp_scltl(
                args,
                abstraction,
                s0,
                formula
            )
        )
        z0 = product.initial_dfa_state
        record_synthesis(args, start, product.lower[s0, z0], product.upper[s0, z0])
        logger.info(
            "Initial product state (s,z)=(%d,%d): "
            "lower=%.6f, upper=%.6f",
            s0,
            z0,
            product.lower[s0, z0],
            product.upper[s0, z0]
        )

        # Save synthesis results
        np.savez_compressed(
            args.output_dir / "product_synthesis.npz",
            V_lower=product.lower,
            V_upper=product.upper,
            policy=product.policy,
            initial_state=s0,
            initial_dfa_state=z0,
            dfa_transitions=product.dfa.transitions,
            dfa_accepting=product.dfa.accepting_states,
            atomic_propositions=np.asarray(
                product.dfa.atomic_propositions
            )
        )

        # Compute the actual control imputs implemented by the product policy
        policy_inputs = extract_policy_inputs(
            args,
            partition,
            product.policy,
            abstraction.nr_states
        )

        # Monte Carlo simulation and plotting
        report_scltl(
            args=args,
            stamp=stamp,
            model=model,
            partition=partition,
            V_lower=product.lower,
            policy=product.policy,
            dfa=product.dfa,
            labels=product.labels,
            policy_inputs=policy_inputs
        )

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------
    finish(args)
