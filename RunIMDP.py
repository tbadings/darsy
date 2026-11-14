"""Build an IMDP and synthesize reach-avoid, safety, or scLTL policies.

All specifications are solved by the JAX robust dynamic programming driver in
core/abstraction/imdp/robust_dp.py.
"""
import logging
import time
from argparse import Namespace
from typing import cast

import jax
import numpy as np

from core.abstraction.imdp.forward_reachability import RectangularForward
from core.abstraction.imdp.imdp import IMDP
from core.abstraction.imdp.probability_intervals import compute_probability_intervals
from core.abstraction.imdp.results import ScltlSynthesisResult, SynthesisResult
from core.abstraction.imdp.robust_dp import (
    encode_imdp_labels,
    initial_product_state,
    solve_reach_avoid,
    solve_safety,
    solve_scltl,
)
from core.abstraction.model import attach_region_data
from core.abstraction.partition import CoarsePartition, DensePartition
from core.automaton.dfa import scltl_to_dfa
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

logger = logging.getLogger(__name__)


def build_imdp(
    args: Namespace, model: Benchmark, partition: DensePartition, s_init: int,
    coarse_factor: int = 1, use_coarse: bool = False,
) -> IMDP:
    """Build fine intervals and optional coarse intervals, then attach region labels."""
    actions = RectangularForward(
        args=args, partition=partition, model=model, coarse_factor=coarse_factor
    )
    coarse_partition = CoarsePartition(partition, coarse_factor)
    intervals = compute_probability_intervals(
        args=args,
        model=model,
        partition=partition,
        actions=actions,
        coarse_partition=coarse_partition,
        debug_state=s_init,
        use_coarse=use_coarse,
    )
    imdp = IMDP(
        states=np.array(partition.regions['idxs']),
        goal_regions=np.array(partition.goal['bools']),
        critical_regions=np.array(partition.critical['bools']),
        P_full=intervals.fine_bounds,
        S_id=intervals.successor_ids,
        A_id=intervals.action_ids,
        P_absorbing=intervals.absorbing_bounds,
        coarse_partition=coarse_partition,
        Q_id=intervals.coarse_successor_ids,
        P_coarse=intervals.coarse_bounds,
    )
    return attach_region_data(imdp, model, partition)


def solve_imdp(args: Namespace, imdp: IMDP, s0: int) -> SynthesisResult:
    """Solve reach-avoid on the configured device and horizon."""
    with jax.default_device(args.rvi_device):
        return solve_reach_avoid(
            imdp, initial_state=s0, horizon=args.horizon, float_dtype=args.floatprecision,
        )


def solve_imdp_scltl(args: Namespace, imdp: IMDP, s0: int, formula: str) -> ScltlSynthesisResult:
    """Translate with Spot and solve unbounded scLTL using fine intervals."""
    dfa = scltl_to_dfa(formula)
    labels = encode_imdp_labels(imdp, dfa)
    with jax.default_device(args.rvi_device):
        result = solve_scltl(
            imdp, dfa, labels, initial_state=s0, float_dtype=args.floatprecision,
        )
    _, z0 = initial_product_state(s0, dfa, labels)
    return ScltlSynthesisResult(
        lower=result.lower, upper=result.upper, policy=result.policy,
        dfa=dfa, initial_dfa_state=z0, labels=labels,
    )


def main() -> None:
    args = parse_arguments()
    args.coarse_factor = 1
    stamp = configure_run(args)
    formula = None
    # prepare_abstraction constructs the model internally; inspect the
    # factor-1 coarse data after inspecting the model in the build callback.
    def build_for_spec(args: Namespace, model: Benchmark, partition: DensePartition, s_init: int) -> IMDP:
        nonlocal formula
        formula = getattr(model, 'specification', None)
        if formula is None:
            raise ValueError('The benchmark does not define a specification.')
        return build_imdp(
            args, model, partition, s_init,
            coarse_factor=1, use_coarse=False,
        )

    model, partition, imdp = prepare_abstraction(
        args, stamp, 'IMDP', build_for_spec
    )
    formula = getattr(model, 'specification', None)
    if formula is None:
        raise ValueError('The benchmark does not define a specification.')
    s0, exists = partition.x2state(model.x0)
    if not exists:
        raise ValueError('Initial continuous state is outside the partition.')
    reach_avoid = is_reach_avoid(formula)
    safety = is_safety(formula)
    logger.info(
        '=== IMDP %s synthesis ===',
        'safety' if safety else ('reach-avoid' if reach_avoid else 'scLTL'),
    )
    t = time.time()

    if safety:
        safety_model = cast(SafetyBenchmark, model)
        mask = safe_state_mask(partition, safety_model.safe_set)
        upper_mask = safe_state_mask(partition, safety_model.safe_set, optimistic=True)
        with jax.default_device(args.rvi_device):
            result = solve_safety(imdp, mask, upper_mask, args.horizon, float_dtype=args.floatprecision)
        record_synthesis(args, t, result.lower[args.horizon, s0], result.upper[args.horizon, s0])
        save_safety_results(args, s0, result.lower, result.upper, result.policy)
        report_safety(args, stamp, safety_model, partition, result.lower, result.upper, result.policy)
    elif reach_avoid:
        result = solve_imdp(args, imdp, s0)
        at = s0 if args.horizon is None else (args.horizon, s0)  # with all steps remaining
        record_synthesis(args, t, result.lower[at], result.upper[at])
        logger.info('Initial state s0=%d: value=%.6f', s0, result.lower[at])
        logger.info('Initial state s0=%d: upper bound=%.6f', s0, result.upper[at])
        policy_inputs = extract_policy_inputs(
            args, partition, result.policy, imdp.nr_states, time_indexed=args.horizon is not None
        )
        report(args, stamp, model, partition, result.lower, result.policy, policy_inputs)
    else:
        product = solve_imdp_scltl(args, imdp, s0, formula)
        z0 = product.initial_dfa_state
        record_synthesis(args, t, product.lower[s0, z0], product.upper[s0, z0])
        logger.info('Initial product state (%d,%d): lower=%.6f, upper=%.6f',
                    s0, z0, product.lower[s0, z0], product.upper[s0, z0])
        np.savez_compressed(
            args.output_dir / 'product_synthesis.npz',
            V_lower=product.lower, V_upper=product.upper, policy=product.policy,
            initial_state=s0, initial_dfa_state=z0,
            dfa_transitions=product.dfa.transitions,
            dfa_accepting=product.dfa.accepting_states,
            atomic_propositions=np.asarray(product.dfa.atomic_propositions),
        )
        policy_inputs = extract_policy_inputs(
            args, partition, product.policy, imdp.nr_states
        )
        report_scltl(
            args=args, stamp=stamp, model=model, partition=partition,
            V_lower=product.lower, policy=product.policy, dfa=product.dfa,
            labels=product.labels, policy_inputs=policy_inputs,
        )

    finish(args)


if __name__ == '__main__':
    main()
