"""Build an IMDP and synthesize reach-avoid or scLTL policies.

For general scLTL, use a dedicated single-layer product-IMDP JAX solver.
"""
import logging
import time

import jax
import numpy as np

from core.abstraction.imdp.forward_reachability import RectangularForward
from core.abstraction.imdp.imdp import IMDP
from core.abstraction.imdp.probability_intervals import compute_probability_intervals
from core.abstraction.imdp.rvi_jax import RVI_JAX
from core.abstraction.imdp.rvi_storm import RVI_STORM
from core.abstraction.imdp.robust_vi_2layer_dfa import (
    encode_imdp_labels,
    initial_product_state,
)
from core.abstraction.imdp.rvi_jax_dfa import robust_scLTL_synthesis_imdp_jax
from core.abstraction.partition import CoarsePartition
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
)

logger = logging.getLogger(__name__)


def identify_states_of_interest(partition, regions):
    """Map each named ROI to a mask of fully contained abstract states."""
    lower_bounds = np.asarray(partition.regions['lower_bounds'])
    upper_bounds = np.asarray(partition.regions['upper_bounds'])
    n_states = partition.size
    states_of_interest = {}
    for r, rectangles in regions.items():
        contained = np.zeros(n_states, dtype=bool)
        for rectangle in np.asarray(rectangles, dtype=float):
            region_lb, region_ub = rectangle
            contained |= (
                np.all(lower_bounds >= region_lb, axis=1)
                & np.all(upper_bounds <= region_ub, axis=1)
            )
        states_of_interest[r] = contained
    return states_of_interest


def build_imdp(args, model, partition, s_init, coarse_factor=1, use_coarse=False):
    """Build fine IMDP intervals, optionally including factor-1 coarse data."""
    actions = RectangularForward(
        args=args, partition=partition, model=model, coarse_factor=coarse_factor
    )
    coarse_partition = CoarsePartition(partition, coarse_factor)
    P_full, S_id, A_id, P_absorbing, Q_id, P_coarse = compute_probability_intervals(
        args=args,
        model=model,
        partition=partition,
        actions=actions,
        coarse_partition=coarse_partition,
        debug_state=s_init,
        use_coarse=use_coarse,
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
        coarse_partition=coarse_partition,
        Q_id=Q_id,
        P_coarse=P_coarse,
    )
    # Prefer partition-provided masks/labels, as in Run2IMDP, when available.
    if hasattr(partition, 'states_of_interest'):
        imdp.states_of_interest = {
            r: np.asarray(mask, dtype=bool).copy()
            for r, mask in partition.states_of_interest.items()
        }
    else:
        imdp.states_of_interest = identify_states_of_interest(partition, model.regions)

    if hasattr(partition, 'partial_region_cells'):
        imdp.partial_region_cells = {
            r: np.asarray(mask, dtype=bool).copy()
            for r, mask in partition.partial_region_cells.items()
        }
    if hasattr(partition, 'region_labels'):
        imdp.region_labels = {
            r: set(labels) for r, labels in partition.region_labels.items()
        }
    else:
        imdp.region_labels = {
            r: set(model.labeling_function(r)) for r in model.regions
        }
    imdp.atomic_propositions = list(
        partition.atomic_propositions
        if hasattr(partition, 'atomic_propositions')
        else model.atomic_propositions
    )
    return imdp


def attach_region_data(imdp, model, partition):
    """Restore region information when loading an abstraction checkpoint."""
    if hasattr(partition, 'states_of_interest'):
        imdp.states_of_interest = {
            r: np.asarray(mask, dtype=bool).copy()
            for r, mask in partition.states_of_interest.items()
        }
    else:
        imdp.states_of_interest = identify_states_of_interest(partition, model.regions)
    if hasattr(partition, 'partial_region_cells'):
        imdp.partial_region_cells = {
            r: np.asarray(mask, dtype=bool).copy()
            for r, mask in partition.partial_region_cells.items()
        }
    if hasattr(partition, 'region_labels'):
        imdp.region_labels = {
            r: set(labels) for r, labels in partition.region_labels.items()
        }
    else:
        imdp.region_labels = {
            r: set(model.labeling_function(r)) for r in model.regions
        }
    imdp.atomic_propositions = list(
        partition.atomic_propositions
        if hasattr(partition, 'atomic_propositions')
        else model.atomic_propositions
    )
    return imdp


def solve_imdp(args, imdp, s0):
    """Existing reach-avoid synthesis, unchanged."""
    if args.solver == 'jax':
        with jax.default_device(args.rvi_device):
            return RVI_JAX(
                args=args, imdp=imdp, s0=s0,
                max_iterations=10000, epsilon=1e-6,
                RND_SWEEPS=True, BATCH_SIZE=10000,
                policy_iteration=args.policy_iteration,
            )
    return RVI_STORM(args=args, imdp=imdp)


def solve_imdp_scLTL(args, imdp, s0, formula):
    """Product synthesis using only single-layer fine probability intervals."""
    if args.solver != 'jax':
        raise ValueError('scLTL IMDP synthesis currently requires --solver jax.')
    if args.policy_iteration:
        logger.warning('scLTL synthesis uses value iteration; policy iteration ignored.')

    dfa = scLTL_to_DFA(formula)
    labels = encode_imdp_labels(imdp, dfa)
    with jax.default_device(args.rvi_device):
        lower, upper, policy = robust_scLTL_synthesis_imdp_jax(
            imdp, dfa, labels, float_dtype=args.floatprecision,
            epsilon=1e-6, max_iter=10000,
        )
    _, z0 = initial_product_state(s0, dfa, labels)
    return lower, upper, policy, dfa, z0, labels


def main():
    args = parse_arguments()
    args.coarse_factor = 1
    stamp = configure_run(args)

    formula = None
    # prepare_abstraction constructs the model internally; inspect the
    # factor-1 coarse data after inspecting the model in the build callback.
    def build_for_spec(args, model, partition, s_init):
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
    attach_region_data(imdp, model, partition)
    formula = getattr(model, 'specification', None)
    if formula is None:
        raise ValueError('The benchmark does not define a specification.')
    s0, exists = partition.x2state(model.x0)
    if not exists:
        raise ValueError('Initial continuous state is outside the partition.')
    reach_avoid = is_reach_avoid(formula)
    logger.info('=== IMDP %s synthesis (solver=%s) ===',
                'reach-avoid' if reach_avoid else 'scLTL', args.solver)
    t = time.time()

    if reach_avoid:
        V, policy = solve_imdp(args, imdp, s0)
        logger.info('Initial state s0=%d: value=%.6f', s0, V[s0])
        policy_inputs = extract_policy_inputs(
            args, partition, policy, imdp.nr_states
        )
        report(args, stamp, model, partition, V, policy, policy_inputs)
    else:
        V_lower, V_upper, policy, dfa, z0, labels = solve_imdp_scLTL(
            args, imdp, s0, formula
        )
        logger.info('Initial product state (%d,%d): lower=%.6f, upper=%.6f',
                    s0, z0, V_lower[s0, z0], V_upper[s0, z0])
        np.savez_compressed(
            args.output_dir / 'product_synthesis.npz',
            V_lower=V_lower, V_upper=V_upper, policy=policy,
            initial_state=s0, initial_dfa_state=z0,
            dfa_transitions=dfa.transitions,
            dfa_accepting=dfa.accepting_states,
            atomic_propositions=np.asarray(dfa.atomic_propositions),
        )
        policy_inputs = extract_policy_inputs(
            args, partition, policy, imdp.nr_states
        )
        report_scLTL(
            args=args, stamp=stamp, model=model, partition=partition,
            V_lower=V_lower, policy=policy, dfa=dfa,
            labels=labels, policy_inputs=policy_inputs,
        )

    logger.info('<<< IMDP synthesis completed (took %.3f sec.) >>>', time.time() - t)
    finish(args)


if __name__ == '__main__':
    main()
