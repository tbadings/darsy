from __future__ import annotations

import logging
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# PREPROCESSING
# ============================================================================


def preprocess_2imdp(imdp, states, max_actions, float_dtype):
    """Build solver arrays directly from the rectangular abstraction.

    Zero/zero entries are unused window slots. The absorbing successor is
    appended explicitly and has no coarse parent.
    """
    states = np.asarray(states, dtype=np.int32)
    parents = np.asarray(imdp.parents, dtype=np.int32)
    records = {}
    max_s = max_q = max_c = 0
    action_mask = np.zeros((len(states), max_actions), dtype=bool)
    for si, s in enumerate(states):
        actions = imdp.A_id.get(int(s), [])
        if len(actions) > max_actions:
            raise ValueError(f"State {s} has too many actions")
        action_mask[si, :len(actions)] = True
        for a in range(len(actions)):
            fine = np.asarray(imdp.P_full[int(s)][a])
            ids = np.asarray(imdp.S_id[int(s)][a], dtype=np.int32)
            valid = np.any(fine > 0, axis=1)
            ids, fine = ids[valid], fine[valid]
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate fine successors at ({s}, {a})")
            ids = np.append(ids, int(imdp.absorbing_state)).astype(np.int32)
            fine = np.vstack((fine, np.asarray(imdp.P_absorbing[int(s)][a])))
            coarse = np.asarray(imdp.P_coarse[int(s)][a])
            qids = np.asarray(imdp.Q_id[int(s)][a], dtype=np.int32)
            qvalid = np.any(coarse > 0, axis=1)
            qids, coarse = qids[qvalid], coarse[qvalid]
            if len(np.unique(qids)) != len(qids):
                raise ValueError(f"Duplicate coarse successors at ({s}, {a})")
            lookup = {int(q): j for j, q in enumerate(qids)}
            cluster_idx = np.array([lookup.get(int(parents[sp]), -1)
                                    if sp < len(parents) else -1 for sp in ids], dtype=np.int32)
            children = [np.flatnonzero(cluster_idx == j).astype(np.int32)
                        for j in range(len(qids))]
            records[si, a] = ids, fine, qids, coarse, cluster_idx, children
            max_s = max(max_s, len(ids))
            max_q = max(max_q, len(qids))
            max_c = max(max_c, *(map(len, children)), 0)

    shape_s = (len(states), max_actions, max_s)
    shape_q = (len(states), max_actions, max_q)
    shape_c = (len(states), max_actions, max_q, max_c)
    result = dict(
        states=states, action_mask=action_mask,
        successors=np.full(shape_s, -1, dtype=np.int32),
        successors_mask=np.zeros(shape_s, dtype=bool),
        prob_lb=np.zeros(shape_s, dtype=float_dtype),
        prob_ub=np.zeros(shape_s, dtype=float_dtype),
        coarse_constraint_idx=np.full(shape_s, -1, dtype=np.int32),
        coarse_states=np.full(shape_q, -1, dtype=np.int32),
        coarse_mask=np.zeros(shape_q, dtype=bool),
        prob_lb_coarse=np.zeros(shape_q, dtype=float_dtype),
        prob_ub_coarse=np.zeros(shape_q, dtype=float_dtype),
        children_indices=np.full(shape_c, -1, dtype=np.int32),
        children_mask=np.zeros(shape_c, dtype=bool),
        n_states=len(states), max_actions=max_actions,
        max_successors=max_s, max_coarse=max_q, max_children=max_c,
    )
    for (si, a), (ids, fine, qids, coarse, cluster_idx, children) in records.items():
        n, m = len(ids), len(qids)
        result['successors'][si, a, :n] = ids
        result['successors_mask'][si, a, :n] = True
        result['prob_lb'][si, a, :n] = fine[:, 0]
        result['prob_ub'][si, a, :n] = fine[:, 1]
        result['coarse_constraint_idx'][si, a, :n] = cluster_idx
        result['coarse_states'][si, a, :m] = qids
        result['coarse_mask'][si, a, :m] = True
        result['prob_lb_coarse'][si, a, :m] = coarse[:, 0]
        result['prob_ub_coarse'][si, a, :m] = coarse[:, 1]
        for q, child in enumerate(children):
            result['children_indices'][si, a, q, :len(child)] = child
            result['children_mask'][si, a, q, :len(child)] = True
    return result


# ============================================================================
# SORTING
# ============================================================================


def compute_sort_indices(
    V: np.ndarray,
    successors: np.ndarray,
    successors_mask: np.ndarray,
    children_indices: np.ndarray,
    children_mask: np.ndarray,
):
    """
    Compute successor and within-coarse-child orderings outside JAX.

    Lower-bound ordering:
        smallest value first.

    Upper-bound ordering:
        largest value first.

    All arrays are shaped as:

        successors       : (state, action, successor)
        children_indices : (state, action, coarse, child)
    """

    # ----------------------------------------------------------------------
    # Fine successor values.
    # ----------------------------------------------------------------------

    safe_successors = np.where(
        successors_mask,
        successors,
        0,
    )

    successor_values = V[safe_successors]

    # ----------------------------------------------------------------------
    # Global successor ordering.
    # ----------------------------------------------------------------------

    values_lower = np.where(
        successors_mask,
        successor_values,
        np.inf,
    )

    successor_order_lower = np.argsort(
        values_lower,
        axis=-1,
    ).astype(np.int32)

    values_upper = np.where(
        successors_mask,
        successor_values,
        -np.inf,
    )

    successor_order_upper = np.argsort(
        values_upper,
        axis=-1,
    )[:, :, ::-1].astype(np.int32)

    # ----------------------------------------------------------------------
    # Within-coarse child ordering.
    # ----------------------------------------------------------------------

    safe_children = np.where(
        children_mask,
        children_indices,
        0,
    )

    child_values = np.take_along_axis(
        successor_values[:, :, None, :],
        safe_children,
        axis=-1,
    )

    values_child_lower = np.where(
        children_mask,
        child_values,
        np.inf,
    )

    child_order_lower = np.argsort(
        values_child_lower,
        axis=-1,
    ).astype(np.int32)

    values_child_upper = np.where(
        children_mask,
        child_values,
        -np.inf,
    )

    child_order_upper = np.argsort(
        values_child_upper,
        axis=-1,
    )[:, :, :, ::-1].astype(np.int32)

    return (
        successor_order_lower,
        successor_order_upper,
        child_order_lower,
        child_order_upper,
    )


# ============================================================================
# ONE STATE-ACTION ROBUST EXPECTATION
# ============================================================================


def robust_expectation_2layer(
    successors,
    successors_mask,
    prob_lb,
    prob_ub,
    coarse_constraint_idx,
    coarse_mask,
    prob_lb_coarse,
    prob_ub_coarse,
    children_indices,
    children_mask,
    successor_order,
    child_order,
    V,
):
    """
    Solve the robust expectation for one padded state-action pair.

    The inner optimization is solved greedily in two stages:

    1. Start from fine lower bounds and satisfy each active coarse lower
       bound by assigning additional mass to children in value order.

    2. Allocate all remaining mass globally in value order, respecting both
       fine and coarse upper bounds.

    `successor_order` and `child_order` are computed outside the JAX mapped
    operations.
    """

    # ----------------------------------------------------------------------
    # Safe successor indexing.
    # ----------------------------------------------------------------------

    safe_successors = jnp.where(
        successors_mask,
        successors,
        0,
    )

    successor_values = V[safe_successors]

    # ----------------------------------------------------------------------
    # Start from individual lower bounds.
    # ----------------------------------------------------------------------

    gamma = jnp.where(
        successors_mask,
        prob_lb,
        0.0,
    )

    # ----------------------------------------------------------------------
    # Aggregate fine lower bounds into coarse constraints.
    #
    # Each successor has at most one active coarse constraint because the
    # coarse sets are disjoint and coarse_constraint_idx is a single index.
    # ----------------------------------------------------------------------

    safe_parent = jnp.where(
        coarse_constraint_idx >= 0,
        coarse_constraint_idx,
        0,
    )

    valid_parent = (
        successors_mask
        & (coarse_constraint_idx >= 0)
        & jnp.take(
            coarse_mask,
            safe_parent,
            axis=0,
        )
    )

    gamma_coarse = jnp.zeros_like(
        prob_lb_coarse,
    )

    gamma_coarse = gamma_coarse.at[
        safe_parent
    ].add(
        jnp.where(
            valid_parent,
            gamma,
            0.0,
        )
    )

    # ----------------------------------------------------------------------
    # Remaining total probability mass.
    # ----------------------------------------------------------------------

    remaining_mass = 1.0 - jnp.sum(gamma)

    # ----------------------------------------------------------------------
    # Reorder children using the precomputed ordering.
    # ----------------------------------------------------------------------

    safe_children = jnp.where(
        children_mask,
        children_indices,
        0,
    )

    sorted_children = jnp.take_along_axis(
        safe_children,
        child_order,
        axis=-1,
    )

    sorted_children_mask = jnp.take_along_axis(
        children_mask,
        child_order,
        axis=-1,
    )

    # ----------------------------------------------------------------------
    # PHASE 1: satisfy coarse lower bounds.
    # ----------------------------------------------------------------------

    def coarse_body(qpi, carry):
        gamma, gamma_coarse, remaining_mass = carry

        required_mass = jnp.where(
            coarse_mask[qpi],
            jnp.maximum(
                0.0,
                prob_lb_coarse[qpi]
                - gamma_coarse[qpi],
            ),
            0.0,
        )

        def child_body(k, inner_carry):
            (
                gamma,
                gamma_coarse,
                remaining_mass,
                required_mass,
            ) = inner_carry

            spi = sorted_children[qpi, k]

            valid_child = (
                sorted_children_mask[qpi, k]
                & (required_mass > 0.0)
                & (remaining_mass > 0.0)
            )

            fine_capacity = jnp.maximum(
                0.0,
                prob_ub[spi] - gamma[spi],
            )

            allocation = jnp.minimum(
                required_mass,
                jnp.minimum(
                    fine_capacity,
                    remaining_mass,
                ),
            )

            allocation = jnp.where(
                valid_child,
                allocation,
                0.0,
            )

            gamma = gamma.at[spi].add(
                allocation,
            )

            gamma_coarse = gamma_coarse.at[qpi].add(
                allocation,
            )

            required_mass -= allocation
            remaining_mass -= allocation

            return (
                gamma,
                gamma_coarse,
                remaining_mass,
                required_mass,
            )

        result = jax.lax.fori_loop(
            0,
            children_indices.shape[-1],
            child_body,
            (
                gamma,
                gamma_coarse,
                remaining_mass,
                required_mass,
            ),
        )

        return result[:3]

    gamma, gamma_coarse, remaining_mass = jax.lax.fori_loop(
        0,
        coarse_mask.shape[0],
        coarse_body,
        (
            gamma,
            gamma_coarse,
            remaining_mass,
        ),
    )

    # ----------------------------------------------------------------------
    # PHASE 2: distribute remaining mass globally in the supplied order.
    # ----------------------------------------------------------------------

    def successor_body(k, carry):
        gamma, gamma_coarse, remaining_mass = carry

        spi = successor_order[k]

        valid_successor = (
            successors_mask[spi]
            & (remaining_mass > 0.0)
        )

        qpi = coarse_constraint_idx[spi]
        safe_qpi = jnp.where(
            qpi >= 0,
            qpi,
            0,
        )

        valid_coarse_constraint = (
            (qpi >= 0)
            & jnp.take(
                coarse_mask,
                safe_qpi,
                axis=0,
            )
        )

        fine_capacity = jnp.maximum(
            0.0,
            prob_ub[spi] - gamma[spi],
        )

        coarse_capacity = jnp.where(
            valid_coarse_constraint,
            jnp.maximum(
                0.0,
                prob_ub_coarse[safe_qpi]
                - gamma_coarse[safe_qpi],
            ),
            jnp.inf,
        )

        allocation = jnp.minimum(
            remaining_mass,
            jnp.minimum(
                fine_capacity,
                coarse_capacity,
            ),
        )

        allocation = jnp.where(
            valid_successor,
            allocation,
            0.0,
        )

        gamma = gamma.at[spi].add(
            allocation,
        )

        gamma_coarse = gamma_coarse.at[safe_qpi].add(
            jnp.where(
                valid_coarse_constraint,
                allocation,
                0.0,
            ),
        )

        remaining_mass -= allocation

        return gamma, gamma_coarse, remaining_mass

    gamma, gamma_coarse, remaining_mass = jax.lax.fori_loop(
        0,
        successors.shape[0],
        successor_body,
        (
            gamma,
            gamma_coarse,
            remaining_mass,
        ),
    )

    return jnp.clip(
        jnp.sum(gamma * successor_values),
        0.0,
        1.0,
    )


# ============================================================================
# VECTORIZED KERNELS
# ============================================================================


robust_expectation_2layer_jit = jax.jit(
    robust_expectation_2layer,
)


vmap_2layer_over_actions = jax.jit(
    jax.vmap(
        robust_expectation_2layer_jit,
        in_axes=(
            0,  # successors
            0,  # successors_mask
            0,  # prob_lb
            0,  # prob_ub
            0,  # coarse_constraint_idx
            0,  # coarse_mask
            0,  # prob_lb_coarse
            0,  # prob_ub_coarse
            0,  # children_indices
            0,  # children_mask
            0,  # successor_order
            0,  # child_order
            None,  # V
        ),
        out_axes=0,
    )
)


vmap_2layer_over_states = jax.jit(
    jax.vmap(
        vmap_2layer_over_actions,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
        ),
        out_axes=0,
    )
)


# ============================================================================
# LOWER VALUE ITERATION
# ============================================================================


def lower_value_iteration_2layer(
    data_prep: Dict[str, Any],
    V_lower: np.ndarray,
    sigma: np.ndarray,
    epsilon: float,
    max_iter: int,
    float_dtype: Any,
):
    """Robust lower-bound value iteration."""

    states_np = data_prep["states"]
    action_mask_np = data_prep["action_mask"]

    # Convert static transition arrays to JAX exactly once.
    successors = jnp.asarray(data_prep["successors"])
    successors_mask = jnp.asarray(data_prep["successors_mask"])
    prob_lb = jnp.asarray(data_prep["prob_lb"])
    prob_ub = jnp.asarray(data_prep["prob_ub"])
    coarse_constraint_idx = jnp.asarray(
        data_prep["coarse_constraint_idx"]
    )
    coarse_mask = jnp.asarray(data_prep["coarse_mask"])
    prob_lb_coarse = jnp.asarray(
        data_prep["prob_lb_coarse"]
    )
    prob_ub_coarse = jnp.asarray(
        data_prep["prob_ub_coarse"]
    )
    children_indices = jnp.asarray(
        data_prep["children_indices"]
    )
    children_mask = jnp.asarray(
        data_prep["children_mask"]
    )
    action_mask = jnp.asarray(action_mask_np)
    states = jnp.asarray(states_np)

    V_lower = jnp.asarray(V_lower, dtype=float_dtype)
    sigma = jnp.asarray(sigma, dtype=jnp.int32)

    for iteration in range(max_iter):

        V_old = V_lower

        # --------------------------------------------------------------
        # Sort outside the vmap/JIT computation, as in the original
        # RVI_JAX implementation.
        # --------------------------------------------------------------

        (
            successor_order_lower,
            _,
            child_order_lower,
            _,
        ) = compute_sort_indices(
            np.asarray(jax.device_get(V_lower)),
            data_prep["successors"],
            data_prep["successors_mask"],
            data_prep["children_indices"],
            data_prep["children_mask"],
        )

        # --------------------------------------------------------------
        # Robust Q-values for every state/action.
        # --------------------------------------------------------------

        Q_lower = vmap_2layer_over_states(
            successors,
            successors_mask,
            prob_lb,
            prob_ub,
            coarse_constraint_idx,
            coarse_mask,
            prob_lb_coarse,
            prob_ub_coarse,
            children_indices,
            children_mask,
            jnp.asarray(successor_order_lower),
            jnp.asarray(child_order_lower),
            V_lower,
        )

        # Disable padded actions.
        Q_lower = jnp.where(
            action_mask,
            Q_lower,
            -jnp.inf,
        )

        best_values = jnp.max(
            Q_lower,
            axis=1,
        )
        best_actions = jnp.argmax(
            Q_lower,
            axis=1,
        )

        old_values = V_lower[states]

        improve = (
            best_values > old_values
        )

        new_values = jnp.where(
            improve,
            jnp.minimum(best_values, 1.0),
            old_values,
        )

        new_policy = jnp.where(
            improve,
            best_actions,
            sigma[states],
        )

        V_lower = V_lower.at[states].set(
            new_values
        )

        sigma = sigma.at[states].set(
            new_policy
        )

        err = jnp.max(
            jnp.abs(V_lower - V_old)
        )

        logger.info(
            "Lower VI iteration %d: error=%.6e",
            iteration + 1,
            float(err),
        )

        if float(err) < epsilon:
            break

    return V_lower, sigma


# ============================================================================
# UPPER VALUE ITERATION
# ============================================================================


def upper_value_iteration_2layer(
    data_prep: Dict[str, Any],
    V_upper: np.ndarray,
    sigma: np.ndarray,
    epsilon: float,
    max_iter: int,
    float_dtype: Any,
):
    """Upper-bound value iteration for the fixed lower-bound policy."""

    states_np = data_prep["states"]

    successors = jnp.asarray(data_prep["successors"])
    successors_mask = jnp.asarray(data_prep["successors_mask"])
    prob_lb = jnp.asarray(data_prep["prob_lb"])
    prob_ub = jnp.asarray(data_prep["prob_ub"])
    coarse_constraint_idx = jnp.asarray(
        data_prep["coarse_constraint_idx"]
    )
    coarse_mask = jnp.asarray(data_prep["coarse_mask"])
    prob_lb_coarse = jnp.asarray(
        data_prep["prob_lb_coarse"]
    )
    prob_ub_coarse = jnp.asarray(
        data_prep["prob_ub_coarse"]
    )
    children_indices = jnp.asarray(
        data_prep["children_indices"]
    )
    children_mask = jnp.asarray(
        data_prep["children_mask"]
    )

    states = jnp.asarray(states_np)
    state_indices = jnp.arange(
        states.shape[0],
        dtype=jnp.int32,
    )

    V_upper = jnp.asarray(V_upper, dtype=float_dtype)
    sigma = jnp.asarray(sigma, dtype=jnp.int32)

    for iteration in range(max_iter):

        V_old = V_upper

        # --------------------------------------------------------------
        # Upper ordering, computed outside vmap/JIT.
        # --------------------------------------------------------------

        (
            _,
            successor_order_upper,
            _,
            child_order_upper,
        ) = compute_sort_indices(
            np.asarray(jax.device_get(V_upper)),
            data_prep["successors"],
            data_prep["successors_mask"],
            data_prep["children_indices"],
            data_prep["children_mask"],
        )

        # --------------------------------------------------------------
        # Gather the action selected by sigma at every state.
        # --------------------------------------------------------------

        selected_actions = sigma[states]

        successors_selected = successors[
            state_indices,
            selected_actions,
        ]
        successors_mask_selected = successors_mask[
            state_indices,
            selected_actions,
        ]
        prob_lb_selected = prob_lb[
            state_indices,
            selected_actions,
        ]
        prob_ub_selected = prob_ub[
            state_indices,
            selected_actions,
        ]
        coarse_constraint_idx_selected = coarse_constraint_idx[
            state_indices,
            selected_actions,
        ]
        coarse_mask_selected = coarse_mask[
            state_indices,
            selected_actions,
        ]
        prob_lb_coarse_selected = prob_lb_coarse[
            state_indices,
            selected_actions,
        ]
        prob_ub_coarse_selected = prob_ub_coarse[
            state_indices,
            selected_actions,
        ]
        children_indices_selected = children_indices[
            state_indices,
            selected_actions,
        ]
        children_mask_selected = children_mask[
            state_indices,
            selected_actions,
        ]
        successor_order_selected = jnp.asarray(
            successor_order_upper
        )[
            state_indices,
            selected_actions,
        ]
        child_order_selected = jnp.asarray(
            child_order_upper
        )[
            state_indices,
            selected_actions,
        ]

        # --------------------------------------------------------------
        # Vectorize the fixed-policy evaluation over states.
        # --------------------------------------------------------------

        upper_values = jax.jit(
            jax.vmap(
                robust_expectation_2layer_jit,
                in_axes=(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                ),
                out_axes=0,
            )
        )(
            successors_selected,
            successors_mask_selected,
            prob_lb_selected,
            prob_ub_selected,
            coarse_constraint_idx_selected,
            coarse_mask_selected,
            prob_lb_coarse_selected,
            prob_ub_coarse_selected,
            children_indices_selected,
            children_mask_selected,
            successor_order_selected,
            child_order_selected,
            V_upper,
        )

        V_upper = V_upper.at[states].set(
            upper_values
        )

        err = jnp.max(
            jnp.abs(V_upper - V_old)
        )

        logger.info(
            "Upper VI iteration %d: error=%.6e",
            iteration + 1,
            float(err),
        )

        if float(err) < epsilon:
            break

    return V_upper


# ============================================================================
# PUBLIC DRIVER
# ============================================================================


def robust_max_reach_probability_2layer_jax(
    data_prep, target_states, n_states, *, float_dtype,
    epsilon=1e-6, max_iter=10000,
):
    """Solve reachability directly from native 2IMDP padded arrays.

    Returns lower and fixed-policy upper values, and local action indices.
    Target states are fixed at value one; other non-updated states at zero.
    """
    states = np.asarray(data_prep['states'], dtype=np.int32)
    targets = np.asarray(target_states, dtype=np.int32)
    if n_states <= 0 or np.any(targets < 0) or np.any(targets >= n_states):
        raise ValueError('Invalid number of states or target state IDs')
    if np.intersect1d(states, targets).size:
        raise ValueError('Target states must not be updated by value iteration')
    if len(states) and not np.all(np.asarray(data_prep['action_mask']).any(axis=1)):
        raise ValueError('Each updated state must have at least one action')
    initial = np.zeros(n_states, dtype=float_dtype)
    initial[targets] = 1.0
    policy = np.full(n_states, -1, dtype=np.int32)
    policy[states] = 0
    if not len(states):
        return initial, initial.copy(), policy
    lower, policy = lower_value_iteration_2layer(
        data_prep, initial, policy, float(epsilon), int(max_iter), float_dtype
    )
    upper = upper_value_iteration_2layer(
        data_prep, initial, policy, float(epsilon), int(max_iter), float_dtype
    )
    return np.asarray(lower), np.asarray(upper), np.asarray(policy)
