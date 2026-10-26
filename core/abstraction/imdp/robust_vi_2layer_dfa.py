from __future__ import annotations

import logging

from typing import Any, Dict

import jax

import jax.numpy as jnp

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================*
# PREPROCESSING*
# ============================================================================*

def encode_imdp_labels(imdp, dfa):
    """
    Encode IMDP state labels using the DFA's atomic proposition ordering.

    Every abstract state whose cell intersects a region of interest r
    receives all propositions in L(r).

    If a state intersects multiple regions, its label is the union
    of the propositions associated with those regions.

    The absorbing state receives an empty label.
    """

    n = int(imdp.absorbing_state)

    aps = list(dfa.atomic_propositions)

    if len(set(aps)) != len(aps) or len(aps) > 62:
        raise ValueError("Invalid DFA atomic proposition list")

    known = set(imdp.atomic_propositions)

    if not set(aps) <= known:
        raise ValueError(
            f"DFA references unknown propositions: {set(aps) - known}"
        )

    labels = np.zeros(n + 1, dtype=np.int64)

    ap_idx = {
        ap: i for i, ap in enumerate(aps)
    }

    for region, mask in imdp.states_of_interest.items():

        mask = np.asarray(mask, dtype=bool)

        if mask.shape != (n,):
            raise ValueError(
                f"Invalid state mask for region {region}"
            )

        propositions = set(imdp.region_labels[region])

        unknown = propositions - known

        if unknown:
            raise ValueError(
                f"Unknown propositions in {region}: {unknown}"
            )

        bits = sum(
            1 << ap_idx[ap]
            for ap in propositions
            if ap in ap_idx
        )

        labels[np.flatnonzero(mask)] |= bits

    return labels

def preprocess_2layer_solution(

    solution: Any,

    states: np.ndarray,

    max_actions: int,
    float_dtype: Any,

) -> Dict[str, Any]:

    """

    Convert the variable-size 2I-MDP structure into padded dense arrays.

    Important structural assumptions:

    * Sp[s][a] contains the successor states of (s,a)

    * Qp[s][a] contains the clusters whose constraints we consider for (s,a)

    * Coarse sets are disjoint.

    * Given (s,a) and sp in Sp[s][a], spi denotes the local index of sp in Sp[s][a]

    * Given (s,a) and qp in Qp[s][a], qpi denotes the local index of qp in Qp[s][a]

    * Every fine state has at most one global parent coarse state.

    * Qp[s][a] contains the coarse states for which a coarse probability

      constraint exists for this particular (s, a).

    * Therefore, a fine successor can have a global parent which is NOT in

      Qp[s][a]. In that case it participates only in its fine-state bounds,

      not in a coarse constraint for this (s, a).

    The array `coarse_constraint_idx` captures this distinction:

        coarse_constraint_idx[spi] = qpi

    when the global parent of successor `spi` is one of Qp[s][a], and

    -1 otherwise.

    """

    states = np.asarray(states, dtype=np.int32)

    n_states = len(states)
# ----------------------------------------------------------------------*
# Determine padding dimensions.*
# ----------------------------------------------------------------------*

    max_successors = 0

    max_clusters = 0

    max_children = 0

    action_mask = np.zeros(

        (n_states, max_actions),

        dtype=np.bool_,

    )
# Obtain *

    for si, s in enumerate(states):

        s = int(s)
# Check that there are at most max_actions enabled at s*

        n_actions = len(solution.A_id[s])

        if n_actions > max_actions:

            raise ValueError(

                f"State {s} has {n_actions} actions, but max_actions="

                f"{max_actions}."

            )

        action_mask[si, :n_actions] = True

        for a in range(n_actions):

            successors_sa = np.asarray(

                solution.Sp[s][a],

                dtype=np.int32,

            )

            clusters_sa = np.asarray(

                solution.Qp[s][a],

                dtype=np.int32,

            )

            max_successors = max(

                max_successors,

                len(successors_sa),

            )

            max_clusters = max(

                max_clusters,

                len(clusters_sa),

            )
# For the current (s,a), make a dictionary that maps the*
# cluster qp to the index qpi in clusters*
# Count fine successors belonging to each coarse*
# state. Because parents are unique, this is a simple grouping.*

            if len(clusters_sa) > 0:

                cluster_lookup_sa = {

                    int(qp): qpi

                    for qpi, qp in enumerate(clusters_sa)

                }

                child_counts_sa = np.zeros(

                    len(clusters_sa),

                    dtype=np.int32,

                )
# For the current (s,a), make a dictionary that maps the*
# cluster qp to the index qpi in clusters*
# Count fine successors belonging to each coarse*
# state. Because parents are unique, this is a simple grouping.   *

                for sp in successors_sa:

                    qp = solution.parents[int(sp)]
# Check that qp actually exists in clusters_sa*

                    if qp is not None and int(qp) in clusters_sa:

                        child_counts_sa[cluster_lookup_sa[int(qp)]] += 1

                if len(child_counts_sa) > 0:

                    max_children = max(

                        max_children,

                        int(np.max(child_counts_sa)),

                    )
# ----------------------------------------------------------------------*
# Allocate padded arrays.*
# ----------------------------------------------------------------------*

    full_successors = np.full(

        (n_states, max_actions, max_successors),

        -1,

        dtype=np.int32,

    )

    full_successors_mask = np.zeros(

        (n_states, max_actions, max_successors),

        dtype=np.bool_,

    )

    full_prob_lb = np.zeros(

        (n_states, max_actions, max_successors),

        dtype=float_dtype,

    )

    full_prob_ub = np.zeros(

        (n_states, max_actions, max_successors),

        dtype=float_dtype,

    )
# Local cluster index qpi for each (si,a,spi).*

    full_cluster_idx = np.full(

        (n_states, max_actions, max_successors),

        -1,

        dtype=np.int32,

    )

    full_clusters = np.full(

        (n_states, max_actions, max_clusters),

        -1,

        dtype=np.int32,

    )

    full_cluster_mask = np.zeros(

        (n_states, max_actions, max_clusters),

        dtype=np.bool_,

    )

    full_prob_lb_coarse = np.zeros(

        (n_states, max_actions, max_clusters),

        dtype=float_dtype,

    )

    full_prob_ub_coarse = np.zeros(

        (n_states, max_actions, max_clusters),

        dtype=float_dtype,

    )
# Fine successor indices spi belonging to each (si,a,qpi).*

    full_children_indices = np.full(

        (n_states, max_actions, max_clusters, max_children),

        -1,

        dtype=np.int32,

    )

    full_children_mask = np.zeros(

        (n_states, max_actions, max_clusters, max_children),

        dtype=np.bool_,

    )
# ----------------------------------------------------------------------*
# Fill arrays.*
# ----------------------------------------------------------------------*

    for si, s in enumerate(states):

        s = int(s)

        n_actions = len(solution.A_id[s])

        for a in range(n_actions):

            successors_sa = np.asarray(

                solution.Sp[s][a],

                dtype=np.int32,

            )

            p_lower_sa = np.asarray(

                solution.P_lower[s][a],

                dtype=float_dtype,

            )

            p_upper_sa = np.asarray(

                solution.P_upper[s][a],

                dtype=float_dtype,

            )

            n_successors_sa = len(successors_sa)

            full_successors[

                si, a, :n_successors_sa

            ] = successors_sa

            full_successors_mask[

                si, a, :n_successors_sa

            ] = True

            full_prob_lb[

                si, a, :n_successors_sa

            ] = p_lower_sa

            full_prob_ub[

                si, a, :n_successors_sa

            ] = p_upper_sa

            clusters_sa = np.asarray(

                solution.Qp[s][a],

                dtype=np.int32,

            )

            p_lower_coarse_sa = np.asarray(

                solution.P_lower_coarse[s][a],

                dtype=float_dtype,

            )

            p_upper_coarse_sa = np.asarray(

                solution.P_upper_coarse[s][a],

                dtype=float_dtype,

            )

            n_clusters_sa = len(clusters_sa)

            full_clusters[

                si, a, :n_clusters_sa

            ] = clusters_sa

            full_cluster_mask[

                si, a, :n_clusters_sa

            ] = True

            full_prob_lb_coarse[

                si, a, :n_clusters_sa

            ] = p_lower_coarse_sa

            full_prob_ub_coarse[

                si, a, :n_clusters_sa

            ] = p_upper_coarse_sa

            cluster_lookup_sa = {

                int(qp): qpi

                for qpi, qp in enumerate(clusters_sa)

            }
# This variable is gonna map the state spi to its parent (the local qpi)*

            cluster_idx_sa = np.full(

                n_successors_sa,

                -1,

                dtype=np.int32,

            )

            for spi, sp in enumerate(successors_sa):

                qp = solution.parents[int(sp)]

                if qp is not None:

                    qp = int(qp)

                    if qp in cluster_lookup_sa:

                        cluster_idx_sa[spi] = cluster_lookup_sa[qp]

            full_cluster_idx[

                si, a, :n_successors_sa

            ] = cluster_idx_sa
# Children are derived from the local cluster index qpi.*

            for qpi in range(n_clusters_sa):
# child_indices_sa is a vector with all the spi that have a parent*
# in clusters_sa*

                child_indices_sa = np.flatnonzero(

                    cluster_idx_sa == qpi

                ).astype(np.int32)

                n_children_sa_qpi = len(child_indices_sa)

                if n_children_sa_qpi == 0:

                    continue

                full_children_indices[

                    si, a, qpi, :n_children_sa_qpi

                ] = child_indices_sa

                full_children_mask[

                    si, a, qpi, :n_children_sa_qpi

                ] = True

    return {

        "states": states,

        "action_mask": action_mask,

        "successors": full_successors,

        "successors_mask": full_successors_mask,

        "prob_lb": full_prob_lb,

        "prob_ub": full_prob_ub,

        "coarse_constraint_idx": full_cluster_idx,

        "coarse_states": full_clusters,

        "coarse_mask": full_cluster_mask,

        "prob_lb_coarse": full_prob_lb_coarse,

        "prob_ub_coarse": full_prob_ub_coarse,

        "children_indices": full_children_indices,

        "children_mask": full_children_mask,

        "n_states": n_states,

        "max_actions": max_actions,

        "max_successors": max_successors,

        "max_coarse": max_clusters,

        "max_children": max_children,

    }

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
# ============================================================================*
# SORTING*
# ============================================================================*

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
# ----------------------------------------------------------------------*
# Fine successor values.*
# ----------------------------------------------------------------------*

    safe_successors = np.where(

        successors_mask,

        successors,

        0,

    )

    successor_values = V[safe_successors]
# ----------------------------------------------------------------------*
# Global successor ordering.*
# ----------------------------------------------------------------------*

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
# ----------------------------------------------------------------------*
# Within-coarse child ordering.*
# ----------------------------------------------------------------------*

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
# ============================================================================*
# ONE STATE-ACTION ROBUST EXPECTATION*
# ============================================================================*

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
# ----------------------------------------------------------------------*
# Safe successor indexing.*
# ----------------------------------------------------------------------*

    safe_successors = jnp.where(

        successors_mask,

        successors,

        0,

    )

    successor_values = V[safe_successors]
# ----------------------------------------------------------------------*
# Start from individual lower bounds.*
# ----------------------------------------------------------------------*

    gamma = jnp.where(

        successors_mask,

        prob_lb,

        0.0,

    )
# ----------------------------------------------------------------------*
# Aggregate fine lower bounds into coarse constraints.*
#*
# Each successor has at most one active coarse constraint because the*
# coarse sets are disjoint and coarse_constraint_idx is a single index.*
# ----------------------------------------------------------------------*

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
# ----------------------------------------------------------------------*
# Remaining total probability mass.*
# ----------------------------------------------------------------------*

    remaining_mass = 1.0 - jnp.sum(gamma)
# ----------------------------------------------------------------------*
# Reorder children using the precomputed ordering.*
# ----------------------------------------------------------------------*

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
# ----------------------------------------------------------------------*
# PHASE 1: satisfy coarse lower bounds.*
# ----------------------------------------------------------------------*

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
# ----------------------------------------------------------------------*
# PHASE 2: distribute remaining mass globally in the supplied order.*
# ----------------------------------------------------------------------*

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
# ============================================================================*
# VECTORIZED KERNELS*
# ============================================================================*

robust_expectation_2layer_jit = jax.jit(

    robust_expectation_2layer,

)

vmap_2layer_over_actions = jax.jit(

    jax.vmap(

        robust_expectation_2layer_jit,

        in_axes=(

            0,  # successors*

            0,  # successors_mask*

            0,  # prob_lb*

            0,  # prob_ub*

            0,  # coarse_constraint_idx*

            0,  # coarse_mask*

            0,  # prob_lb_coarse*

            0,  # prob_ub_coarse*

            0,  # children_indices*

            0,  # children_mask*

            0,  # successor_order*

            0,  # child_order*

            None,  # V*

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
# ============================================================================*
# LOWER VALUE ITERATION*
# ============================================================================*

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
# Convert static transition arrays to JAX exactly once.*

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
# --------------------------------------------------------------*
# Sort outside the vmap/JIT computation, as in the original*
# RVI_JAX implementation.*
# --------------------------------------------------------------*

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
# --------------------------------------------------------------*
# Robust Q-values for every state/action.*
# --------------------------------------------------------------*

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
# Disable padded actions.*

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
# ============================================================================*
# UPPER VALUE ITERATION*
# ============================================================================*

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
# --------------------------------------------------------------*
# Upper ordering, computed outside vmap/JIT.*
# --------------------------------------------------------------*

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
# --------------------------------------------------------------*
# Gather the action selected by sigma at every state.*
# --------------------------------------------------------------*

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
# --------------------------------------------------------------*
# Vectorize the fixed-policy evaluation over states.*
# --------------------------------------------------------------*

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
# ============================================================================*
# PUBLIC DRIVER*
# ============================================================================*

def robust_max_reach_probability_2layer_jax(

    user_param: Any,

    initial_condition: Any,

    solution: Any,

    solver_opts: Any,

    *,

    data_prep: Dict[str, Any] | None = None,
    float_dtype: Any,

):

    """

    Run robust lower and upper value iteration for the 2I-MDP.

    The lower iteration selects sigma by maximizing the worst-case

    transition expectation. The upper iteration evaluates that fixed sigma

    using the best-case transition expectation.

    """

    max_iter = int(user_param.max_iter)

    epsilon = float(solver_opts.tol_VI)

    states = np.asarray(

        user_param.S_not_terminal,

        dtype=np.int32,

    )

    if data_prep is None:

        if solution is None:

            raise ValueError("Provide either solution or preprocessed data_prep")

        max_actions = max((len(solution.A_id[int(s)]) for s in states), default=0)

        data_prep = preprocess_2layer_solution(solution, states, max_actions, float_dtype)

    elif not np.array_equal(data_prep['states'], states):

        raise ValueError("data_prep states must match S_not_terminal")

    V_lower = np.asarray(

        initial_condition.V_lower,

        dtype=float_dtype,

    ).copy()

    V_upper = np.asarray(

        initial_condition.V_upper,

        dtype=float_dtype,

    ).copy()

    sigma = np.asarray(

        initial_condition.sigma,

        dtype=np.int32,

    ).copy()
# Target states are absorbing with value 1 for reachability.*

    for s in np.asarray(

        user_param.S_target,

        dtype=np.int32,

    ):

        V_lower[int(s)] = 1.0

        V_upper[int(s)] = 1.0
# Lower bound / policy synthesis.*

    V_lower, sigma = lower_value_iteration_2layer(

        data_prep=data_prep,

        V_lower=V_lower,

        sigma=sigma,

        epsilon=epsilon,

        max_iter=max_iter,
        float_dtype=float_dtype,

    )
# Upper bound / fixed-policy evaluation.*

    V_upper = upper_value_iteration_2layer(

        data_prep=data_prep,

        V_upper=V_upper,

        sigma=sigma,

        epsilon=epsilon,

        max_iter=max_iter,
        float_dtype=float_dtype,

    )

    return (

        np.asarray(V_lower),

        np.asarray(V_upper),

        np.asarray(sigma),

    )

def save_rvi_results_matlab(

    filename,

    P_reach_lower,

    P_reach_upper,

    sigma,

):

    """

    Save 2I-MDP robust value iteration results in a text format

    readable from MATLAB.

    Python action indices are converted from 0-based to MATLAB's

    1-based indexing.

    """

    P_reach_lower = np.asarray(P_reach_lower).reshape(-1)

    P_reach_upper = np.asarray(P_reach_upper).reshape(-1)

    sigma = np.asarray(sigma).reshape(-1)
# sigma may contain an extra absorbing-state entry.*
# The MATLAB policy has one entry for each of the Ns states.*

    if len(sigma) > len(P_reach_lower) - 1:

        sigma = sigma[:len(P_reach_lower) - 1]
# Convert Python action indices to MATLAB action indices.*
#*
# Preserve -1 if it is used to denote "no action".*

    sigma_matlab = np.where(

        sigma >= 0,

        sigma + 1,

        -1,

    )

    with open(filename, "w") as f:

        f.write("P_reach_lower\n")

        for value in P_reach_lower:

            f.write(f"{float(value):.17g}\n")

        f.write("P_reach_upper\n")

        for value in P_reach_upper:

            f.write(f"{float(value):.17g}\n")

        f.write("sigma\n")

        for value in sigma_matlab:

            f.write(f"{int(value)}\n")
# ============================================================================
# IMPLICIT 2IMDP x DFA PRODUCT
# ============================================================================

def preprocess_2layer_dfa_product(data_prep, dfa, labels, n_base_states=None):
    """
    Lift padded 2IMDP constraints to nonterminal product states.

    labels[s] is a bitmask of propositions true at fine state s.

    Product index:
        (s, z) -> s * n_dfa_states + z

    The same 2IMDP ambiguity set is retained for each DFA state z.

    Terminal DFA states are excluded from Bellman updates.
    """

    transitions = np.asarray(dfa.transitions, dtype=np.int32)
    accepting = np.asarray(dfa.accepting_states, dtype=bool)
    # All accepting states are terminal; a rejecting self-loop is terminal too.
    sinks = np.all(transitions == np.arange(transitions.shape[0])[:, None], axis=1)
    terminal = accepting | sinks

    labels = np.asarray(labels, dtype=np.int64)

    n_z = transitions.shape[0]

    # --------------------------------------------------
    # Validate DFA
    # --------------------------------------------------

    if (
        transitions.ndim != 2
        or transitions.shape[1] != 2 ** len(dfa.atomic_propositions)
    ):
        raise ValueError("Invalid DFA transition table shape")

    if accepting.shape != (n_z,):
        raise ValueError("Invalid accepting-state mask")

    if terminal.shape != (n_z,):
        raise ValueError("Invalid terminal-state mask")

    if np.any(transitions < 0) or np.any(transitions >= n_z):
        raise ValueError("DFA must be complete and deterministic")

    if np.any(
        transitions[accepting]
        != np.flatnonzero(accepting)[:, None]
    ):
        raise ValueError("Accepting DFA states must be absorbing")

    # --------------------------------------------------
    # Validate base states and labels
    # --------------------------------------------------

    original_states = np.asarray(
        data_prep["states"],
        dtype=np.int32
    )

    original_successors = np.asarray(
        data_prep["successors"],
        dtype=np.int32
    )

    if n_base_states is None:

        valid = np.asarray(
            data_prep["successors_mask"],
            dtype=bool
        )

        max_successor = (
            int(original_successors[valid].max())
            if valid.any()
            else -1
        )

        n_base_states = max(
            len(labels),
            int(original_states.max()) + 1,
            max_successor + 1
        )

    if len(labels) != n_base_states:
        raise ValueError(
            "labels must include all base states, "
            "including absorbing states"
        )

    if np.any(labels < 0) or np.any(
        labels >= transitions.shape[1]
    ):
        raise ValueError(
            "A state label is not a valid DFA alphabet bitmask"
        )

    if np.any(original_states < 0) or np.any(
        original_states >= n_base_states
    ):
        raise ValueError("Invalid base state index")

    # --------------------------------------------------
    # Identify DFA states requiring Bellman updates
    # --------------------------------------------------

    nonterminal = np.flatnonzero(~terminal)

    # Repeat each base state for every nonterminal DFA state.
    base_rows = np.repeat(
        np.arange(len(original_states)),
        len(nonterminal)
    )

    automaton_rows = np.tile(
        nonterminal,
        len(original_states)
    )

    # Product index: (s,z) -> s*n_z + z
    product_states = (
        original_states[base_rows] * n_z
        + automaton_rows
    )

    # --------------------------------------------------
    # Lift original 2IMDP data
    # --------------------------------------------------

    product = {}

    for key, value in data_prep.items():

        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == len(original_states)
            and key != "states"
        ):
            product[key] = value[base_rows].copy()

        else:
            product[key] = value

    product["states"] = product_states.astype(np.int32)
    product["n_states"] = len(product_states)

    # --------------------------------------------------
    # Construct product successor indices
    # --------------------------------------------------

    # Probability bounds, coarse constraints, cluster
    # membership, and child indices remain unchanged.

    mask = np.asarray(
        product["successors_mask"],
        dtype=bool
    )

    base_successors = original_successors[base_rows]

    safe = np.where(
        mask,
        base_successors,
        0
    )

    if (
        np.any(base_successors[mask] >= n_base_states)
        or np.any(base_successors[mask] < 0)
    ):
        raise ValueError("Invalid fine successor index")

    # DFA state for each product state, broadcast across actions and successors
    current_z = automaton_rows[:, None, None]

    # Encoded labels of successor IMDP states
    successor_labels = labels[safe]

    # Compute successor DFA states
    next_z = transitions[current_z, successor_labels]

    sink_state = n_base_states - 1
    next_z = np.where(safe == sink_state, current_z, next_z)

    # Construct product successor indices
    product["successors"] = np.where(
        mask,
        safe * n_z + next_z,
        -1
    ).astype(np.int32)

    # --------------------------------------------------
    # Store product information
    # --------------------------------------------------

    product["n_product_states"] = n_base_states * n_z
    product["n_dfa_states"] = n_z
    product["n_base_states"] = n_base_states

    product["accepting_dfa_states"] = accepting
    product["terminal_dfa_states"] = terminal

    return product

def robust_scLTL_synthesis_2layer_jax(
    data_prep, dfa, labels, *, float_dtype, epsilon=1e-6, max_iter=1000,
):
    """Maximize worst-case sc-LTL satisfaction; evaluate fixed policy best-case.

    Parameters
    ----------
    data_prep : dict
        Output of preprocess_2imdp or preprocess_2layer_solution.
        Its 'states' should list precisely the nonterminal base states to update.
    dfa : DFA
        Complete good-prefix DFA with absorbing accepting states.
    labels : array-like of int
        One proposition bitmask per base state, including the absorbing state.
        Labels must use dfa.atomic_propositions bit order.

    Returns
    -------
    V_lower, V_upper, sigma : arrays of shape (n_base_states, n_dfa_states)
        sigma contains local action indices and -1 for non-updated states.
    """
    prep = preprocess_2layer_dfa_product(data_prep, dfa, labels)
    n_s, n_z = prep['n_base_states'], prep['n_dfa_states']
    accepting = np.asarray(dfa.accepting_states, dtype=bool)
    initial_values = np.broadcast_to(accepting[None, :], (n_s, n_z)).astype(float_dtype).copy()
    # Out-of-partition transitions count as failure unless already accepted.
    # These rows are not updated: only physical base states are preprocessed.
    absorbing_state = n_s - 1
    initial_values[absorbing_state, ~accepting] = 0.0
    initial_policy = np.full((n_s, n_z), -1, dtype=np.int32)
    active = prep['states']
    if active.size == 0:
        return initial_values, initial_values.copy(), initial_policy
    if not np.all(np.asarray(prep['action_mask']).any(axis=1)):
        raise ValueError('Every nonterminal product state must have an enabled action')
    initial_policy.reshape(-1)[active] = 0

    lower, policy = lower_value_iteration_2layer(
        prep, initial_values.reshape(-1), initial_policy.reshape(-1),
        float(epsilon), int(max_iter), float_dtype
    )
    upper = upper_value_iteration_2layer(
        prep, jnp.asarray(initial_values.reshape(-1), dtype=float_dtype), policy,
        float(epsilon), int(max_iter), float_dtype
    )
    return (
        np.asarray(lower).reshape(n_s, n_z),
        np.asarray(upper).reshape(n_s, n_z),
        np.asarray(policy).reshape(n_s, n_z),
    )


def initial_product_state(s0, dfa, labels):
    """Initial product state under the successor-label convention."""
    s0 = int(s0)
    z0 = int(dfa.initial_state)
    return s0, int(dfa.transitions[z0, int(labels[s0])])
    """Initial product state under the successor-label convention."""
    s0 = int(s0)
    q0 = int(dfa.initial_state)
    return s0, int(dfa.transitions[q0, int(labels[s0])])