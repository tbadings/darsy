"""Finite-horizon safety for IMDPs and 2IMDPs (remaining-horizon indexing)."""
import numpy as np
import jax
import jax.numpy as jnp
from core.abstraction.imdp.rvi_jax_dfa import _robust_expectation
from core.abstraction.imdp.robust_vi_2layer import preprocess_2imdp
from core.abstraction.imdp.robust_vi_2layer_dfa import (
    compute_sort_indices, vmap_2layer_over_states,
)


def _fine_data(imdp, dtype):
    states = np.array([int(s) for s in imdp.states if len(imdp.A_id.get(int(s), []))], dtype=np.int32)
    na = max((len(imdp.A_id[int(s)]) for s in states), default=0)
    nk = max((imdp.S_id[int(s)].shape[1] + 1 for s in states), default=1)
    ids = np.zeros((len(states), na, nk), dtype=np.int32)
    lb = np.zeros(ids.shape, dtype=dtype)
    ub = np.zeros(ids.shape, dtype=dtype)
    mask = np.zeros(ids.shape[:2], dtype=bool)
    actions = np.full(ids.shape[:2], -1, dtype=np.int32)
    for i, s in enumerate(states):
        for a, action in enumerate(imdp.A_id[int(s)]):
            mask[i, a] = True
            actions[i, a] = action
            successors = np.asarray(imdp.S_id[int(s)][a], dtype=np.int32)
            intervals = np.asarray(imdp.P_full[int(s)][a], dtype=dtype)
            valid = np.any(intervals > 0, axis=-1)
            k = int(valid.sum())
            ids[i, a, :k] = successors[valid]
            lb[i, a, :k] = intervals[valid, 0]
            ub[i, a, :k] = intervals[valid, 1]
            ids[i, a, k] = int(imdp.absorbing_state)
            lb[i, a, k] = imdp.P_absorbing[int(s)][a, 0]
            ub[i, a, k] = imdp.P_absorbing[int(s)][a, 1]
            if lb[i, a].sum() > 1 + 1e-5 or ub[i, a].sum() < 1 - 1e-5:
                raise ValueError(f'Infeasible intervals for state {s}, action {a}')
    return states, jnp.asarray(ids), jnp.asarray(lb), jnp.asarray(ub), jnp.asarray(mask), actions


def safety_imdp_jax(imdp, safe_mask, horizon, float_dtype, safe_upper_mask=None):
    """Return lower[h,s], upper[h,s], global policy[h,s]."""
    ns = imdp.nr_states
    safe = np.zeros(ns, dtype=bool)
    safe[:-1] = safe_mask
    safe_upper = np.zeros(ns, dtype=bool)
    safe_upper[:-1] = safe_mask if safe_upper_mask is None else safe_upper_mask
    states, ids, lb, ub, action_mask, actions = _fine_data(imdp, float_dtype)
    active = np.flatnonzero(safe[states])
    active_states = states[active]
    lower = np.zeros((horizon + 1, ns), dtype=float_dtype)
    upper = np.zeros_like(lower)
    policy = np.full((horizon + 1, ns), -1, dtype=np.int32)
    lower[0] = safe
    upper[0] = safe_upper
    for h in range(1, horizon + 1):
        lower[h] = lower[h - 1]  # overwritten for active states; unsafe remains zero
        upper[h] = safe_upper.astype(float_dtype)
        if not len(active):
            continue
        q = _robust_expectation(jnp.asarray(lower[h - 1])[ids[active]], lb[active], ub[active], False)
        q = jnp.where(action_mask[active], q, -jnp.inf)
        best = np.asarray(jnp.argmax(q, axis=1))
        lower[h, active_states] = np.asarray(jnp.max(q, axis=1))
        policy[h, active_states] = actions[active, best]
        idx = jnp.asarray(best)
        selected_ids = ids[active, idx]
        selected_lb = lb[active, idx]
        selected_ub = ub[active, idx]
        upper[h, active_states] = np.asarray(_robust_expectation(
            jnp.asarray(upper[h - 1])[selected_ids], selected_lb, selected_ub, True))
    return lower, upper, policy


def safety_2imdp_jax(imdp, safe_mask, horizon, float_dtype, safe_upper_mask=None):
    """Use the existing 2-layer greedy robust expectation, with finite-horizon recursion."""
    ns = imdp.nr_states
    safe = np.zeros(ns, dtype=bool)
    safe[:-1] = safe_mask
    safe_upper = np.zeros(ns, dtype=bool)
    safe_upper[:-1] = safe_mask if safe_upper_mask is None else safe_upper_mask
    states = np.array([int(s) for s in imdp.states if safe[int(s)] and len(imdp.A_id.get(int(s), []))], dtype=np.int32)
    na = max((len(imdp.A_id[int(s)]) for s in states), default=0)
    lower = np.zeros((horizon + 1, ns), dtype=float_dtype)
    upper = np.zeros_like(lower)
    policy = np.full((horizon + 1, ns), -1, dtype=np.int32)
    lower[0] = safe
    upper[0] = safe_upper
    if not len(states):
        return lower, upper, policy
    prep = preprocess_2imdp(imdp, states, na, float_dtype)
    keys = ('successors', 'successors_mask', 'prob_lb', 'prob_ub',
            'coarse_constraint_idx', 'coarse_mask', 'prob_lb_coarse',
            'prob_ub_coarse', 'children_indices', 'children_mask')
    arrays = [jnp.asarray(prep[k]) for k in keys]
    action_mask = jnp.asarray(prep['action_mask'])
    for h in range(1, horizon + 1):
        lower[h] = lower[h-1]
        upper[h] = upper[h-1]
        order_lo, _, child_lo, _ = compute_sort_indices(
            lower[h-1], prep['successors'], prep['successors_mask'],
            prep['children_indices'], prep['children_mask'])
        q = vmap_2layer_over_states(*arrays, jnp.asarray(order_lo),
                                    jnp.asarray(child_lo), jnp.asarray(lower[h-1]))
        q = jnp.where(action_mask, q, -jnp.inf)
        best = np.asarray(jnp.argmax(q, axis=1))
        lower[h, states] = np.asarray(jnp.max(q, axis=1))
        policy[h, states] = np.array([imdp.A_id[int(s)][int(a)] for s, a in zip(states, best)])
        _, order_hi, _, child_hi = compute_sort_indices(
            upper[h-1], prep['successors'], prep['successors_mask'],
            prep['children_indices'], prep['children_mask'])
        optimistic = vmap_2layer_over_states(*arrays, jnp.asarray(order_hi),
                                              jnp.asarray(child_hi), jnp.asarray(upper[h-1]))
        upper[h, states] = np.asarray(optimistic)[np.arange(len(states)), best]
    return lower, upper, policy
