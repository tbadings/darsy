"""Single-layer IMDP x good-prefix DFA robust scLTL synthesis.

Install as core/abstraction/imdp/rvi_jax_dfa.py.
The ambiguity set contains ONLY fine transition intervals (no coarse constraints).
"""
import logging
import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def preprocess_imdp_dfa(imdp, dfa, labels, float_dtype=np.float32):
    """Build padded transitions of nonterminal product states (s,z)."""
    transitions = np.asarray(dfa.transitions, dtype=np.int32)
    accepting = np.asarray(dfa.accepting_states, dtype=bool)
    nz = len(accepting)
    ns = int(imdp.nr_states)
    sink = int(imdp.absorbing_state)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (ns,):
        raise ValueError('labels must include the physical and absorbing states')
    if transitions.ndim != 2 or transitions.shape != (nz, 2 ** len(dfa.atomic_propositions)):
        raise ValueError('Invalid DFA transition table')
    if np.any(transitions < 0) or np.any(transitions >= nz):
        raise ValueError('DFA transitions must be complete')
    if np.any(labels < 0) or np.any(labels >= transitions.shape[1]):
        raise ValueError('Invalid label bitmask')
    if np.any(transitions[accepting] != np.flatnonzero(accepting)[:, None]):
        raise ValueError('Accepting DFA states must be absorbing')

    # Rejecting self-loop DFA states are terminal failures.
    terminal = accepting | np.all(transitions == np.arange(nz)[:, None], axis=1)
    rows = [(int(s), int(z)) for s in imdp.states if int(s) != sink
            and len(imdp.A_id.get(int(s), [])) > 0
            for z in np.flatnonzero(~terminal)]
    max_a = max((len(imdp.A_id[int(s)]) for s, _ in rows), default=0)
    max_k = max((imdp.S_id[s].shape[1] + 1 for s, _ in rows), default=1)
    shape = (len(rows), max_a, max_k)
    successors = np.zeros(shape, dtype=np.int32)
    lb = np.zeros(shape, dtype=float_dtype)
    ub = np.zeros(shape, dtype=float_dtype)
    action_mask = np.zeros(shape[:2], dtype=bool)
    action_ids = np.full(shape[:2], -1, dtype=np.int32)
    for row, (s, z) in enumerate(rows):
        ids = np.asarray(imdp.S_id[s], dtype=np.int32)
        probs = np.asarray(imdp.P_full[s], dtype=float_dtype)
        sink_prob = np.asarray(imdp.P_absorbing[s], dtype=float_dtype)
        actions = np.asarray(imdp.A_id[s], dtype=np.int32)
        for a, global_a in enumerate(actions):
            action_mask[row, a] = True
            action_ids[row, a] = global_a
            # A zero/zero fine entry is an unused rectangular-window slot.
            valid = np.any(probs[a] > 0, axis=-1)
            fine_ids = ids[a, valid]
            fine_p = probs[a, valid]
            if len(np.unique(fine_ids)) != len(fine_ids):
                raise ValueError(f'Duplicate successors at ({s},{a})')
            count = len(fine_ids)
            successors[row, a, :count] = fine_ids * nz + transitions[z, labels[fine_ids]]
            lb[row, a, :count] = fine_p[:, 0]
            ub[row, a, :count] = fine_p[:, 1]
            # Out-of-partition successor is a failure, even if its empty label
            # could otherwise cause the DFA to enter an accepting state.
            successors[row, a, count] = sink * nz + z
            lb[row, a, count] = sink_prob[a, 0]
            ub[row, a, count] = sink_prob[a, 1]
            if lb[row, a].sum() > 1 + 1e-5 or ub[row, a].sum() < 1 - 1e-5:
                raise ValueError(f'Infeasible probability intervals at ({s},{a})')
    return dict(rows=np.asarray([s * nz + z for s, z in rows], dtype=np.int32),
                successors=successors, lb=lb, ub=ub,
                action_mask=action_mask, action_ids=action_ids,
                accepting=accepting, terminal=terminal, ns=ns, nz=nz)


@jax.jit
def _robust_expectation(successor_values, lb, ub, maximize):
    """Exact greedy solution of min/max p @ V subject to lb <= p <= ub, sum p=1."""
    order = jnp.argsort(jnp.where(maximize, -successor_values, successor_values), axis=-1)
    lo = jnp.take_along_axis(lb, order, axis=-1)
    hi = jnp.take_along_axis(ub, order, axis=-1)
    vals = jnp.take_along_axis(successor_values, order, axis=-1)
    budget = jnp.maximum(0.0, 1.0 - jnp.sum(lo, axis=-1, keepdims=True))
    capacity = jnp.maximum(0.0, hi - lo)
    preceding = jnp.cumsum(capacity, axis=-1) - capacity
    extra = jnp.minimum(capacity, jnp.maximum(0.0, budget - preceding))
    return jnp.clip(jnp.sum((lo + extra) * vals, axis=-1), 0.0, 1.0)


@jax.jit
def _bellman_all(V, successors, lb, ub, maximize):
    return _robust_expectation(V[successors], lb, ub, maximize)


def robust_scLTL_synthesis_imdp_jax(imdp, dfa, labels, *, float_dtype=np.float32,
                                     epsilon=1e-6, max_iter=10000):
    """Maximize worst-case satisfaction, then evaluate its best-case probability.

    Returns (V_lower, V_upper, policy), each of shape (nr_states, nr_dfa_states).
    The policy contains GLOBAL action IDs and -1 for terminal product states.
    """
    data = preprocess_imdp_dfa(imdp, dfa, labels, float_dtype)
    ns, nz = data['ns'], data['nz']
    initial = np.broadcast_to(data['accepting'][None, :], (ns, nz)).astype(float_dtype).copy()
    # Sink is always a failure, including when a DFA state is accepting:
    # only physical accepting product states count as successes.
    initial[int(imdp.absorbing_state), :] = 0
    lower = jnp.asarray(initial.reshape(-1))
    rows = jnp.asarray(data['rows'])
    policy = np.full((ns * nz,), -1, dtype=np.int32)
    if len(rows) == 0:
        return initial, initial.copy(), policy.reshape(ns, nz)

    succ = jnp.asarray(data['successors'])
    lb = jnp.asarray(data['lb'])
    ub = jnp.asarray(data['ub'])
    mask = jnp.asarray(data['action_mask'])
    actions = jnp.asarray(data['action_ids'])
    chosen = jnp.zeros((len(rows),), dtype=jnp.int32)

    # Monotone lower VI, retaining a maximizing policy at each iteration.
    for iteration in range(max_iter):
        q = _bellman_all(lower, succ, lb, ub, False)
        q = jnp.where(mask, q, -jnp.inf)
        best = jnp.argmax(q, axis=1)
        values = jnp.maximum(lower[rows], jnp.max(q, axis=1))
        improvement = values > lower[rows]
        chosen = jnp.where(improvement, best, chosen)
        new_lower = lower.at[rows].set(values)
        error = float(jnp.max(jnp.abs(new_lower - lower)))
        lower = new_lower
        if error < epsilon:
            logger.info('IMDP scLTL lower VI converged after %d iterations', iteration + 1)
            break
    else:
        logger.warning('IMDP scLTL lower VI reached max_iter=%d', max_iter)

    policy[np.asarray(data['rows'])] = np.asarray(
        jnp.take_along_axis(actions, chosen[:, None], axis=1)[:, 0]
    )

    # Evaluate the fixed lower-optimal policy optimistically.
    upper = jnp.asarray(initial.reshape(-1))
    selected_succ = jnp.take_along_axis(succ, chosen[:, None, None], axis=1)[:, 0]
    selected_lb = jnp.take_along_axis(lb, chosen[:, None, None], axis=1)[:, 0]
    selected_ub = jnp.take_along_axis(ub, chosen[:, None, None], axis=1)[:, 0]
    for iteration in range(max_iter):
        q = _robust_expectation(upper[selected_succ], selected_lb, selected_ub, True)
        new_upper = upper.at[rows].set(jnp.maximum(upper[rows], q))
        error = float(jnp.max(jnp.abs(new_upper - upper)))
        upper = new_upper
        if error < epsilon:
            logger.info('IMDP scLTL upper VI converged after %d iterations', iteration + 1)
            break
    else:
        logger.warning('IMDP scLTL upper VI reached max_iter=%d', max_iter)

    return (np.asarray(lower).reshape(ns, nz),
            np.asarray(upper).reshape(ns, nz), policy.reshape(ns, nz))
