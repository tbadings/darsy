"""Robust dynamic programming for IMDPs and 2IMDPs.

One driver solves reach-avoid and scLTL objectives over a finite or an unbounded horizon, and
safety objectives over a finite horizon, for both abstractions; only the robust expectation kernel
differs (see robust_expectation.py).

Every objective is solved on the product of the abstraction with an automaton: reach-avoid and
safety use a trivial one-state automaton, scLTL uses the DFA of the specification. The intervals
are never copied per automaton state; only the successor's automaton state depends on z.

The states are processed in chunks that are read straight from the abstraction (imdp.P_full,
imdp.S_id, ...), so memory stays at the abstraction plus one chunk. When all chunks fit in
max_resident_bytes, they are built once and kept on the device instead.
"""

import itertools
import logging
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.abstraction.imdp.robust_expectation import expectation_2imdp, expectation_imdp

logger = logging.getLogger(__name__)


# ============================================================================
# OBJECTIVES
# ============================================================================


@dataclass
class Objective:
    """Values to compute on the product of the abstraction with an automaton.

    Rows are product states (s, z), where s = imdp.absorbing_state is the sink. Free rows are
    updated by the robust Bellman operator; all other rows keep their initial value.
    """
    next_z: np.ndarray      # (S+1, Z) int: automaton state after entering state s' from automaton state z
    free: np.ndarray        # (S+1, Z) bool: rows updated by the Bellman operator
    lower_init: np.ndarray  # (S+1, Z): lower-bound values at horizon 0
    upper_init: np.ndarray  # (S+1, Z): upper-bound values at horizon 0
    reach: bool             # reach-type (values rise from below) or safety-type (values fall from above)


def _has_actions(imdp):
    """Per state, including the sink: whether any action is enabled."""
    has = np.array([len(imdp.A_id.get(s, [])) > 0 for s in range(imdp.nr_states)], dtype=bool)
    has[imdp.absorbing_state] = False
    return has


def _state_mask(imdp, mask):
    """Boolean mask over the physical states, extended with False for the sink."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:  # no such regions
        mask = np.zeros(imdp.nr_states - 1, dtype=bool)
    return np.append(mask, False)


def reach_avoid_objective(imdp):
    """(!unsafe) U goal: goal states are worth one, critical states and the sink zero."""
    goal = _state_mask(imdp, imdp.goal_regions)
    critical = _state_mask(imdp, imdp.critical_regions)
    init = goal.astype(float)[:, None]
    return Objective(next_z=np.zeros((imdp.nr_states, 1), dtype=np.int32),
                     free=(~goal & ~critical & _has_actions(imdp))[:, None],
                     lower_init=init, upper_init=init.copy(), reach=True)


def safety_objective(imdp, safe_mask, safe_upper_mask=None):
    """G(safe): safe states start at one, all others and the sink are worth zero.

    The lower bound uses safe_mask (cells inside the safe set); the upper bound may use a larger
    safe_upper_mask (e.g. cells that only touch the safe set).
    """
    safe = _state_mask(imdp, safe_mask)
    safe_upper = safe if safe_upper_mask is None else _state_mask(imdp, safe_upper_mask)
    return Objective(next_z=np.zeros((imdp.nr_states, 1), dtype=np.int32),
                     free=(safe & _has_actions(imdp))[:, None],
                     lower_init=safe.astype(float)[:, None],
                     upper_init=safe_upper.astype(float)[:, None], reach=False)


def scLTL_objective(imdp, dfa, labels):
    """Product with a complete good-prefix DFA whose accepting states are absorbing.

    The automaton reads the label of the state being entered. Accepting product states are worth
    one. Rejecting sinks are worth zero, and so is the out-of-partition sink: it keeps the current
    automaton state, which is never accepting for a free row.
    """
    transitions = np.asarray(dfa.transitions, dtype=np.int32)
    accepting = np.asarray(dfa.accepting_states, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    nz, ns, sink = len(accepting), imdp.nr_states, imdp.absorbing_state
    if labels.shape != (ns,):
        raise ValueError('labels must include the physical and absorbing states')
    if transitions.shape != (nz, 2 ** len(dfa.atomic_propositions)):
        raise ValueError('Invalid DFA transition table')
    if np.any(transitions < 0) or np.any(transitions >= nz):
        raise ValueError('DFA transitions must be complete')
    if np.any(labels < 0) or np.any(labels >= transitions.shape[1]):
        raise ValueError('Invalid label bitmask')
    if np.any(transitions[accepting] != np.flatnonzero(accepting)[:, None]):
        raise ValueError('Accepting DFA states must be absorbing')

    # Accepting states and rejecting self-loops end the run
    terminal = accepting | np.all(transitions == np.arange(nz)[:, None], axis=1)
    next_z = transitions[:, labels].T.copy()  # next_z[s, z] = transitions[z, labels[s]]
    next_z[sink] = np.arange(nz)
    init = np.broadcast_to(accepting, (ns, nz)).astype(float)
    init[sink] = 0.0
    return Objective(next_z=next_z, free=_has_actions(imdp)[:, None] & ~terminal[None, :],
                     lower_init=init, upper_init=init.copy(), reach=True)


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


def initial_product_state(s0, dfa, labels):
    """Initial product state under the successor-label convention."""
    s0 = int(s0)
    z0 = int(dfa.initial_state)
    return s0, int(dfa.transitions[z0, int(labels[s0])])


# ============================================================================
# CHUNKED BELLMAN OPERATOR
# ============================================================================


@partial(jax.jit, static_argnames=('expectation', 'maximize'))
def _q_values(V, next_z, zs, data, expectation, maximize):
    """q[c, j, a]: worst (best) expected value of V for row (state c, automaton state zs[j]) and action a."""
    succ = data['succ']                                         # (C, A, K)
    values = V[succ[..., None], next_z[succ][..., zs]]          # (C, A, K, Z')
    values = jnp.moveaxis(values, -1, 1)                        # (C, Z', A, K)
    q = expectation(values, {k: v[:, None] for k, v in data.items()}, maximize)
    return jnp.where(data['action_mask'][:, None, :], q, -jnp.inf)


@partial(jax.jit, static_argnames=('expectation', 'maximize'))
def _policy_values(V, next_z, zs, data, actions, expectation, maximize):
    """Worst (best) expected value of V for row (state c, automaton state zs[j]) under action actions[c, j]."""
    rows = jnp.arange(actions.shape[0])[:, None]
    selected = {k: v[rows, actions] for k, v in data.items() if k != 'action_mask'}  # (C, Z', ...)
    succ = selected['succ']                                     # (C, Z', K)
    values = V[succ, next_z[succ, zs[None, :, None]]]           # (C, Z', K)
    return expectation(values, selected, maximize)


class _Bellman:
    """Robust Bellman sweeps over the free rows of an objective, chunk by chunk.

    A chunk holds C states with their intervals as arrays of shape (C, A, K): states, actions,
    successors (the absorbing state is the last successor). All states share the same successor
    window, as built by compute_probability_intervals. Padding rows point to the sink, which is
    never free. Values V and the policy (local action indices) have shape (S+1, Z).
    """

    def __init__(self, imdp, objective, two_layer, float_dtype, chunk_entries, max_resident_bytes):
        self.imdp = imdp
        self.two_layer = two_layer
        self.float_dtype = float_dtype
        self.expectation = expectation_2imdp if two_layer else expectation_imdp
        self.sink = int(imdp.absorbing_state)
        self.next_z = jnp.asarray(objective.next_z, dtype=jnp.int32)

        # Automaton states and physical states with free rows
        zs = np.flatnonzero(objective.free.any(axis=0)).astype(np.int32)
        states = np.flatnonzero(objective.free.any(axis=1)).astype(np.int32)
        self.zs = jnp.asarray(zs)
        self.num_actions = max((len(imdp.A_id[int(s)]) for s in states), default=1)
        self.num_successors = imdp.S_id[int(states[0])].shape[1] + 1 if len(states) else 1
        if two_layer:
            self.num_blocks = imdp.Q_id[int(states[0])].shape[1] if len(states) else 1
            self.parents = np.append(np.asarray(imdp.parents, dtype=np.int32), -1)  # the sink has no block

        # Fixed-size chunks; the last one is padded with the sink
        per_state = len(zs) * self.num_actions * self.num_successors
        size = max(1, min(len(states), chunk_entries // max(per_state, 1)))
        padded = np.full(-(-len(states) // size) * size, self.sink, dtype=np.int32)
        padded[:len(states)] = states
        self.chunk_states = [jnp.asarray(c) for c in padded.reshape(-1, size)]
        self.chunk_free = [jnp.asarray(objective.free[np.asarray(c)][:, zs]) for c in self.chunk_states]

        # Keep the chunks on the device when they fit, otherwise rebuild them every sweep
        itemsize = np.dtype(float_dtype).itemsize
        per_entry = 4 + 2 * itemsize + (5 if two_layer else 0)
        self.resident = padded.size * self.num_actions * self.num_successors * per_entry <= max_resident_bytes
        self._cache = {}
        logger.info('- Robust DP: %d states with free rows in %d chunks of %d states (%s)',
                    len(states), len(self.chunk_states), size,
                    'kept on device' if self.resident else 'streamed from the abstraction')

    def _load(self, states):
        """Intervals of the given states as padded arrays (C, A, K)."""
        imdp, sink = self.imdp, self.sink
        shape = (len(states), self.num_actions, self.num_successors)
        succ = np.full(shape, sink, dtype=np.int32)
        lb = np.zeros(shape, dtype=self.float_dtype)
        ub = np.zeros(shape, dtype=self.float_dtype)
        action_mask = np.zeros(shape[:2], dtype=bool)
        if self.two_layer:
            block_shape = shape[:2] + (self.num_blocks,)
            block_ids = np.zeros(block_shape, dtype=np.int32)
            block_lb = np.zeros(block_shape, dtype=self.float_dtype)
            block_ub = np.zeros(block_shape, dtype=self.float_dtype)

        for c, s in enumerate(np.asarray(states)):
            if s == sink:  # padding row
                continue
            n = len(imdp.A_id[int(s)])
            action_mask[c, :n] = True
            succ[c, :n, :-1] = imdp.S_id[int(s)]
            lb[c, :n, :-1] = imdp.P_full[int(s)][..., 0]
            ub[c, :n, :-1] = imdp.P_full[int(s)][..., 1]
            lb[c, :n, -1] = imdp.P_absorbing[int(s)][:, 0]
            ub[c, :n, -1] = imdp.P_absorbing[int(s)][:, 1]
            if self.two_layer:
                block_ids[c, :n] = imdp.Q_id[int(s)]
                block_lb[c, :n] = imdp.P_coarse[int(s)][..., 0]
                block_ub[c, :n] = imdp.P_coarse[int(s)][..., 1]

        data = dict(succ=succ, lb=lb, ub=ub, action_mask=action_mask)
        if self.two_layer:
            # Unused window slots have zero intervals; the absorbing successor always exists
            mask = ub > 0
            mask[..., -1] = True
            block_mask = block_ub > 0
            block = self._block_positions(succ, block_ids, block_mask)
            data.update(mask=mask, block=block, block_lb=block_lb, block_ub=block_ub, block_mask=block_mask)
        return data

    def _block_positions(self, succ, block_ids, block_mask):
        """Position of each successor's coarse block in its pair's block window, or -1 if not there.

        Looks up (pair, block id) keys in sorted order, so memory grows with successors + blocks.
        """
        num_blocks = block_ids.shape[-1]
        if num_blocks == 0:
            return np.full(succ.shape, -1, dtype=np.int32)
        pairs = np.arange(succ.shape[0] * succ.shape[1]).reshape(succ.shape[:2] + (1,))
        span = int(self.parents.max()) + 2  # block ids lie in [0, span - 2], the sink's parent is -1
        block_keys = np.where(block_mask, pairs * span + block_ids, -2).ravel()  # -2: no successor matches
        successor_keys = (pairs * span + self.parents[succ]).ravel()
        order = np.argsort(block_keys, kind='stable')  # first occurrence wins
        pos = np.minimum(np.searchsorted(block_keys[order], successor_keys), len(order) - 1)
        found = block_keys[order][pos] == successor_keys
        return np.where(found, order[pos] % num_blocks, -1).reshape(succ.shape).astype(np.int32)

    def _data(self, i):
        if i in self._cache:
            return self._cache[i]
        data = jax.device_put(self._load(self.chunk_states[i]))
        if self.resident:
            self._cache[i] = data
        return data

    def improve(self, V, policy, *, in_place, keep_unless_better):
        """One Bellman sweep: the best action and its value for every free row.

        :param in_place: Read values updated earlier in the same sweep (faster convergence); False
            reads only V, as needed for a finite horizon.
        :param keep_unless_better: Keep a row's value and action unless strictly improved (for
            reach-type objectives, so that the policy cannot loop without making progress).
        """
        V_new = V
        for i, (states, free) in enumerate(zip(self.chunk_states, self.chunk_free)):
            q = _q_values(V_new if in_place else V, self.next_z, self.zs, self._data(i),
                          self.expectation, False)
            best_value, best_action = jnp.max(q, axis=-1), jnp.argmax(q, axis=-1).astype(jnp.int32)
            rows = (states[:, None], self.zs[None, :])
            update = free & (best_value > V_new[rows]) if keep_unless_better else free
            V_new = V_new.at[rows].set(jnp.where(update, best_value, V_new[rows]))
            policy = policy.at[rows].set(jnp.where(update, best_action, policy[rows]))
        return V_new, policy

    def evaluate(self, V, policy, *, in_place, maximize, monotone):
        """One sweep of V under a fixed policy (worst case, or best case if maximize).

        :param monotone: Never decrease a value (for iterations that rise from below).
        """
        V_new = V
        for i, (states, free) in enumerate(zip(self.chunk_states, self.chunk_free)):
            rows = (states[:, None], self.zs[None, :])
            value = _policy_values(V_new if in_place else V, self.next_z, self.zs, self._data(i),
                                   jnp.maximum(policy[rows], 0), self.expectation, maximize)
            if monotone:
                value = jnp.maximum(value, V_new[rows])
            V_new = V_new.at[rows].set(jnp.where(free, value, V_new[rows]))
        return V_new

    def global_actions(self, policy, free):
        """Map local action indices to global action ids (-1 on rows that are not free)."""
        table = np.full((self.sink + 1, self.num_actions), -1, dtype=np.int32)
        for s in range(self.sink):
            ids = self.imdp.A_id.get(s, [])
            table[s, :len(ids)] = ids
        policy = np.asarray(policy)
        states = np.arange(self.sink + 1).reshape((-1, 1))
        return np.where(free & (policy >= 0), table[states, np.maximum(policy, 0)], -1)


# ============================================================================
# DYNAMIC PROGRAMMING
# ============================================================================


def _max_change(V, V_old):
    return float(jnp.max(jnp.abs(V - V_old)))


def _value_iteration(bellman, V, policy, *, keep_unless_better, epsilon, max_iter, initial_row):
    """Iterate Bellman sweeps until the values change less than epsilon."""
    pbar = tqdm(desc='Value iteration', unit='it', dynamic_ncols=True)
    for iteration in range(max_iter):
        V_old = V
        V, policy = bellman.improve(V, policy, in_place=True, keep_unless_better=keep_unless_better)
        error = _max_change(V, V_old)
        pbar.update(1)
        pbar.set_postfix(_progress(V, initial_row, error))
        if error < epsilon:
            logger.info('Value iteration converged after %d iterations', iteration + 1)
            break
    else:
        logger.warning('Value iteration reached max_iter=%d', max_iter)
    pbar.close()
    return V, policy


def _policy_iteration(bellman, V, policy, *, epsilon, max_iter, initial_row, satprob):
    """Robust policy iteration with partial policy evaluation, for reach-type objectives.

    Early iterations evaluate the policy only partially (at most 10 + 10 * iteration sweeps, capped
    at 100); once improvements stall, each policy is evaluated until convergence. Improvement stops
    as soon as the initial row reaches satprob.
    """
    pbar = tqdm(desc='Policy iteration', unit='it', dynamic_ncols=True)
    fully_evaluated = False
    for iteration in range(max_iter):
        max_sweeps = None if fully_evaluated else min(10 + 10 * iteration, 100)
        for sweep in itertools.count():
            V_old = V
            V = bellman.evaluate(V, policy, in_place=True, maximize=False, monotone=True)
            error = _max_change(V, V_old)
            if error < epsilon or (max_sweeps is not None and sweep >= max_sweeps):
                break
        pbar.update(1)
        pbar.set_postfix(_progress(V, initial_row, error))

        if initial_row is not None and float(V[initial_row]) >= satprob:
            logger.info('Policy is above the satisfaction threshold %.2f after %d iterations',
                        satprob, iteration + 1)
            if not fully_evaluated:
                fully_evaluated = True
                continue  # evaluate this policy until convergence, then stop
            break

        V_old = V
        V, policy = bellman.improve(V, policy, in_place=True, keep_unless_better=True)
        if float(jnp.max(V - V_old)) < epsilon:
            if fully_evaluated:
                logger.info('Policy iteration converged after %d iterations', iteration + 1)
                break
            fully_evaluated = True
    else:
        logger.warning('Policy iteration reached max_iter=%d', max_iter)
    pbar.close()
    return V, policy


def _evaluate_upper(bellman, U, policy, *, monotone, epsilon, max_iter):
    """Best-case values of a fixed policy, iterated until convergence."""
    for iteration in range(max_iter):
        U_old = U
        U = bellman.evaluate(U, policy, in_place=True, maximize=True, monotone=monotone)
        if _max_change(U, U_old) < epsilon:
            logger.info('Upper-bound evaluation converged after %d iterations', iteration + 1)
            break
    else:
        logger.warning('Upper-bound evaluation reached max_iter=%d', max_iter)
    return U


def _progress(V, initial_row, error):
    postfix = {'error': f'{error:.2e}'}
    if initial_row is not None:
        postfix['v0'] = f'{float(V[initial_row]):.6f}'
    return postfix


def robust_dp(imdp, objective, *, two_layer=False, horizon=None, policy_iteration=False,
              initial_row=None, satprob=1.0, epsilon=1e-6, max_iter=10000, float_dtype=np.float64,
              chunk_entries=2 ** 22, max_resident_bytes=2 ** 31):
    """Lower and upper bounds on the objective, and a policy that maximizes the lower bound.

    The upper bound is the best case of that policy.

    :param imdp: IMDP (two_layer=False) or 2IMDP abstraction (two_layer=True).
    :param objective: Objective, see reach_avoid_objective, safety_objective, scLTL_objective.
    :param horizon: None to iterate until convergence (reach-type objectives only); H for exactly H steps.
    :param policy_iteration: Use policy iteration (without a horizon).
    :param initial_row: Product state (s0, z0), for progress output and satprob.
    :param satprob: Stop improving the policy once the initial row reaches this value (policy iteration).
    :param chunk_entries: Interval entries per chunk (states * automaton states * actions * successors).
    :param max_resident_bytes: Keep all chunks on the device if they fit in this many bytes.
    :return: lower, upper, policy (global action ids, -1 on rows that are not free). Without a
        horizon, each has shape (S+1, Z); with horizon H, shape (H+1, S+1, Z), where index h is the
        number of remaining steps.
    """
    if horizon is None and not objective.reach:
        # Values falling from above converge to an over-approximation of the lower bound
        raise ValueError('Safety-type objectives require a finite horizon.')

    bellman = _Bellman(imdp, objective, two_layer, float_dtype, chunk_entries, max_resident_bytes)
    lower = jnp.asarray(objective.lower_init, dtype=float_dtype)
    upper = jnp.asarray(objective.upper_init, dtype=float_dtype)
    policy = jnp.zeros(objective.free.shape, dtype=jnp.int32)
    if initial_row is not None:
        initial_row = tuple(int(i) for i in initial_row)
        if not objective.free[initial_row]:  # e.g. a goal state: its value is fixed, so satprob says nothing
            satprob = np.inf

    if horizon is None:
        # Infinite horizon: iterate until convergence, then evaluate the upper bound of the final policy
        if policy_iteration:
            # Policy iteration
            lower, policy = _policy_iteration(bellman, lower, policy, epsilon=epsilon, max_iter=max_iter,
                                              initial_row=initial_row, satprob=satprob)
        else:
            # Value iteration
            lower, policy = _value_iteration(bellman, lower, policy, keep_unless_better=True,
                                             epsilon=epsilon, max_iter=max_iter, initial_row=initial_row)
        upper = _evaluate_upper(bellman, upper, policy, monotone=True,
                                epsilon=epsilon, max_iter=max_iter)
        return np.asarray(lower), np.asarray(upper), bellman.global_actions(policy, objective.free)

    # Finite horizon: step h reads only the values of step h - 1
    lowers, uppers, policies = [lower], [upper], [policy]
    for _ in tqdm(range(horizon), desc='Finite-horizon DP', unit='step', dynamic_ncols=True):
        lower, policy = bellman.improve(lowers[-1], policies[-1], in_place=False, keep_unless_better=False)
        upper = bellman.evaluate(uppers[-1], policy, in_place=False, maximize=True, monotone=False)
        lowers.append(lower)
        uppers.append(upper)
        policies.append(policy)
    policy = np.stack([np.asarray(p) for p in policies])
    policy[0] = -1  # no step left to take
    return (np.stack([np.asarray(v) for v in lowers]), np.stack([np.asarray(v) for v in uppers]),
            bellman.global_actions(policy, objective.free))


# ============================================================================
# SPECIFICATIONS
# ============================================================================


def solve_reach_avoid(imdp, *, two_layer=False, initial_state=None, **kwargs):
    """Reach-avoid synthesis: returns lower, upper, policy, each of shape (S+1,), or (H+1, S+1)
    with horizon H, where index h is the number of remaining steps."""
    initial_row = None if initial_state is None else (initial_state, 0)
    lower, upper, policy = robust_dp(imdp, reach_avoid_objective(imdp), two_layer=two_layer,
                                     initial_row=initial_row, **kwargs)
    return lower[..., 0], upper[..., 0], policy[..., 0]


def solve_safety(imdp, safe_mask, safe_upper_mask, horizon, *, two_layer=False, **kwargs):
    """Finite-horizon safety: returns lower, upper, policy of shape (H+1, S+1), index h = remaining steps."""
    lower, upper, policy = robust_dp(imdp, safety_objective(imdp, safe_mask, safe_upper_mask),
                                     two_layer=two_layer, horizon=horizon, **kwargs)
    return lower[..., 0], upper[..., 0], policy[..., 0]


def solve_scLTL(imdp, dfa, labels, *, two_layer=False, initial_state=None, **kwargs):
    """scLTL synthesis on the product with the DFA: returns lower, upper, policy of shape (S+1, Z)."""
    initial_row = None if initial_state is None else initial_product_state(initial_state, dfa, labels)
    return robust_dp(imdp, scLTL_objective(imdp, dfa, labels), two_layer=two_layer,
                     initial_row=initial_row, **kwargs)
