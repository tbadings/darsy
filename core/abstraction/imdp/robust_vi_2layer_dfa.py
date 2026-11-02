"""DFA product construction and scLTL synthesis for native 2IMDP abstractions."""
import numpy as np
import jax.numpy as jnp
from .robust_vi_2layer import lower_value_iteration_2layer, upper_value_iteration_2layer

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
        Output of preprocess_2imdp.
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
