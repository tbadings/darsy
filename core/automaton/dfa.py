import re
import subprocess
from dataclasses import dataclass

import numpy as np


SPOT_EXECUTABLE = (
    "/home/ibon/miniconda3/envs/spot-env/bin/ltl2tgba"
)


@dataclass
class DFA:
    """
    Complete deterministic finite automaton.

    States are indexed from 0 to n_states - 1.

    The transition table has shape:
        (n_states, 2**n_ap)

    Atomic propositions are encoded using bitmasks.
    """

    states: np.ndarray
    initial_state: int
    accepting_states: np.ndarray
    terminal_states: np.ndarray
    atomic_propositions: list[str]
    transitions: np.ndarray

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_ap(self):
        return len(self.atomic_propositions)

    def encode_label(self, true_propositions):
        """
        Encode a set of true atomic propositions as a bitmask.
        """
        true_propositions = set(true_propositions)

        unknown = true_propositions.difference(
            self.atomic_propositions
        )

        if unknown:
            raise ValueError(
                f"Unknown atomic propositions: {unknown}"
            )

        label = 0

        for i, ap in enumerate(self.atomic_propositions):
            if ap in true_propositions:
                label |= 1 << i

        return label


    def next_state(self, state, label):
        """
        Evaluate the DFA transition function.

        label is an integer bitmask.
        """
        return int(self.transitions[state, label])

    def is_accepting(self, state):
        return bool(self.accepting_states[state])

    def is_sink(self, state):
        return bool(self.terminal_states[state])


def _evaluate_guard(guard, valuation):
    """
    Evaluate an explicit HOA transition guard.

    Supported operators:
        !   negation
        &   conjunction
        |   disjunction
        t   true
        f   false

    Atomic propositions are indexed by integers.
    """

    tokens = re.findall(
        r"\d+|[!&|()]|[tf]",
        guard.replace(" ", "")
    )

    if "".join(tokens) != guard.replace(" ", ""):
        raise ValueError(f"Unsupported HOA guard: {guard}")

    position = 0

    def parse_or():
        nonlocal position

        value = parse_and()

        while position < len(tokens) and tokens[position] == "|":
            position += 1
            rhs = parse_and()
            value = value or rhs

        return value

    def parse_and():
        nonlocal position

        value = parse_not()

        while position < len(tokens) and tokens[position] == "&":
            position += 1
            rhs = parse_not()
            value = value and rhs

        return value

    def parse_not():
        nonlocal position

        if position >= len(tokens):
            raise ValueError("Unexpected end of guard")

        token = tokens[position]
        position += 1

        if token == "!":
            return not parse_not()

        if token == "(":
            value = parse_or()

            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("Unmatched parenthesis")

            position += 1
            return value

        if token == "t":
            return True

        if token == "f":
            return False

        if token.isdigit():
            return bool(valuation & (1 << int(token)))

        raise ValueError(f"Unexpected token: {token}")

    result = parse_or()

    if position != len(tokens):
        raise ValueError(f"Invalid guard: {guard}")

    return result

def scLTL_to_DFA(
    formula,
    spot_executable=SPOT_EXECUTABLE
):
    """
    Translate an sc-LTL formula into a complete DFA.

    Uses Spot to construct a deterministic, complete,
    state-based Buchi automaton.

    This function requires all accepting states to be
    absorbing. It rejects outputs that do not satisfy
    this condition.

    Parameters
    ----------
    formula : str
        sc-LTL specification.

    spot_executable : str
        Path to Spot's ltl2tgba executable.

    Returns
    -------
    DFA
        Deterministic finite automaton.
    """

    command = [
        spot_executable,
        "-f", formula,
        "--deterministic",
        "--complete",
        "--buchi",
        "--state-based-acceptance",
        "--hoaf"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    hoa = result.stdout

    # --------------------------------------------------
    # Parse HOA header
    # --------------------------------------------------

    states_match = re.search(
        r"^States:\s*(\d+)",
        hoa,
        re.MULTILINE
    )

    initial_match = re.search(
        r"^Start:\s*(\d+)\s*$",
        hoa,
        re.MULTILINE
    )

    ap_match = re.search(
        r'^AP:\s*(\d+)(.*)$',
        hoa,
        re.MULTILINE
    )

    acceptance_match = re.search(
        r"^Acceptance:\s*(.*)$",
        hoa,
        re.MULTILINE
    )

    if not all([
        states_match,
        initial_match,
        ap_match,
        acceptance_match
    ]):
        raise ValueError("Invalid or unsupported HOA header")

    n_states = int(states_match.group(1))
    initial_state = int(initial_match.group(1))

    n_ap = int(ap_match.group(1))

    atomic_propositions = re.findall(
        r'"((?:\\.|[^"\\])*)"',
        ap_match.group(2)
    )

    if len(atomic_propositions) != n_ap:
        raise ValueError("Could not parse atomic propositions")

    if acceptance_match.group(1).strip() != "1 Inf(0)":
        raise ValueError(
            "Expected a single Buchi acceptance condition"
        )

    if "state-acc" not in hoa:
        raise ValueError(
            "Expected state-based acceptance"
        )

    if "deterministic" not in hoa:
        raise ValueError(
            "Spot did not produce a deterministic automaton"
        )

    # --------------------------------------------------
    # Initialize DFA
    # --------------------------------------------------

    states = np.arange(n_states)

    accepting_states = np.zeros(
        n_states,
        dtype=bool
    )

    transitions = np.full(
        (n_states, 2**n_ap),
        -1,
        dtype=np.int32
    )

    # --------------------------------------------------
    # Parse states and transitions
    # --------------------------------------------------

    current_state = None

    for line in hoa.splitlines():

        line = line.strip()

        if line.startswith("State:"):

            match = re.match(
                r"State:\s*(\d+)(.*)",
                line
            )

            current_state = int(match.group(1))

            if "{0}" in match.group(2):
                accepting_states[current_state] = True

        elif line.startswith("["):

            if current_state is None:
                raise ValueError(
                    "Transition without source state"
                )

            match = re.match(
                r"\[(.*?)\]\s*(\d+)\s*$",
                line
            )

            if match is None:
                raise ValueError(
                    f"Unsupported HOA transition: {line}"
                )

            guard = match.group(1)
            destination = int(match.group(2))

            for valuation in range(2**n_ap):

                if _evaluate_guard(guard, valuation):

                    previous = transitions[
                        current_state, valuation
                    ]

                    if previous != -1:
                        raise ValueError(
                            "Overlapping transition guards"
                        )

                    transitions[
                        current_state, valuation
                    ] = destination
    
    sink_states = np.all(
        transitions == states[:, None],
        axis=1
    )
    
    # --------------------------------------------------
    # Validate completeness
    # --------------------------------------------------

    if np.any(transitions == -1):
        raise ValueError(
            "The automaton is not complete"
        )

    # --------------------------------------------------
    # Validate absorbing accepting states
    # --------------------------------------------------

    for q in np.flatnonzero(accepting_states):

        if not np.all(transitions[q] == q):
            raise ValueError(
                "Accepting states are not absorbing. "
                "This translation requires a more "
                "general good-prefix construction."
            )

    return DFA(
        states=states,
        initial_state=initial_state,
        accepting_states=accepting_states,
        terminal_states=sink_states,
        atomic_propositions=atomic_propositions,
        transitions=transitions
    )