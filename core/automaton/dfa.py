import re
import subprocess
from dataclasses import dataclass

import numpy as np

from core.interfaces import BoolArray, IntArray


@dataclass
class DFA:
    """Complete DFA with transitions (Z, 2**n_ap) indexed by proposition bitmasks."""

    states: IntArray
    initial_state: int
    accepting_states: BoolArray
    atomic_propositions: list[str]
    transitions: IntArray

    @property
    def n_states(self) -> int:
        return len(self.states)


def _evaluate_guard(guard: str, valuation: int) -> bool:
    """Evaluate an HOA guard using !, &, |, parentheses, t/f, and proposition indices."""
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

def scltl_to_dfa(formula: str) -> DFA:
    """Translate with Spot's ltl2tgba to a complete DFA with absorbing accepting states.

    Raise FileNotFoundError if Spot is missing, CalledProcessError if it fails,
    and ValueError for unsupported or incomplete HOA output.
    """
    command = [
        "ltl2tgba",
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
        atomic_propositions=atomic_propositions,
        transitions=transitions
    )
