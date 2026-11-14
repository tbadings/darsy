"""Named results for abstraction and synthesis; tuple unpacking is supported."""

from typing import NamedTuple

from core.automaton.dfa import DFA
from core.interfaces import FloatArray, IntArray


class ProbabilityIntervals(NamedTuple):
    """Arrays keyed by physical state; interval bounds are [lower, upper].

    A counts actions, M fine successors, and Mc coarse successors.
    Coarse dictionaries are empty when coarse estimation is disabled.
    """

    fine_bounds: dict[int, FloatArray]              # (A, M, 2)
    successor_ids: dict[int, IntArray]              # (A, M)
    action_ids: dict[int, IntArray]                 # (A,)
    absorbing_bounds: dict[int, FloatArray]         # (A, 2)
    coarse_successor_ids: dict[int, IntArray]       # (A, Mc)
    coarse_bounds: dict[int, FloatArray]            # (A, Mc, 2)


class SynthesisResult(NamedTuple):
    """Worst/best values of the chosen policy and its global action IDs.

    Shapes are (S+1,) or (S+1, Z) with a DFA; a finite horizon adds a leading
    H+1 axis indexed by remaining steps. S excludes the sink. Policy -1 means
    no chosen action; both bounds evaluate the returned policy.
    """

    lower: FloatArray
    upper: FloatArray
    policy: IntArray


class ScltlSynthesisResult(NamedTuple):
    """Unbounded product synthesis plus its DFA and label encoding.

    Values/policy have shape (S+1, Z), labels (S+1,). initial_dfa_state follows
    the initial state's label; bounds and action IDs follow SynthesisResult.
    """

    lower: FloatArray
    upper: FloatArray
    policy: IntArray
    dfa: DFA
    initial_dfa_state: int
    labels: IntArray
