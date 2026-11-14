import numpy as np

from core.abstraction.partition import CoarsePartition
from core.interfaces import BoolArray, FloatArray, IntArray


class IMDP:
    """Fine/coarse transition bounds and labels for physical states plus a sink."""

    states_of_interest: dict[str, BoolArray]
    partial_region_cells: dict[str, BoolArray]
    region_labels: dict[str, set[str]]
    atomic_propositions: list[str]

    def __init__(
        self, states: IntArray, goal_regions: BoolArray, critical_regions: BoolArray,
        P_full: dict[int, FloatArray], S_id: dict[int, IntArray], A_id: dict[int, IntArray],
        P_absorbing: dict[int, FloatArray], coarse_partition: CoarsePartition,
        Q_id: dict[int, IntArray], P_coarse: dict[int, FloatArray],
    ) -> None:
        """Store arrays by reference; states and boolean masks have shape (S,).

        State IDs are 0..S-1; sink ID S gives nr_states = S+1. Transition
        dictionaries follow ProbabilityIntervals in the same field order.
        coarse_partition.parents maps each physical state to its coarse block.
        """
        self.states = states
        self.goal_regions = goal_regions
        self.critical_regions = critical_regions
        self.P_full = P_full
        self.S_id = S_id
        self.A_id = A_id
        self.P_absorbing = P_absorbing

        # Coarse layer
        self.Q_id = Q_id
        self.P_coarse = P_coarse

        # Fine/coarse maps
        self.parents = coarse_partition.parents

        # Define absorbing state
        self.absorbing_state = np.max(self.states) + 1

        # Number of states
        self.nr_states = len(self.states) + 1
