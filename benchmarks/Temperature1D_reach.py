from benchmarks.models import Temperature1DDynamics
import jax.numpy as jnp
import numpy as np


class Temperature1D_reach(Temperature1DDynamics):
    """Single-room temperature benchmark with binary heater control and additive process noise; reach-avoid objective."""

    def __init__(self, args):
        Temperature1DDynamics.__init__(self, args)

        self.plot_dimensions = [0]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""
        
        self.partition = {}
        self.targets = {}

        # Discrete control grid: bounds and samples per input dimension.
        self.uMin = [0]
        self.uMax = [1]
        self.num_actions = [2]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[19], [22]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = np.array([1000])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.array([
                    [[20.25], [20.75]]
                    ], dtype=float),

                    "r_unsafe": np.empty((0, 2, 1), dtype=float)
                }

        # Expose goal/unsafe geometry for the reach-avoid solver interface.
        self.goal = self.regions["r_goal"]
        self.critical = self.regions["r_unsafe"]

        # --------------------------------------------------
        # Atomic propositions
        # --------------------------------------------------

        self.atomic_propositions = [
            "goal",
            "unsafe"
        ]

        # --------------------------------------------------
        # Labeling function L: R -> 2^AP
        # --------------------------------------------------

        # Assign atomic propositions to each named region of interest.
        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"}
        }

        # --------------------------------------------------
        # Temporal-logic specification (reach-avoid or general scLTL)
        # --------------------------------------------------

        self.specification = (
            "(!unsafe) U (goal)"
        )

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------


        self.x0 = np.array([19.5])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]