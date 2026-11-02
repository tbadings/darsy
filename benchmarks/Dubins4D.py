from benchmarks.models import DubinsDynamics4D
import jax.numpy as jnp
import numpy as np


class Dubins4D(DubinsDynamics4D):
    """Four-state Dubins vehicle with planar position, heading, and velocity; two control inputs."""

    def __init__(self, args):
        DubinsDynamics4D.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # Discrete control grid: bounds and samples per input dimension.
        self.uMin = [-0.5 * np.pi, -5]
        self.uMax = [0.5 * np.pi, 5]
        self.num_actions = [7, 7]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[-10, 0, -np.pi, -3], [10, 10, np.pi, 3]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        # Round scaled resolution to integer cell counts.
        self.partition['number_per_dim'] = np.rint(1.35 * np.array([40, 20, 20, 20])).astype(int)

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.array([
                    [[6, 6, -np.pi, -3], [9, 9, np.pi, 3]]
                    ], dtype=float),

                    "r_unsafe": np.array([
                    [[4, 5, -2 * np.pi, -3], [5, 10, 2 * np.pi, 3]],
                    [[-1, 0, -2 * np.pi, -3], [0, 5, 2 * np.pi, 3]],
                    [[-5, 4, -2 * np.pi, -3], [-1, 5, 2 * np.pi, 3]],
                    ], dtype=float)
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

        self.x0 = np.array([-3, 2, 0, 0])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]
        