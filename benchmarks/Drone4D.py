from benchmarks.models import DroneDynamics
import jax.numpy as jnp
import numpy as np


class Drone4D(DroneDynamics):
    """Planar double-integrator drone benchmark with two position and two velocity states and two acceleration controls."""

    def __init__(self, args):
        DroneDynamics.__init__(self, args)

        self.plot_dimensions = [0, 2]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # Discrete control grid: bounds and samples per input dimension.
        self.uMin = [-1, -1]
        self.uMax = [1 ,1]
        self.num_actions = [5, 5]

        v_min = self.v_min
        v_max = self.v_max

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[-20.0, v_min, -20.0, v_min], [20.0, v_max, 20.0, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = np.array([80, 10, 80, 10])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.array([
                    [[12.0, v_min, 12.0, v_min], [18.0, v_max, 18.0, v_max]]
                    ], dtype=float),

                    "r_unsafe": np.array([
                    # Lower wall with right-side opening
                    [[-20.0, v_min, -7.0, v_min], [8.0, v_max, -4.0, v_max]],
                    # Upper wall with left-side opening
                    [[-8.0, v_min, 4.0, v_min], [20.0, v_max, 7.0, v_max]],
                    # Lower lane obstacle
                    [[-4.0, v_min, -14.0, v_min], [4.0, v_max, -10.0, v_max]],
                    # Upper lane obstacle 
                    [[-4.0, v_min, 10.0, v_min], [4.0, v_max, 14.0, v_max]],
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


        self.x0 = np.array([-15.0, 0.01, -15.0, 0.01])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]
            