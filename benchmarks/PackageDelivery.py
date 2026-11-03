from benchmarks.models import PackageDeliveryDynamics
import jax.numpy as jnp
import numpy as np


class PackageDelivery(PackageDeliveryDynamics):
    """ARCH-COMP22 package delivery benchmark: pick up a parcel at p1 and deliver it at p3, losing it at p2."""

    def __init__(self, args):
        PackageDeliveryDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # --------------------------------------------------
        # Control inputs
        # --------------------------------------------------

        # The benchmark leaves the input unbounded; we use [-1, 1]^2, as IMPaCT did in ARCH-COMP25.
        self.uMin = [-1, -1]
        self.uMax = [1, 1]
        self.num_actions = [3, 3]

        # --------------------------------------------------
        # State-space partition
        # --------------------------------------------------

        # State-space bounds used for abstraction.
        self.partition["boundary"] = np.array([
            [-6, -6],
            [6, 6]
        ])

        self.partition["boundary_jnp"] = jnp.array(
            self.partition["boundary"]
        )

        # Nominal partition resolution along each state dimension (cell width 0.5, as in ARCH-COMP25).
        self.partition["number_per_dim"] = np.array([400, 400])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
            # Pick-up location
            "r_p1": np.array([
                [[5, -1], [6, 1]]
            ], dtype=float),

            # Region where a carried parcel is lost
            "r_p2": np.array([
                [[0, -5], [1, 1]]
            ], dtype=float),

            # Delivery location
            "r_p3": np.array([
                [[-4, -4], [-2, -3]]
            ], dtype=float)
        }

        # No goal or unsafe sets: the task is specified only through the labels below.
        self.goal = np.empty((0, 2, 2), dtype=float)
        self.critical = np.empty((0, 2, 2), dtype=float)

        # --------------------------------------------------
        # Atomic propositions
        # --------------------------------------------------

        self.atomic_propositions = [
            "p1",
            "p2",
            "p3"
        ]

        # --------------------------------------------------
        # Labeling function L: R -> 2^AP
        # --------------------------------------------------

        # Assign atomic propositions to each named region of interest.
        self.labels = {
            "r_p1": {"p1"},
            "r_p2": {"p2"},
            "r_p3": {"p3"}
        }

        # --------------------------------------------------
        # sc-LTL specification
        # --------------------------------------------------

        # Pick up the parcel at p1, then deliver it at p3 without passing through p2 on the way.
        self.specification = "F (p1 & ((!p2) U p3))"

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------

        # As used by SySCoRe in ARCH-COMP25.
        self.x0 = np.array([-5, -5])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]
