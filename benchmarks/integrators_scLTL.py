from benchmarks.models import DoubleIntegratorDynamics
import jax.numpy as jnp
import numpy as np


class DoubleIntegrator(DoubleIntegratorDynamics):

    def __init__(self, args):
        super().__init__(args)

        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        """
        Set abstraction parameters, regions of interest,
        labeling function, and sc-LTL specification.
        """

        self.partition = {}
        self.targets = {}

        # --------------------------------------------------
        # Control inputs
        # --------------------------------------------------

        self.uMin = [-5]
        self.uMax = [5]
        self.num_actions = [21]

        # --------------------------------------------------
        # State-space partition
        # --------------------------------------------------

        self.partition["boundary"] = np.array([
            [-21, -10.5],
            [21, 10.5]
        ])

        self.partition["boundary_jnp"] = jnp.array(
            self.partition["boundary"]
        )

        self.partition["number_per_dim"] = np.array([51, 51])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        self.regions = {
            "r_goal": np.array([
                [[-4, -2], [4, 2]]
            ], dtype=float),

            "r_unsafe": np.empty((0, 2, 2), dtype=float),

            "r_wet": np.array([
                [[5, -5], [10, 0]]
            ], dtype=float),

            "r_carpet": np.array([
                [[-10, 2], [-5, 5]]
            ], dtype=float)
        }

        # Preserve compatibility with reach-avoid synthesis
        self.goal = self.regions["r_goal"]
        self.critical = self.regions["r_unsafe"]

        # --------------------------------------------------
        # Atomic propositions
        # --------------------------------------------------

        self.atomic_propositions = [
            "goal",
            "unsafe",
            "wet",
            "carpet"
        ]

        # --------------------------------------------------
        # Labeling function L: R -> 2^AP
        # --------------------------------------------------

        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"},
            "r_wet": {"wet"},
            "r_carpet": {"carpet"}
        }

        # --------------------------------------------------
        # sc-LTL specification
        # --------------------------------------------------

        self.specification = (
            "((!unsafe) & (!wet | ((!goal) U carpet))) "
            "U (goal & (!unsafe))"
        )

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------

        self.x0 = np.array([0, -8])

    def labeling_function(self, r):
        """
        Return the atomic propositions that hold
        in region r.

        Parameters
        ----------
        r : str
            Identifier of a region of interest.

        Returns
        -------
        set[str]
            Atomic propositions true in region r.
        """

        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")

        return self.labels[r]
