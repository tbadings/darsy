
from benchmarks.models import DubinsDynamics3D
import jax.numpy as jnp
import numpy as np



def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Dubins3D(DubinsDynamics3D):
    """Dubins vehicle benchmark with planar position and heading states, controlled by turning rate and speed."""

    def __init__(self, args):
        DubinsDynamics3D.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # Discrete control grid: bounds and samples per input dimension.
        self.uMin = [-0.50 * np.pi, -3]
        self.uMax = [0.50 * np.pi, 3]
        self.num_actions = [7, 5]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[-10, -10, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = 3 * np.array([20, 20, 11])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.array([
                    [[-10, 5, -np.pi], [-5, 10, np.pi]]
                    ], dtype=float),

                    "r_unsafe": np.array([
                    [[-10, -1, -np.pi], [-1, 1, np.pi]],
                    [[-1, -5, -np.pi], [1, 5, np.pi]]
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
              
        self.x0 = np.array([-7.5, -7.5, 0])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]
        
class Dubins3D_scLTL(DubinsDynamics3D):
    """Three-state Dubins vehicle with a temporal-logic objective involving goal, unsafe, wet, and carpet regions."""

    def __init__(self, args):
        DubinsDynamics3D.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # Discrete control grid: bounds and samples per input dimension.
        self.uMin = [-0.50 * np.pi, -3]
        self.uMax = [0.50 * np.pi, 3]
        self.num_actions = [7, 5]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[-10, -10, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = 3 * np.array([20, 20, 11])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
            "r_goal": np.array([
                [[-10, -10, -np.pi], [-6, -6, np.pi]]
            ], dtype=float),

            "r_unsafe": np.array([
                [[-2, -2, -np.pi], [2, 6, np.pi]]
            ], dtype=float),

            "r_wet": np.array([
                [[-10, -2, -np.pi], [-6, 10, np.pi]]
            ], dtype=float),

            "r_carpet": np.array([
                [[6, -10, -np.pi], [10, -6, np.pi]]
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
            "unsafe",
            "wet",
            "carpet"
        ]

        # --------------------------------------------------
        # Labeling function L: R -> 2^AP
        # --------------------------------------------------

        # Assign atomic propositions to each named region of interest.
        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"},
            "r_wet": {"wet"},
            "r_carpet": {"carpet"}
        }

        # --------------------------------------------------
        # Temporal-logic specification (reach-avoid or general scLTL)
        # --------------------------------------------------

        self.specification = (
            "((!unsafe) & (!wet | ((!goal) U carpet))) "
            "U (goal & (!unsafe))"
        )
              
        self.x0 = np.array([-7.5, -7.5, 0])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]
        
