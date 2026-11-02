
from benchmarks.models import DubinsDynamics3D
import jax.numpy as jnp
import numpy as np


class Dubins3D(DubinsDynamics3D):
    """Unicycle reach-avoid benchmark with position and heading states."""

    def __init__(self, args):
        DubinsDynamics3D.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # Steering rates {-pi/2, 0, pi/2} and velocities {-1, 0, 1}.
        self.uMin = [-0.50 * np.pi, -1]
        self.uMax = [0.50 * np.pi, 1]
        self.num_actions = [3, 3]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[0, 0, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = np.array([50, 50, 50])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.array([
                    # Bottom-right corner, behind the wall.
                    [[8, 0, -np.pi], [10, 2, np.pi]]
                    ], dtype=float),

                    "r_unsafe": np.array([
                    [[4.5, 0, -np.pi], [5.5, 7, np.pi]]
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
              
        self.x0 = np.array([0.5, 0.5, 0])

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

        # Steering rates {-pi/2, 0, pi/2} and velocities {-1, 0, 1}.
        self.uMin = [-0.50 * np.pi, -1]
        self.uMax = [0.50 * np.pi, 1]
        self.num_actions = [3, 3]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[0, 0, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = np.array([50, 50, 50])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
            "r_goal": np.array([
                # Bottom-right corner, behind the wall.
                [[8, 0, -np.pi], [10, 2, np.pi]]
            ], dtype=float),

            "r_unsafe": np.array([
                [[4.5, 0, -np.pi], [5.5, 7, np.pi]]
            ], dtype=float),

            "r_wet": np.array([
                # Upper-left strip on the approach to the wall opening.
                [[0, 4, -np.pi], [2, 10, np.pi]]
            ], dtype=float),

            "r_carpet": np.array([
                # Upper-right corner, reachable before entering the goal.
                [[8, 8, -np.pi], [10, 10, np.pi]]
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
              
        self.x0 = np.array([0.5, 0.5, 0])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]
        
