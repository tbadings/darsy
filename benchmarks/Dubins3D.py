
from benchmarks.models import DubinsDynamics3D
import jax.numpy as jnp
import numpy as np



def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Dubins3D(DubinsDynamics3D):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics3D.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-0.50 * np.pi, -3]
        self.uMax = [0.50 * np.pi, 3]
        self.num_actions = [7, 5]

        self.partition['boundary'] = np.array([[-10, -10, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = 3 * np.array([20, 20, 11])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        self.regions = {
                    "r_goal": np.array([
                    [[-10, 5, -np.pi], [-5, 10, np.pi]]
                    ], dtype=float),

                    "r_unsafe": np.array([
                    [[-10, -1, -np.pi], [-1, 1, np.pi]],
                    [[-1, -5, -np.pi], [1, 5, np.pi]]
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
            "unsafe"
        ]

        # --------------------------------------------------
        # Labeling function L: R -> 2^AP
        # --------------------------------------------------

        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"}
        }

        # --------------------------------------------------
        # sc-LTL specification
        # --------------------------------------------------

        self.specification = (
            "(!unsafe) U (goal)"
        )
              
        self.x0 = np.array([-7.5, -7.5, 0])

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
        
