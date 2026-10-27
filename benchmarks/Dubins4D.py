from benchmarks.models import DubinsDynamics4D
import jax.numpy as jnp
import numpy as np


class Dubins4D(DubinsDynamics4D):
    '''
    Dubin's vehicle benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics4D.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-0.5 * np.pi, -5]
        self.uMax = [0.5 * np.pi, 5]
        self.num_actions = [7, 7]

        self.partition['boundary'] = np.array([[-10, 0, -np.pi, -3], [10, 10, np.pi, 3]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = 1.35 * np.array([40, 20, 20, 20])

# --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

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
        # self.specification = ("(!unsafe) U goal")

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------        

        self.x0 = np.array([-3, 2, 0, 0])

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
        