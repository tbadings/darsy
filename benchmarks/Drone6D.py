from benchmarks.models import DroneDynamics
import jax.numpy as jnp
import numpy as np


class Drone6D(DroneDynamics):
    '''
    Drone benchmark, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]
        # Position dimensions, shown as a 3D trace plot in addition to the 2D projection.
        self.plot_dimensions_3d = [0, 2, 4]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1, -1]
        self.uMax = [1, 1, 1]
        self.num_actions = [5, 5, 5]

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-17, v_min, -9, v_min, -7, v_min], 
                                               [17, v_max, 9, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([68, 14, 36, 14, 28, 14])

        
        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        self.regions = {
                    "r_goal": np.array([
            [[11, v_min, 1, v_min, -7, v_min], [15, v_max, 5, v_max, -3, v_max]]
        ], dtype=float),

                    "r_unsafe": np.array([
                        # Hole 1
                        [[-11, v_min, -1, v_min, -7, v_min], [-5, v_max, 9, v_max, -5, v_max]],
                        [[-11, v_min, 5, v_min, -5, v_min], [-5, v_max, 9, v_max, 5, v_max]],
                        [[-11, v_min, -1, v_min, -5, v_min], [-5, v_max, 3, v_max, 3, v_max]],

                        # # Hole 2
                        [[-1, v_min, 1, v_min, -7, v_min], [3, v_max, 9, v_max, -1, v_max]],
                        [[-1, v_min, 1, v_min, 3, v_min], [3, v_max, 9, v_max, 5, v_max]],
                        [[-1, v_min, 1, v_min, -1, v_min], [3, v_max, 3, v_max, 3, v_max]],
                        [[-1, v_min, 7, v_min, -1, v_min], [3, v_max, 9, v_max, 3, v_max]],

                        # # Tower
                        [[-1, v_min, -3, v_min, -7, v_min], [3, v_max, 1, v_max, 7, v_max]],

                        # # Wall between routes
                        [[3, v_min, -3, v_min, -7, v_min], [9, v_max, 1, v_max, -1, v_max]],

                        # # Long route obstacles
                        # [[-11, v_min, -5, v_min, -7, v_min], [-7, v_max, -1, v_max, 1, v_max]],
                        [[-1, v_min, -9, v_min, -7, v_min], [3, v_max, -3, v_max, -5, v_max]],

                        # Overhanging
                        [[-1, v_min, -9, v_min, 3, v_min], [3, v_max, -3, v_max, 7, v_max]],

                        # Small last obstacle
                        [[11, v_min, -9, v_min, -7, v_min], [15, v_max, -5, v_max, -5, v_max]],

                        # Obstacle next to goal
                        [[9, v_min, 5, v_min, -7, v_min], [15, v_max, 9, v_max, 1, v_max]],
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


        self.x0 = np.array([-14.5, 0.01, 6, 0.01, 2, 0.01])

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



class Drone6D_small(DroneDynamics):
    '''
    Drone benchmark, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]
        # Position dimensions, shown as a 3D trace plot in addition to the 2D projection.
        self.plot_dimensions_3d = [0, 2, 4]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1, -1]
        self.uMax = [1, 1, 1]
        self.num_actions = [5, 5, 5]

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-7, v_min, -7, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([14, 10, 14, 10, 14, 10])

        self.goal = np.array([
            [[3, v_min, 3, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max]]
        ], dtype=float)

        self.critical = np.array([
            [[-7, v_min, 1, v_min, -7, v_min], [-1, v_max, 3, v_max, 7, v_max]],
            [[3, v_min, -7, v_min, -7, v_min], [7, v_max, -3, v_max, 7, v_max]],
        ], dtype=float)

        self.x0 = np.array([-5.5, 0.01, -5.5, 0.01, 0.01, 0.01])

        return