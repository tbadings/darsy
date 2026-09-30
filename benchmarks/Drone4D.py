from benchmarks.models import DroneDynamics
import jax.numpy as jnp
import numpy as np


class Drone4D(DroneDynamics):
    '''
    Drone benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args)

        self.plot_dimensions = [0, 2]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1]
        self.uMax = [1 ,1]
        self.num_actions = [5, 5]

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-20.0, v_min, -20.0, v_min], [20.0, v_max, 20.0, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([80, 10, 80, 10])

        self.goal = np.array([
            [[12.0, v_min, 12.0, v_min], [18.0, v_max, 18.0, v_max]]
        ], dtype=float)

        self.critical = np.array([
            # Lower wall with right-side opening
            [[-20.0, v_min, -7.0, v_min], [8.0, v_max, -4.0, v_max]],
            # Upper wall with left-side opening
            [[-8.0, v_min, 4.0, v_min], [20.0, v_max, 7.0, v_max]],
            # Lower lane obstacle
            [[-4.0, v_min, -14.0, v_min], [4.0, v_max, -10.0, v_max]],
            # Upper lane obstacle 
            [[-4.0, v_min, 10.0, v_min], [4.0, v_max, 14.0, v_max]],
        ], dtype=float)

        self.x0 = np.array([-15.0, 0.01, -15.0, 0.01])

        return