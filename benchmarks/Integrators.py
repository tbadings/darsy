from benchmarks.models import DoubleIntegratorDynamics
import jax.numpy as jnp
import numpy as np


class DoubleIntegrator(DoubleIntegratorDynamics):

    def __init__(self, args):
        DoubleIntegratorDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''
        
        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-5]
        self.uMax = [5]
        self.num_actions = [21]

        self.partition['boundary'] = np.array([[-21, -10.5], [21, 10.5]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([51, 51])

        self.goal = np.array([
            [[-4, -2], [4, 2]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([0, -8])

        return
