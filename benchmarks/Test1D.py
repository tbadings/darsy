from benchmarks.models import Test1DDynamics
import jax.numpy as jnp
import numpy as np


class Test1D(Test1DDynamics):
    '''
    Dubin's vehicle benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        Test1DDynamics.__init__(self, args)

        self.plot_dimensions = [0]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''
        
        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1]
        self.uMax = [1]
        self.num_actions = [3]

        self.partition['boundary'] = np.array([[-20], [20]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([100])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        self.regions = {
                    "r_goal": np.array([
                    [[-1], [1]]
                    ], dtype=float),

                    "r_unsafe": np.array([
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


        self.x0 = np.array([-15])

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