from benchmarks.models import DubinsDynamics2D
import jax.numpy as jnp
import numpy as np




class Dubins2D_scLTL(DubinsDynamics2D):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics2D.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-np.pi, 0]
        self.uMax = [np.pi, 0.1]
        self.num_actions = [8, 5]

        self.partition['boundary'] = np.array([[0, 0], [1, 1]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = 3 * np.array([10, 10])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        self.regions = {
            "r_goal": np.array([
                [[0, 0], [0.2, 0.2]]
            ], dtype=float),

            "r_unsafe": np.array([
                [[0.4, 0.4], [0.6, 0.8]]
            ], dtype=float),

            "r_wet": np.array([
                [[0, 0.4], [0.2, 1]]
            ], dtype=float),

            "r_carpet": np.array([
                [[0.8, 0], [1, 0.2]]
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

        self.x0 = np.array([-9.0, 9.0])

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
    
