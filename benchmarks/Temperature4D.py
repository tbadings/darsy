from benchmarks.models import Temperature4DDynamics
import jax.numpy as jnp
import numpy as np


class Temperature4D(Temperature4DDynamics):
    '''
    4-room temperature regulation benchmark.
    The state is 4D, with each element representing the temperature of each room. 
    The control input is a 4D vector, with each element taking values in {0,1}. The
    values represent whether the heater of that room is ON or OFF.
    The disturbance is multiplicative and its distribution is gaussian and truncated in a ball.
    The specification is safety with a time horizon of 20 time units
    '''

    def __init__(self, args):
        Temperature4DDynamics.__init__(self, args)

        self.plot_dimensions = [0,1]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the specification.
        '''
        
        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative

        self.uMin = np.zeros(4)
        self.uMax = np.ones(4)
        self.num_actions = np.full(4, 2, dtype=int)

        # Specify the boundary of the state set to partition.
        # In this benchmark, this is also the safe set

        boundary = [[18.5,18.5,18.5,18.5], [23.5,23.5,23.5,23.5]]

        self.partition['boundary'] = np.array(boundary)
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([12])
        self.safe_set = np.array([boundary])
        
        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        self.regions = {
                    "r_goal": np.array([
                    
                    ], dtype=float),

                    "r_unsafe": np.array([
                    ], dtype=float),
                    
                    "r_safe": self.safe_set
                            
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
            "safe",
        ]

        # --------------------------------------------------
        # Labeling function L: R -> 2^AP
        # --------------------------------------------------

        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"},
            "r_safe": {"safe"},
        }

        # --------------------------------------------------
        # Safety specification
        # --------------------------------------------------

        self.specification = "G(safe)"
        self.horizon = 20

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------


        self.x0 = np.array([21.0,21.0,21.0,21.0])

    def labeling_function(self, r):
                """
                Returns the atomic propositions that hold
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