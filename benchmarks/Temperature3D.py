from benchmarks.models import Temperature3DDynamics
import jax.numpy as jnp
import numpy as np


class Temperature3D(Temperature3DDynamics):
    """
    Three-room temperature regulation with three independent binary
    heater controls and ball-truncated multiplicative Gaussian noise.
    """

    def __init__(self, args):
        Temperature3DDynamics.__init__(self, args)

        self.plot_dimensions = [0,1]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""
        
        self.partition = {}
        self.targets = {}

        # Three independent on/off heaters: 2**3 = 8 possible actions.

        self.uMin = np.zeros(3)
        self.uMax = np.ones(3)
        self.num_actions = np.full(3, 2, dtype=int)

        # State-space bounds and the safety region coincide.
        # In this benchmark, this is also the safe set

        boundary = [[18.5,18.5,18.5], [23.5,23.5,23.5]]

        self.partition['boundary'] = np.array(boundary)
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.safe_set = np.array([boundary], dtype=float)

        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = np.array([12,12,12])
                
        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.empty((0, 2, 3), dtype=float),

                    "r_unsafe": np.empty((0, 2, 3), dtype=float),
                    
                    "r_safe": self.safe_set
                            
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
            "safe",
        ]

        # --------------------------------------------------
        # Labeling function L: R -> 2^AP
        # --------------------------------------------------

        # Assign atomic propositions to each named region of interest.
        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"},
            "r_safe": {"safe"},
        }

        # --------------------------------------------------
        # Safety specification
        # --------------------------------------------------

        self.specification = "G(safe)"
        self.horizon = 15

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------


        self.x0 = np.array([21.0,21.0,21.0])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]