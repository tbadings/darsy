import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MonteCarloSim():
    '''
    Class to run Monte Carlo simulations on the discrete-time stochastic system closed under a fixed Markov policy.
    '''

    def __init__(self, model, partition, policy, policy_inputs, x0, iterations=100, sim_horizon=1000, random_initial_state=False, verbose=True, **kwargs):

        logger.info('=== Start Monte Carlo simulations ===')

        self.verbose = verbose

        self.model = model
        self.partition = partition

        self.policy = policy
        self.policy_inputs = policy_inputs
        self.horizon = sim_horizon
        self.iterations = iterations
        self.random_initial_state = random_initial_state

        # Predefine noise to speed up computations
        self.define_noise_values()

        self.results = {
            'satprob': -1,
            'goal_reached': np.full(self.iterations, False, dtype=bool),
            'traces': {}
        }

        # For each of the monte carlo iterations
        for m in tqdm(range(self.iterations)):
            self.results['traces'][m], self.results['goal_reached'][m] = self._runIteration(x0, m)

        self.results['satprob'] = np.mean(self.results['goal_reached'])

    def define_noise_values(self):
        '''
        Predefine the noise values to speed up computations.
        '''

        if hasattr(self.model.noise, 'sample'):
            self.noise = self.model.noise.sample(size=(self.iterations, self.horizon))
            return

        # Gaussian noise mode
        self.noise = np.random.multivariate_normal(
            self.model.noise['mean'], self.model.noise['cov'],
            (self.iterations, self.horizon))

    def _runIteration(self, x0, m):
        '''
        Run a Monte Carlo simulation from x0.

        :param x0: Initial continuous state.
        :param m: Simulation number.
        :return:
            - trace: Dictionary containing the state and input at each time step.
            - success: Boolean indicating whether goal was reached.
        '''

        # Initialize variables at start of iteration
        success = False
        trace = {'k': [], 'x': [], 'u': []}
        k = 0

        # Initialize the current simulation
        x = np.zeros((self.horizon + 1, self.model.n))
        x_tuple = np.zeros((self.horizon + 1, self.model.n)).astype(int)
        s = np.zeros(self.horizon + 1).astype(int)
        u = np.zeros((self.horizon, self.model.p))
        a = np.zeros(self.horizon).astype(int)

        # Determine initial state
        if self.random_initial_state:
            s0, _ = self.partition.x2state(x0)
            x[0] = np.random.uniform(
                low=self.partition.regions['lower_bounds'][s0],
                high=self.partition.regions['upper_bounds'][s0])

        else:
            x[0] = x0

        # Add current state, belief, etc. to trace
        trace['k'] += [0]
        trace['x'] += [x[0]]

        ######

        # For each time step in the finite time horizon
        while k <= self.horizon:

            # Determine to which region the state belongs
            s[k], in_partition = self.partition.x2state(x[k])

            if in_partition:
                # Save that state is currently in state s_current
                x_tuple[k] = self.partition.state_id2idx[s[k]]

            else:
                # Absorbing region reached
                x_tuple[k] = -1

                if self.verbose:
                    logger.info(f'- Absorbing state reached at k = {k} (x_k = {x[k]}; s_(k-1) = {s[max(0,k-1)]}; a_(k-1) = {max(0,a[k-1])}), so abort')
                return trace, success

            # If current region is the goal state ...
            if s[k] in self.partition.goal['idxs']:
                # Then abort the current iteration, as we have achieved the goal
                success = True
                if self.verbose:
                    logger.info(f'- Goal state reached (x = {x[k]})')
                return trace, success

            # If current region is in critical states...
            elif s[k] in self.partition.critical['idxs']:
                # Then abort current iteration
                if self.verbose:
                    logger.info('- Critical state reached, so abort')
                return trace, success

            # Check if we can still perform another action within the horizon
            elif k >= self.horizon:
                return trace, success

            # Retreive the action from the policy.
            if len(self.policy.shape) == 1:
                # If infinite horizon, policy does not have a time index.
                action_idx = self.policy[s[k]]
            else:
                # If finite horizon, use action for the current time step k.
                action_idx = self.policy[k, s[k]]

            if action_idx < 0:
                if self.verbose:
                    logger.info('- No enabled action in the current state (x = %s, k = %d), so abort', x[k], k)
                return trace, success

            a[k] = action_idx
            u[k] = self.policy_inputs[s[k]] if len(self.policy.shape) == 1 else self.policy_inputs[k, s[k]]

            ###

            # If loop was not aborted, we have a valid action
            if self.verbose:
                logger.info(f'In state {s[k]} (x = {x[k]}), take action {a[k]} (u = {u[k]})')
                logger.info(f'- w = {self.noise[m, k]}')

            x[k + 1] = self.model.step(x[k], u[k], self.noise[m, k])

            # Add current state, belief, etc. to trace
            trace['k'] += [k + 1]
            trace['u'] += [u[k]]
            trace['x'] += [x[k + 1]]

            # Increase iterator variable by one
            k += 1

        ######

        return trace, success


class MonteCarloSim_scLTL(MonteCarloSim):
    """
    Monte Carlo validation of a stationary product policy sigma[s,z].

    The DFA processes the initial abstract-state label before the first
    action, then processes each successor label.

    Acceptance is checked after each DFA update.

    A transition outside the partition fails unless acceptance
    occurred earlier.
    """

    def __init__(
        self,
        model,
        partition,
        policy,
        policy_inputs,
        x0,
        dfa,
        labels,
        iterations=100,
        sim_horizon=1000,
        random_initial_state=False,
        verbose=False,
        **kwargs
    ):

        logger.info("=== Start sc-LTL Monte Carlo simulations ===")

        # --------------------------------------------------
        # Store model and simulation parameters
        # --------------------------------------------------

        self.model = model
        self.partition = partition

        self.verbose = verbose
        self.horizon = int(sim_horizon)
        self.iterations = int(iterations)
        self.random_initial_state = random_initial_state

        if self.horizon < 0 or self.iterations <= 0:
            raise ValueError(
                "Require sim_horizon >= 0 and iterations > 0"
            )

        # --------------------------------------------------
        # Store DFA and policy
        # --------------------------------------------------

        self.dfa = dfa

        self.labels = np.asarray(labels, dtype=np.int64)

        self.policy = np.asarray(policy, dtype=np.int32)

        self.policy_inputs = np.asarray(
            policy_inputs,
            dtype=float
        )

        self._accepting = np.asarray(
            dfa.accepting_states,
            dtype=bool
        )

        self._transitions = np.asarray(
            dfa.transitions,
            dtype=np.int32
        )

        # --------------------------------------------------
        # Validate policy dimensions
        # --------------------------------------------------

        if self.policy.ndim != 2:
            raise ValueError(
                "sc-LTL policy must have shape "
                "(n_base_states, n_dfa_states)"
            )

        n_base_states, n_dfa_states = self.policy.shape

        if n_dfa_states != len(self._accepting):
            raise ValueError(
                "DFA state dimension does not match policy"
            )

        # Physical states exclude the absorbing state
        n_physical_states = n_base_states - 1

        if n_physical_states != self.partition.size:
            raise ValueError(
                "Policy state dimension does not match partition"
            )

        # --------------------------------------------------
        # Validate control input dimensions
        # --------------------------------------------------

        expected_shape = (
            n_physical_states,
            n_dfa_states,
            self.model.p
        )

        if self.policy_inputs.shape != expected_shape:
            raise ValueError(
                f"Invalid policy_inputs shape: "
                f"{self.policy_inputs.shape}. "
                f"Expected {expected_shape}."
            )

        # --------------------------------------------------
        # Validate labels
        # --------------------------------------------------

        if self.labels.shape != (n_base_states,):
            raise ValueError(
                "labels must contain one DFA bitmask per "
                "base state, including the absorbing state"
            )

        # --------------------------------------------------
        # Validate DFA
        # --------------------------------------------------

        if self._transitions.ndim != 2:
            raise ValueError(
                "DFA transitions must be a 2D array"
            )

        if self._transitions.shape[0] != n_dfa_states:
            raise ValueError(
                "Invalid DFA transition table"
            )

        if (
            np.any(self._transitions < 0)
            or np.any(self._transitions >= n_dfa_states)
        ):
            raise ValueError(
                "DFA transition table contains invalid states"
            )

        if (
            np.any(self.labels < 0)
            or np.any(self.labels >= self._transitions.shape[1])
        ):
            raise ValueError(
                "Invalid DFA label bitmask"
            )

        # --------------------------------------------------
        # Generate noise samples
        # --------------------------------------------------

        self.define_noise_values()

        # --------------------------------------------------
        # Initialize simulation results
        # --------------------------------------------------

        self.results = {
            "satprob": 0.0,
            "satisfied": np.zeros(
                self.iterations,
                dtype=bool
            ),
            "goal_reached": np.zeros(
                self.iterations,
                dtype=bool
            ),
            "traces": {},
        }

        # --------------------------------------------------
        # Run Monte Carlo simulations
        # --------------------------------------------------

        for m in tqdm(range(self.iterations)):

            trace, success = self._runIteration(x0, m)

            self.results["traces"][m] = trace

            self.results["satisfied"][m] = success

        # Compatibility with existing reporting functions
        self.results["goal_reached"] = (
            self.results["satisfied"].copy()
        )

        self.results["satprob"] = float(
            np.mean(self.results["satisfied"])
        )

        logger.info(
            "Empirical sc-LTL satisfaction probability: %.6f",
            self.results["satprob"]
        )

    def _runIteration(self, x0, m):
        """
        Run one Monte Carlo simulation.

        Returns
        -------
        trace : dict
            Continuous states, abstract states, DFA states,
            actions, labels, and termination status.

        success : bool
            True if the DFA reaches an accepting state.
        """

        x = np.asarray(x0, dtype=float).copy()

        # --------------------------------------------------
        # Randomize initial continuous state if requested
        # --------------------------------------------------

        if self.random_initial_state:

            s0, exists = self.partition.x2state(x)

            if not exists:
                raise ValueError(
                    "Initial state lies outside partition"
                )

            x = np.random.uniform(
                np.asarray(
                    self.partition.regions["lower_bounds"][s0]
                ),
                np.asarray(
                    self.partition.regions["upper_bounds"][s0]
                ),
            )

        # --------------------------------------------------
        # Initialize trace
        # --------------------------------------------------

        trace = {
            "k": [0],
            "x": [x.copy()],
            "u": [],
            "s": [],
            "z": [],
            "a": [],
            "labels": [],
            "status": "horizon",
        }

        z = int(self.dfa.initial_state)

        # --------------------------------------------------
        # Simulate system
        # --------------------------------------------------

        for k in range(self.horizon + 1):

            # Identify current abstract state
            s, exists = self.partition.x2state(x)

            # ----------------------------------------------
            # Check absorbing state
            # ----------------------------------------------

            if not exists:

                trace["s"].append(
                    int(self.partition.missing_state)
                )

                trace["z"].append(z)

                trace["labels"].append(0)

                trace["status"] = "outside_partition"

                if self.verbose:
                    logger.info(
                        "MC %d: partition exited at k=%d",
                        m, k
                    )

                return trace, False

            s = int(s)

            # ----------------------------------------------
            # Update DFA state
            # ----------------------------------------------

            z = int(
                self._transitions[
                    z,
                    self.labels[s]
                ]
            )

            trace["s"].append(s)
            trace["z"].append(z)
            trace["labels"].append(int(self.labels[s]))

            # ----------------------------------------------
            # Check acceptance
            # ----------------------------------------------

            if self._accepting[z]:

                trace["status"] = "accepted"

                if self.verbose:
                    logger.info(
                        "MC %d: specification satisfied at k=%d",
                        m, k
                    )

                return trace, True

            # ----------------------------------------------
            # Check simulation horizon
            # ----------------------------------------------

            if k == self.horizon:
                return trace, False

            # ----------------------------------------------
            # Retrieve product policy
            # ----------------------------------------------

            a = int(self.policy[s, z])

            if a < 0:

                trace["status"] = "no_action"

                if self.verbose:
                    logger.info(
                        "MC %d: no action at (s,z)=(%d,%d)",
                        m, s, z
                    )

                return trace, False

            # The absorbing state is excluded from
            # policy_inputs, but we have already verified
            # that s belongs to the physical partition.

            u = np.asarray(
                self.policy_inputs[s, z],
                dtype=float
            )

            if not np.all(np.isfinite(u)):

                trace["status"] = "invalid_control"

                return trace, False

            trace["a"].append(a)
            trace["u"].append(u.copy())

            if self.verbose:
                logger.info(
                    "MC %d step %d: (s,z)=(%d,%d), action=%d",
                    m, k, s, z, a
                )

            # ----------------------------------------------
            # Simulate one stochastic transition
            # ----------------------------------------------

            x = np.asarray(
                self.model.step(
                    x,
                    u,
                    self.noise[m, k]
                ),
                dtype=float
            )

            trace["x"].append(x.copy())
            trace["k"].append(k + 1)

        return trace, False