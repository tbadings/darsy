"""Stochastic dynamics shared by benchmark definitions.

The step methods implement the state transition and the Lipschitz matrices
are used for conservative successor-box inflation.
"""

import logging

import jax.numpy as jnp
import numpy as np
import scipy

from benchmarks.dynamics.distributions import GaussianDistr, TriangularDistr, TruncatedGaussianBallDistr

logger = logging.getLogger(__name__)


def wrap_theta(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

class DubinsDynamics2D:
    def __init__(self, args):

        # Discretization step size
        self.tau = 1

        self.n = 2
        self.p = 2
        self.state_variables = ['x', 'y']
        self.wrap = jnp.array([False, False], dtype=bool)

        self.alpha_min = 0.85
        self.alpha_max = 0.85
        self.alpha = 0.85

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.15, 0.15])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.02, 0.02])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

    def step(self, state, action, noise):
        x, y = state[0], state[1]
        u1, u2 = action[0], action[1]
        x_next = x + self.tau * u2 * jnp.cos(u1) + noise[0]
        y_next = y + self.tau * u2 * jnp.sin(u1) + noise[1]
        
        state_next = jnp.array([x_next, y_next])

        return state_next


class DubinsDynamics3D:
    """Unicycle with position and periodic heading subject to additive noise."""

    def __init__(self, args):

        # Discretization step size
        self.tau = 1

        self.n = 3
        self.p = 2
        self.state_variables = ['x', 'y', 'angle']
        self.wrap = jnp.array([False, False, True], dtype=bool)

        self.alpha_min = 0.85
        self.alpha_max = 0.85
        self.alpha = 0.85

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.0, 0.0, 0.05])**2)
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.0, 0.0, 0.05])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])

    def step(self, state, action, noise):
        x, y, theta = state[0], state[1], state[2]
        u1, u2 = action[0], action[1]
        x_next = x + self.tau * u2 * jnp.cos(theta)
        y_next = y + self.tau * u2 * jnp.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])

        state_next = jnp.array([x_next, y_next, theta_next])

        return state_next


class DubinsDynamics4D:
    def __init__(self, args):

        # Discretization step size
        self.tau = 0.5

        self.n = 4
        self.p = 2
        self.state_variables = ['x', 'y', 'angle', 'velocity']
        self.wrap = jnp.array([False, False, True, False], dtype=bool)

        if args.model_version == 0:
            logger.info('- Load Dubins without parameter uncertainty')
            # No parameter uncertainty
            self.alpha_min = 0.85
            self.alpha_max = 0.85
            self.alpha = 0.85

            self.beta_min = 0.85
            self.beta_max = 0.85
            self.beta = 0.85
        elif args.model_version == 1:
            logger.info('- Load Dubins with uncertain parameters in the interval [0.80,0.90]')
            # High parameter uncertainty
            self.alpha_min = 0.80
            self.alpha_max = 0.90
            self.alpha = 0.85

            self.beta_min = 0.80
            self.beta_max = 0.90
            self.beta = 0.85
        else:
            logger.info('- Load Dubins with uncertain parameters in the interval [0.75,0.95]')
            # High parameter uncertainty
            self.alpha_min = 0.75
            self.alpha_max = 0.95
            self.alpha = 0.85

            self.beta_min = 0.75
            self.beta_max = 0.95
            self.beta = 0.85

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0, 0, 0.1, 0])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0, 0, 0.1, 0])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.array([
            [1.0, 0.0, 1.5, 0.5],
            [0.0, 1.0, 1.5, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, self.beta_max],
        ])

    def step(self, state, action, noise):
        x, y, theta, V = state[0], state[1], state[2], state[3]
        u1, u2 = action[0], action[1]
        x_next = x + self.tau * V * jnp.cos(theta)
        y_next = y + self.tau * V * jnp.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])
        V_next = self.beta * V + self.tau * u2

        state_next = jnp.array([x_next,
                                y_next,
                                theta_next,
                                jnp.clip(V_next, self.partition['boundary_jnp'][0][3] + 1e-3, self.partition['boundary_jnp'][1][3] - 1e-3)])
        return state_next


class DroneDynamics:
    def __init__(self, args, dim=2):

        if dim not in [2,3]:
            raise ValueError(f"DroneDynamics only supports dim in [2, 3], got {dim}")


        if dim == 2:
            self.n = 4
            self.p = 2
            self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel']
            self.wrap = jnp.array([False, False, False, False], dtype=bool)
        else:
            self.n = 6
            self.p = 3
            self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel', 'z_pos', 'z_vel']
            self.wrap = jnp.array([False, False, False, False, False, False], dtype=bool)

        self.v_min = -2.5
        self.v_max = 2.5

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
        if dim == 2:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock)

            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0]])

            # Covariance of the process noise
            if args.noise_distr == 'gaussian':
                cov = np.array([0.15, 0, 0.15, 0])**2 # From stdev to covariance
                self.noise = GaussianDistr(cov)
            elif args.noise_distr == 'triangular':
                cov = np.array([0.15, 0, 0.15, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)
            else:
                raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        else:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)

            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0], [0], [0]])

            # Covariance of the process noise
            if args.noise_distr == 'gaussian':
                cov = np.array([0.1, 0, 0.1, 0, 0.1, 0])**2 # From stdev to covariance
                self.noise = GaussianDistr(cov)
            elif args.noise_distr == 'triangular':
                cov = np.array([0.1, 0, 0.1, 0, 0.1, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)
            else:
                raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.abs(self.A)

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise
        state_next = state_next.at[1::2].set(jnp.clip(state_next[1::2], self.v_min + 1e-4, self.v_max - 1e-4))

        return state_next


class PendulumDynamics:
    def __init__(self, args):

        self.n = 2
        self.p = 1
        self.state_variables = ['angle', 'velocity']
        self.wrap = jnp.array([True, False], dtype=bool)

        # Discretization step size
        self.tau = 0.05 * 2

        # Pendulum parameters
        self.G = 10
        self.m = 1.0
        self.l = 1.0
        self.b = 0.0 # Gymnasium pendulum does not have damping

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.03, 0.1])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.03, 0.1])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.array([
            [1.15, 0.1],
            [1.5, 1.0],
        ])

    def step(self, state, action, noise):

        new_velo = (1 - self.b) * state[1] + \
                   (3 * self.G / (2 * self.l) * jnp.sin(state[0])) * self.tau + \
                   (3.0 / (self.m * self.l**2) * action[0]) * self.tau
        new_angle = wrap_theta(state[0] + self.tau * new_velo + noise[0])
        new_velo = new_velo + noise[1]

        return jnp.array([new_angle, new_velo])

    
class MountainCarDynamics:
    def __init__(self, args):

        self.n = 2
        self.p = 1
        self.state_variables = ['position', 'velocity']
        self.wrap = jnp.array([False, False], dtype=bool)

        # Discretization step size
        self.tau = 1

        # Parameters
        self.max_speed = 0.07
        self.gravity = 0.0025
        self.power = 0.0015

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.005, 0.0005])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.005, 0.0005])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.array([
            [1.0075, 1.0],
            [0.0075, 1.0],
        ])

    def step(self, state, action, noise):

        position, velocity = state[0], state[1]

        velocity = velocity + self.tau * (action[0] * self.power - self.gravity * jnp.cos(3 * position))
        velocity = jnp.clip(velocity, -self.max_speed+1e-4, self.max_speed-1e-4)
        position = position + self.tau * velocity + noise[0]
        velocity = velocity + noise[1]

        return jnp.array([position, velocity])

    
class CartPoleDynamics:
    def __init__(self, args):
        '''
        CartPole dynamics model, based on the Gymnasium CartPole environment. 
        The state consists of the cart position, cart velocity, pole angle, and pole angular velocity. 
        The action is a force applied to the cart. The dynamics are nonlinear and include process noise.
        
        Differences with the original CartPole environment:
        - The time step is 0.02 * 5 seconds instead of 0.02 seconds.
        - Original environment has no process noise.
        - Original environment has a discrete action space (left/right), while this model uses a continuous action space (force).
        '''
        

        self.n = 4
        self.p = 1
        self.state_variables = ['position', 'velocity', 'angle', 'angular_velocity']
        self.wrap = jnp.array([False, False, False, False], dtype=bool)

        # Discretization step size
        self.tau = 0.02 * 5

        # CartPole parameters (Gymnasium CartPole convention)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # Actually half the pole's length
        self.polemass_length = self.masspole * self.length

        # Covariance of the process noise (on cart position and pole angle)
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0, 0.001, 0, 0])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.005, 0, 0.005, 0])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.array([
            [1.0, 0.1, 0.0,    0.0   ],
            [0.0, 1.0, 0.096,  0.0072],
            [0.0, 0.0, 1.0,    0.1   ],
            [0.0, 0.0, 1.9179, 1.0105],
        ])

    def step(self, state, action, noise):
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        force = action[0]

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return jnp.array([x, x_dot, theta, theta_dot]) + noise


class DoubleIntegratorDynamics:
    def __init__(self, args):

        self.n = 2
        self.p = 1
        self.state_variables = ['position', 'velocity']
        self.wrap = jnp.array([False, False], dtype=bool)

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        self.A  = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        self.B  = np.array([[self.tau**2/2],
                           [self.tau]])
    
        # Disturbance matrix
        self.Q  = np.array([[0],[0],])

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.15, 0.15])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.15, 0.15])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.abs(self.A)

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise

        return state_next

    
class Test1DDynamics:
    def __init__(self, args):

        self.n = 1
        self.p = 1
        self.state_variables = ['position']
        self.wrap = jnp.array([False], dtype=bool)

        # State transition matrix
        self.A  = np.array([[1]])
        
        # Input matrix
        self.B  = np.array([[1]])

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.2])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.2])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.abs(self.A)

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise

        return state_next

class Temperature1DDynamics:
    def __init__(self, args):

        self.n = 1
        self.p = 1
        self.state_variables = ['temperature']
        self.wrap = jnp.array([False], dtype=bool)

        # State transition matrix
        self.A  = np.array([[1.021051862145107]])
        
        # Input matrix
        self.B  = np.array([[3.713548482396956e-01]])

        # Affine term
        self.b  = np.array([-6.736595886434387e-01])

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.15])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.15])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.abs(self.A)

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + self.b + noise

        return state_next

class Temperature1DDynamicsLowNoise:
    def __init__(self, args):

        self.n = 1
        self.p = 1
        self.state_variables = ['temperature']
        self.wrap = jnp.array([False], dtype=bool)

        # State transition matrix
        self.A  = np.array([[1.021051862145107]])
        
        # Input matrix
        self.B  = np.array([[3.713548482396956e-01]])

        # Affine term
        self.b  = np.array([-6.736595886434387e-01])

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.05])**2) # From stdev to covariance
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.05])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.lipschitz = np.abs(self.A)

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + self.b + noise

        return state_next

class Temperature4DDynamics:
    """Four coupled room temperatures with multiplicative ball-truncated noise."""
    def __init__(self, args, noise_truncation=4.0):
        self.n = 4
        self.p = 4
        self.state_variables = [f'temperature_room{i}' for i in range(1, 5)]
        self.wrap = jnp.zeros(4, dtype=bool)

        self.A = np.array([
            [0.9010, 0.0625, 0.0, 0.0],
            [0.0625, 0.8385, 0.0625, 0.0],
            [0.0, 0.0625, 0.8385, 0.0625],
            [0.0, 0.0, 0.0625, 0.9010],
        ])
        self.B = 0.7 * np.eye(4)
        self.b = 0.219 * np.ones(4)
        self.noise_truncation = float(noise_truncation)
        sigma = 0.02 / 3.0
        self.noise = TruncatedGaussianBallDistr(
            dimension=4, stdev=sigma, truncation=self.noise_truncation
        )
        logger.info('-Assuming truncated gaussian disturbance')

        # ||w||_2 <= truncation*sigma implies |1+w_i| <= 1+truncation*sigma.
        self.lipschitz = (1.0 + self.noise.radius) * np.abs(self.A)

    def step(self, state, action, noise):
        return ((1.0 + noise) * (jnp.asarray(self.A) @ state)
                + jnp.asarray(self.B) @ action + jnp.asarray(self.b))
