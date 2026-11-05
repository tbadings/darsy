from benchmarks.models import PendulumDynamics
import jax.numpy as jnp
import numpy as np
from matplotlib import animation


class Pendulum(PendulumDynamics):
    """Pendulum benchmark with angular position and angular velocity and scalar torque input."""

    def __init__(self, args):
        PendulumDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]
        # State dimension animated by plot_trajectory_gif.
        self.gif_dimensions = [0]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # Discrete control grid: bounds and samples per input dimension.
        self.uMin = [-2]
        self.uMax = [2]
        self.num_actions = [3]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[-np.pi, -8], [np.pi, 8]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = np.array([200, 200])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.array([
                    [[-0.1*np.pi, -1], [0.1*np.pi, 1]]
                    ], dtype=float),

                    "r_unsafe": np.empty((0, 2, 2), dtype=float)
                }

        # Goal/unsafe geometry for the reach-avoid specification
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

        # Assign atomic propositions to each named region of interest.
        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"}
        }

        # --------------------------------------------------
        # Reach-avoid specification
        # --------------------------------------------------

        self.specification = (
            "(!unsafe) U (goal)"
        )

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------


        self.x0 = np.array([0.99*np.pi, 0])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]


    def plot_trajectory_gif(self, trajectory, filename="pendulum_trajectory.gif"):
        """
        Plots a trajectory of the pendulum as an animation and stores it as a gif.

        Args:
            trajectory: np.ndarray of shape (T, 2), where each row is [theta, theta_dot]
            filename: Output filename for the gif
        """
        import matplotlib.pyplot as plt

        # Prepare figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=5)
        time_template = 'step {:d}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        L = 1.0  # Assume pendulum length is 1

        # fig.tight_layout()

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            theta = trajectory[i]
            x = L * np.sin(theta)
            y = L * np.cos(theta)  # Flip the sign to invert the y-axis
            line.set_data([0, x], [0, y])
            time_text.set_text(time_template.format(i))
            return line, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=len(trajectory), interval=50, blit=True, init_func=init
        )

        ani.save(filename, writer='pillow')
        plt.close(fig)



class PendulumHard(PendulumDynamics):
    """Pendulum benchmark with angular position and angular velocity and scalar torque input."""

    def __init__(self, args):
        PendulumDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]
        # State dimension animated by plot_trajectory_gif.
        self.gif_dimensions = [0]

        self.set_spec()

    def set_spec(self):
        """Configure actions, partition, labeled regions, and specification."""

        self.partition = {}
        self.targets = {}

        # Discrete control grid: bounds and samples per input dimension.
        self.uMin = [-1.2]
        self.uMax = [1.2]
        self.num_actions = [3]

        # State-space bounds used for abstraction.
        self.partition['boundary'] = np.array([[-np.pi, -8], [np.pi, 8]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        # Nominal partition resolution along each state dimension.
        self.partition['number_per_dim'] = np.array([200, 200])

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Each region is a union of boxes with shape (N, 2, state_dim).
        self.regions = {
                    "r_goal": np.array([
                    [[-0.1*np.pi, -1], [0.1*np.pi, 1]]
                    ], dtype=float),

                    "r_unsafe": np.empty((0, 2, 2), dtype=float)
                }

        # Goal/unsafe geometry for the reach-avoid specification
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

        # Assign atomic propositions to each named region of interest.
        self.labels = {
            "r_goal": {"goal"},
            "r_unsafe": {"unsafe"}
        }

        # --------------------------------------------------
        # Reach-avoid specification
        # --------------------------------------------------

        self.specification = (
            "(!unsafe) U (goal)"
        )

        # --------------------------------------------------
        # Initial condition
        # --------------------------------------------------


        self.x0 = np.array([0.99*np.pi, 0])

    def labeling_function(self, r):
        """Return the atomic propositions assigned to a named region."""
        if r not in self.regions:
            raise ValueError(f"Unknown region: {r}")
        return self.labels[r]


    def plot_trajectory_gif(self, trajectory, filename="pendulum_trajectory.gif"):
        """
        Plots a trajectory of the pendulum as an animation and stores it as a gif.

        Args:
            trajectory: np.ndarray of shape (T, 2), where each row is [theta, theta_dot]
            filename: Output filename for the gif
        """
        import matplotlib.pyplot as plt

        # Prepare figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=5)
        time_template = 'step {:d}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        L = 1.0  # Assume pendulum length is 1

        # fig.tight_layout()

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            theta = trajectory[i]
            x = L * np.sin(theta)
            y = L * np.cos(theta)  # Flip the sign to invert the y-axis
            line.set_data([0, x], [0, y])
            time_text.set_text(time_template.format(i))
            return line, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=len(trajectory), interval=50, blit=True, init_func=init
        )

        ani.save(filename, writer='pillow')
        plt.close(fig)