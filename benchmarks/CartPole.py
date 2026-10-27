from benchmarks.models import CartPoleDynamics
import jax.numpy as jnp
import numpy as np
from matplotlib import animation


class CartPole(CartPoleDynamics):
    '''
    CartPole benchmark (cart position, cart velocity, pole angle, pole angular velocity).
    '''

    def __init__(self, args):
        CartPoleDynamics.__init__(self, args)

        # Plot the cart position against the pole angle
        self.plot_dimensions = [0, 2]
        # State dimensions (cart position, pole angle) animated by plot_trajectory_gif.
        self.gif_dimensions = [0, 2]

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u (force on the cart), both positive and negative
        self.uMin = [-10]
        self.uMax = [10]
        self.num_actions = [41]

        self.partition['boundary'] = np.array([[-2.4, -3.0, -0.25, -3.0],
                                               [2.4, 3.0, 0.25, 3.0]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = 20*np.array([12, 8, 6, 10]) + 1

        # --------------------------------------------------
        # Regions of interest
        # --------------------------------------------------

        # Goal: balance the pole upright, close to the center of the track

        self.regions = {
                    "r_goal": np.array([
                    [[1, -3.0, -0.05, -3.0], [2.4, 3.0, 0.05, 3.0]]
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

        # Start with the pole tilted away from upright
        self.x0 = np.array([-1, 0.0, 0, 0.0])


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

    def plot_trajectory_gif(self, trajectory, filename="cartpole_trajectory.gif"):
        """
        Plots a trajectory of the cartpole as an animation and stores it as a gif.

        Args:
            trajectory: np.ndarray of shape (T, 2), where each row is [cart_position, pole_angle]
            filename: Output filename for the gif
        """
        import matplotlib.pyplot as plt

        trajectory = np.asarray(trajectory, dtype=float)
        x_positions = trajectory[:, 0]
        thetas = trajectory[:, 1]

        x_lim = 2.4
        pole_len = 1.0  # Visual length of the pole
        cart_w, cart_h = 0.4, 0.2

        # Prepare figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(-x_lim - 1, x_lim + 1)
        ax.set_ylim(-0.5, pole_len + 0.5)
        ax.set_aspect('equal')
        ax.grid()

        # Track
        ax.axhline(0, color='black', lw=1)

        from matplotlib.patches import Rectangle
        cart = Rectangle((0, 0), cart_w, cart_h, facecolor='black')
        ax.add_patch(cart)
        pole, = ax.plot([], [], '-', lw=4, color='tab:brown')
        time_template = 'step {:d}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            cart.set_xy((-cart_w / 2, 0))
            pole.set_data([], [])
            time_text.set_text('')
            return cart, pole, time_text

        def animate(i):
            x = x_positions[i]
            theta = thetas[i]
            cart.set_xy((x - cart_w / 2, 0))
            # Pole pivots at the top-center of the cart; theta=0 is upright
            px = x + pole_len * np.sin(theta)
            py = cart_h + pole_len * np.cos(theta)
            pole.set_data([x, px], [cart_h, py])
            time_text.set_text(time_template.format(i))
            return cart, pole, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=len(trajectory), interval=50, blit=True, init_func=init
        )

        ani.save(filename, writer='pillow')
        plt.close(fig)
