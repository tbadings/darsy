import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.utils import create_batches

logger = logging.getLogger(__name__)


def _make_forward_reach(step_fn, noise_sample_fn, num_samples, alpha,
                        number_per_dim, boundary_lb, boundary_ub, lipschitz):
    """FRS windows (grid index + span) per (state, action), jitted and vmapped."""
    percentile_lo = 100 * alpha / 2  # lower tail percentile
    percentile_hi = 100 * (1 - alpha / 2)  # upper tail percentile
    grid_scale = number_per_dim / (boundary_ub - boundary_lb)  # state -> grid index

    def single_action_frs(state_min, state_max, action, rng):
        # two rng streams: one for states, one for noise
        rng_state, rng_noise = jax.random.split(rng)

        # sample successor states
        states = jax.random.uniform(rng_state, shape=(num_samples, state_min.shape[0]),
                                    minval=state_min, maxval=state_max)
        noise = noise_sample_fn(rng_noise, shape=(num_samples,))
        next_states = jax.vmap(step_fn, in_axes=(0, None, 0))(states, action, noise)

        # central (1 - alpha) box of the samples
        frs_min = jnp.percentile(next_states, percentile_lo, axis=0)
        frs_max = jnp.percentile(next_states, percentile_hi, axis=0)

        # inflate by cell diameter * Lipschitz matrix
        cell_diameter = state_max - state_min
        inflation = jnp.dot(cell_diameter, lipschitz.T)
        frs_min = frs_min - inflation
        frs_max = frs_max + inflation

        # snap to grid indices, kept inside the grid
        idx_low = jnp.clip(jnp.floor((frs_min - boundary_lb) * grid_scale), 0, number_per_dim - 1).astype(jnp.int16)
        idx_upp = jnp.clip(jnp.floor((frs_max - boundary_lb) * grid_scale), 0, number_per_dim - 1).astype(jnp.int32)

        span = idx_upp - idx_low + 1  # inclusive cell count per dim
        return idx_low, span

    # batched over states and then over actions
    return jax.jit(jax.vmap(
        jax.vmap(single_action_frs, in_axes=(None, None, 0, 0)),
        in_axes=(0, 0, 0, 0),
    ))


def _snap_window_to_blocks(idx_lb, span, factor, fine_shape):
    """Grow index windows out to coarse block borders."""
    start = idx_lb.astype(np.int64)  # window start per (state, action)
    stop = start + span  # raw window end (exclusive)
    start = start // factor * factor  # move start down to a block border
    stop = (stop + factor - 1) // factor * factor  # move end up to a block border
    stop = np.minimum(stop, fine_shape)  # clip at the grid end
    return start, stop


class RectangularForward:
    """FRS windows per (state, action), snapped to coarse block borders."""

    def __init__(self, args, partition, model, coarse_factor):
        logger.info('=== Start forward reachability computations ===')
        t_start = time.time()

        self.coarse_factor = int(coarse_factor)  # block edge, in fine cells
        max_span = self._compute_windows(args, partition, model)

        # widest FRS window over all (state, action) pairs
        factor = self.coarse_factor
        self.max_slice = tuple(max_span.tolist())
        self.max_slice_coarse = tuple(int(np.ceil(span / factor)) for span in self.max_slice)

        logger.info(f'- Max state-slice to compute probability intervals over: {self.max_slice}')
        logger.info(f'- Max coarse block-slice (factor {factor}): {self.max_slice_coarse}')
        logger.info(f'- Forward reachable sets computed (took {(time.time() - t_start):.3f} sec.)')

    def _compute_windows(self, args, partition, model):
        """Fill self.frs_idx_lb and return the per-dim max window span."""
        # grid sizes
        num_regions = len(partition.regions['lower_bounds'])
        num_actions = partition.regions['actions'].shape[1]
        dim = partition.dimension
        fine_shape = np.asarray(partition.number_per_dim)
        factor = self.coarse_factor

        # Lipschitz matrix for the sample inflation
        lipschitz_np = np.asarray(np.atleast_2d(model.lipschitz), dtype=np.float32)

        # jitted (states x actions) kernel
        batch_forward_reach = _make_forward_reach(
            model.step,
            model.noise.sample_jax,
            args.num_samples,
            args.alpha,
            jax.device_put(partition.number_per_dim),
            jax.device_put(partition.boundary_lb),
            jax.device_put(partition.boundary_ub),
            jax.device_put(lipschitz_np),
        )

        # output buffer: window start per (state, action, dim)
        self.frs_idx_lb = np.zeros((num_regions, num_actions, dim), dtype=np.int16)
        max_span = np.zeros(dim, dtype=int)  # running max window span

        # bounds and actions, shared over regions
        actions_slice = partition.regions['actions']
        lower_bounds = partition.regions['lower_bounds']
        upper_bounds = partition.regions['upper_bounds']

        # rng stream distinct from the PAC counts: the window must not
        # reuse those samples, or it gets fitted to them
        master_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
        starts, ends = create_batches(num_regions, args.frs_batch_size)
        pbar = tqdm(zip(starts, ends), total=len(starts))

        for batch_start, batch_end in pbar:
            batch_size = batch_end - batch_start

            # broadcast shared actions over the batch
            actions_batch = jnp.broadcast_to(actions_slice, (batch_size, *actions_slice.shape[1:]))

            # fresh rng key per (state, action) of the batch
            master_rng, batch_rng = jax.random.split(master_rng)
            rngs = jax.random.split(batch_rng, batch_size * num_actions).reshape(batch_size, num_actions, 2)

            batch_idx_lb, batch_span = batch_forward_reach(
                lower_bounds[batch_start:batch_end],
                upper_bounds[batch_start:batch_end],
                actions_batch,
                rngs,
            )

            # snap raw windows out to coarse block borders
            idx_lb_np, span_np = jax.device_get((batch_idx_lb, batch_span))
            start, stop = _snap_window_to_blocks(idx_lb_np, span_np, factor, fine_shape)

            self.frs_idx_lb[batch_start:batch_end] = start  # stored as int16
            # running max of the window spans
            np.maximum(max_span, np.max(stop - start, axis=(0, 1)), out=max_span)

        return max_span
