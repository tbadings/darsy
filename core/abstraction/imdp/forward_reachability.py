"""Forward reachable slices (FRS) per (cell, action), snapped to coarse block borders."""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.utils import create_batches, wrap_diff

logger = logging.getLogger(__name__)


def sample_next_states(step_fn, noise_sample_fn, num_samples, state_min, state_max, action, rng):
    """Sample states uniformly"""
    rng_state, rng_noise = jax.random.split(rng)
    states = jax.random.uniform(rng_state, (num_samples, state_min.shape[0]),
                                minval=state_min, maxval=state_max)
    noise = noise_sample_fn(rng_noise, shape=(num_samples,))
    next_states = jax.vmap(step_fn, in_axes=(0, None, 0))(states, action, noise)
    return states, next_states


def _make_forward_reach(step_fn, noise_sample_fn, num_samples, alpha, partition, lipschitz):
    """JIT kernel: per (cell, action) the FRS slice start index and its width."""
    number_per_dim = jnp.asarray(partition.number_per_dim)
    boundary_lb = jnp.asarray(partition.boundary_lb)
    boundary_ub = jnp.asarray(partition.boundary_ub)

    pct_lo = 100 * alpha / 2        # lower percentile of the samples
    pct_hi = 100 * (1 - alpha / 2)  # upper percentile of the samples
    period = boundary_ub - boundary_lb
    wrap = partition.wrap
    norm_scale = number_per_dim / (boundary_ub - boundary_lb)  # state -> grid index

    def frs_one_action(state_min, state_max, action, rng):
        # 1. Sample transitions from the current cell under the given action
        _, next_states = sample_next_states(
            step_fn, noise_sample_fn, num_samples, state_min, state_max, action, rng
        )

        # 2. Compute an empirical box around the next_states
        mean_state = next_states[0] + jnp.mean(wrap_diff(next_states - next_states[0], period, wrap), axis=0)
        difference = wrap_diff(next_states - mean_state, period, wrap)
        if alpha != 0:
            frs_min = mean_state + jnp.percentile(difference, pct_lo, axis=0)
            frs_max = mean_state + jnp.percentile(difference, pct_hi, axis=0)
        else:
            frs_min = mean_state + jnp.min(difference, axis=0)
            frs_max = mean_state + jnp.max(difference, axis=0)

        # 3. Inflate by cell diameter * Lipschitz constant
        cell_diameter = state_max - state_min
        inflation = jnp.dot(cell_diameter, lipschitz.T)
        frs_min -= inflation
        frs_max += inflation

        # 4. Map to discrete indices
        def continuous_to_grid(coord):
            raw_idx = jnp.floor((coord - boundary_lb) * norm_scale)
            clipped_idx = jnp.clip(raw_idx, 0, number_per_dim - 1)
            return jnp.where(wrap, raw_idx, clipped_idx).astype(jnp.int32)

        idx_lb = continuous_to_grid(frs_min)
        idx_ub = continuous_to_grid(frs_max)

        # 5. Compute the span and put the lower bound into [0, number_per_dim - 1]
        span = jnp.minimum(idx_ub - idx_lb + 1, number_per_dim).astype(jnp.int32)
        idx_lb = jnp.mod(idx_lb, number_per_dim).astype(jnp.int16)

        return idx_lb, span

    # vmap: actions innermost, cells outermost
    per_action = jax.vmap(frs_one_action, in_axes=(None, None, 0, 0))
    return jax.jit(jax.vmap(per_action, in_axes=(0, 0, 0, 0)))


def _snap_window_to_blocks(idx_lb, span, factor, fine_shape, wrap):
    """Grow index windows out to coarse block borders."""
    start = idx_lb.astype(np.int64)  # window start per (cell, action)
    stop = start + span  # raw window end (exclusive); past the grid end only on wrapped dims
    start = start // factor * factor  # move start down to a block border

    # on wrapped dims, the part past the grid end is a second lap starting at index 0
    laps = np.where(wrap, (stop - 1) // fine_shape, 0)
    local_stop = stop - laps * fine_shape
    local_stop = (local_stop + factor - 1) // factor * factor  # move end up to a block border
    local_stop = np.minimum(local_stop, fine_shape)  # clip at the grid end
    stop = laps * fine_shape + local_stop

    stop = np.minimum(stop, start + fine_shape)  # at most one full lap
    return start, stop


class RectangularForward:
    """FRS slice (start index + width) for every (cell, action), snapped to coarse block borders."""

    def __init__(self, args, partition, model, coarse_factor):
        logger.info('=== Start forward reachability computations ===')
        t_start = time.time()

        self.coarse_factor = int(coarse_factor)  # block edge, in fine cells
        fine_shape = np.asarray(partition.number_per_dim)

        lipschitz = jnp.asarray(np.atleast_2d(model.lipschitz), dtype=jnp.float32)
        batch_frs = _make_forward_reach(model.step, model.noise.sample_jax,
                                        args.frs_num_samples, args.alpha, partition, lipschitz)

        num_cells = len(partition.regions['lower_bounds'])
        all_actions = partition.regions['actions']  # (1 or num_cells, num_actions, action_dim)
        num_actions = all_actions.shape[1]
        dim = partition.dimension

        self.frs_idx_lb = np.zeros((num_cells, num_actions, dim), dtype=np.int16)
        max_span = np.zeros(dim, dtype=int)

        rng = jax.random.PRNGKey(args.seed)
        starts, ends = create_batches(num_cells, args.frs_batch_size)

        for start, end in tqdm(zip(starts, ends), total=len(starts), desc='Forward reachability'):
            batch_size = end - start
            rng, batch_rng = jax.random.split(rng)  # new keys per batch
            rngs = jax.random.split(batch_rng, batch_size * num_actions)
            rngs = rngs.reshape(batch_size, num_actions, 2)  # one key per (cell, action)
            action_batch = jnp.broadcast_to(all_actions, (batch_size, *all_actions.shape[1:]))

            idx_lb, span = batch_frs(
                partition.regions['lower_bounds'][start:end],
                partition.regions['upper_bounds'][start:end],
                action_batch, rngs,
            )
            idx_lb, span = jax.device_get((idx_lb, span))

            # snap raw windows out to coarse block borders
            window_start, window_stop = _snap_window_to_blocks(idx_lb, span, self.coarse_factor,
                                                               fine_shape, partition.wrap)

            self.frs_idx_lb[start:end] = window_start
            np.maximum(max_span, np.max(window_stop - window_start, axis=(0, 1)), out=max_span)

        self.max_slice = tuple(max_span.tolist())  # widest slice per dimension
        self.max_slice_coarse = tuple(int(np.ceil(m / self.coarse_factor)) for m in self.max_slice)

        logger.info(f'- Max state-slice to compute probability intervals over: {self.max_slice}')
        logger.info(f'- Max coarse block-slice (factor {self.coarse_factor}): {self.max_slice_coarse}')
        logger.info(f'Reachability computations took {time.time() - t_start:.3f} sec.')
