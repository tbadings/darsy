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


def _count_blocks(start, width, factor, size, wrap):
    # Number of blocks completely inside
    cells = start + np.arange(width)
    if wrap:
        cells = cells % size
    else:
        cells = cells[cells < size]
    blocks, cells_in_window = np.unique(cells // factor, return_counts=True)
    block_sizes = np.minimum((blocks + 1) * factor, size) - blocks * factor  # outer blocks are smaller
    return int(np.sum(cells_in_window == block_sizes))


def _max_blocks_per_dim(frs_idx_lb, max_slice, factor, number_per_dim, wrap):
    max_blocks = []
    for d in range(len(max_slice)):
        most = 0
        for start in np.unique(frs_idx_lb[..., d]):
            blocks = _count_blocks(int(start), max_slice[d], factor, int(number_per_dim[d]), wrap[d])
            most = max(most, blocks)
        max_blocks.append(most)
    return tuple(max_blocks)

class RectangularForward:
    """FRS slice (start index + width) for every (cell, action)"""

    def __init__(self, args, partition, model, coarse_factor):
        logger.info('=== Start forward reachability computations ===')
        t_start = time.time()

        self.coarse_factor = int(coarse_factor)

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

            self.frs_idx_lb[start:end] = idx_lb
            np.maximum(max_span, np.max(span, axis=(0, 1)), out=max_span)

        self.max_slice = tuple(max_span.tolist())
        self.max_slice_coarse = _max_blocks_per_dim(self.frs_idx_lb, self.max_slice, self.coarse_factor,
                                                    partition.number_per_dim, partition.wrap)

        logger.info(f'- Max state-slice to compute probability intervals over: {self.max_slice}')
        logger.info(f'- Max coarse block-slice (factor {self.coarse_factor}): {self.max_slice_coarse}')
        logger.info(f'Reachability computations took {time.time() - t_start:.3f} sec.')
