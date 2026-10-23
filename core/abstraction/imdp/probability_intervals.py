"""PAC transition probability intervals for (2-layer) IMDPs.

Samples are inflated via the Lipschitz matrix, counted against successor cells (and coarse blocks),
and turned into Clopper-Pearson PAC bounds.
"""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from tqdm import tqdm

from core.abstraction.imdp.forward_reachability import sample_next_states
from core.abstraction.partition import _compute_linear_strides
from core.utils import create_batches, wrap_diff

logger = logging.getLogger(__name__)


def compute_cp_table(num_samples: int, beta: float, float_dtype) -> tuple[np.ndarray, np.ndarray]:
    """Clopper-Pearson lookup table: binomial count k -> confidence bounds at level 1 - beta."""
    k = np.arange(num_samples + 1)
    cp_lb = np.zeros(num_samples + 1, dtype=float_dtype)  # count 0 -> lower bound 0
    cp_ub = np.ones(num_samples + 1, dtype=float_dtype)   # count N -> upper bound 1

    cp_lb[1:] = scipy.stats.beta.ppf(beta / 2.0, k[1:], num_samples - k[1:] + 1)          # lower quantile
    cp_ub[:-1] = scipy.stats.beta.ppf(1.0 - beta / 2.0, k[:-1] + 1, num_samples - k[:-1])  # upper quantile

    return cp_lb, cp_ub


def _window_offsets(window_shape):
    """Grid index offsets of all cells in a window, one row per cell."""
    dim = len(window_shape)
    return np.indices(window_shape).reshape(dim, -1).T  # row-major order


def _linear_ids(coords, strides, grid_shape, valid, wrap):
    """Row-major cell ids; invalid cells get id 0."""
    coords = jnp.where(wrap, jnp.mod(coords, grid_shape), coords)
    clipped = jnp.clip(coords, 0, grid_shape - 1)  # keep ids in range
    ids = jnp.sum(clipped * strides, axis=-1).astype(jnp.int32)  # row-major id
    return jnp.where(valid, ids, 0)  # invalid cells -> 0


def _count_boxes(next_states, inflation, centers, half_widths, period, wrap):
    """How many sample boxes fall in / overlap each box given by its centers and half-widths."""
    # Distance from each sample center to each cell center
    dist = jnp.abs(wrap_diff(next_states[:, None, :] - centers, period, wrap))
    r = inflation[:, None, :]
    inside = jnp.sum(jnp.all(dist + r <= half_widths, axis=-1), axis=0)    # sample box fully inside cell
    overlap = jnp.sum(jnp.all(dist - r <= half_widths, axis=-1), axis=0)  # sample box overlaps cell
    return inside.astype(jnp.int32), overlap.astype(jnp.int32)


def _make_counts_kernel(step_fn, noise_sample_fn, num_samples, partition, lipschitz, max_slice,
                        coarse_partition, max_slice_coarse, use_coarse):
    """JIT kernel: how many sample boxes fall in / overlap each candidate cell (and coarse block)."""
    number_per_dim = jnp.asarray(partition.number_per_dim)
    cell_width = jnp.asarray(partition.cell_width)
    boundary_lb = jnp.asarray(partition.boundary_lb)
    boundary_ub = jnp.asarray(partition.boundary_ub)
    wrap = partition.wrap
    period = boundary_ub - boundary_lb

    offsets = jnp.asarray(_window_offsets(max_slice))
    strides = jnp.asarray(_compute_linear_strides(partition.number_per_dim))
    slice_width = jnp.asarray(max_slice) * cell_width
    # slice covers the whole circle so nothing can leave
    whole_circle = wrap & (np.asarray(max_slice) >= np.asarray(partition.number_per_dim))

    if use_coarse:
        factor = coarse_partition.factor
        coarse_offsets = jnp.asarray(_window_offsets(max_slice_coarse))
        coarse_strides = jnp.asarray(coarse_partition.coarse_strides)
        coarse_shape = jnp.asarray(coarse_partition.coarse_shape)

    def counts(state_min, state_max, action, idx_lb, rng):
        states, next_states = sample_next_states(step_fn, noise_sample_fn, num_samples,
                                                 state_min, state_max, action, rng)

        # every x in the cell maps into this box around its sampled next state
        inflation = jnp.dot(jnp.maximum(states - state_min, state_max - states), lipschitz.T)

        # cells in the FRS
        cell_coords = idx_lb + offsets
        cell_centers = (cell_coords + 0.5) * cell_width + boundary_lb
        valid = jnp.all((cell_coords < number_per_dim) | wrap, axis=-1)
        inside, overlap = _count_boxes(next_states, inflation, cell_centers, cell_width / 2, period, wrap)
        succ_ids = _linear_ids(cell_coords, strides, number_per_dim, valid, wrap)

        # sink: inflated box out of the FRS
        slice_lb = idx_lb * cell_width + boundary_lb
        slice_ub = slice_lb + slice_width
        slice_lb = jnp.where(wrap, slice_lb, jnp.maximum(slice_lb, boundary_lb))
        slice_ub = jnp.where(wrap, slice_ub, jnp.minimum(slice_ub, boundary_ub))

        slice_center = (slice_lb + slice_ub) / 2
        slice_half_width = (slice_ub - slice_lb) / 2

        slice_dist = jnp.abs(wrap_diff(next_states - slice_center, period, wrap))
        leaves_dim = (slice_dist + inflation > slice_half_width) & ~whole_circle
        leaves = jnp.sum(jnp.any(leaves_dim, axis=-1)).astype(jnp.int32)

        # no second layer
        if not use_coarse:
            return inside, overlap, valid, succ_ids, leaves, None, None, None, None

        # coarse blocks
        first_block = jnp.where(whole_circle, 0, (idx_lb + factor - 1) // factor)
        block_coords = first_block + coarse_offsets
        block_valid = jnp.all((block_coords < coarse_shape) | wrap, axis=-1)

        # on wrapped dims, blocks past the grid end are a second lap starting at block 0
        laps = jnp.where(wrap, block_coords // coarse_shape, 0)
        block_start = (block_coords - laps * coarse_shape) * factor  # first fine cell of the block
        block_end = jnp.minimum(block_start + factor, number_per_dim)  # shorter at the edge

        # only blocks that are completely inside the FRS
        slice_end = idx_lb + jnp.asarray(max_slice)
        slice_end = jnp.where(wrap, slice_end, jnp.minimum(slice_end, number_per_dim))
        block_fits = (laps * number_per_dim + block_end <= slice_end) | whole_circle
        block_valid = block_valid & jnp.all(block_fits, axis=-1)

        # block boxes
        block_lb = boundary_lb + laps * period + block_start * cell_width
        block_ub = boundary_lb + laps * period + block_end * cell_width

        block_centers = (block_lb + block_ub) / 2
        block_inside, block_overlap = _count_boxes(next_states, inflation, block_centers,
                                                   (block_ub - block_lb) / 2, period, wrap)
        block_ids = _linear_ids(block_coords, coarse_strides, coarse_shape, block_valid, wrap)

        return inside, overlap, valid, succ_ids, leaves, block_inside, block_overlap, block_valid, block_ids

    # vmap: actions innermost, cells outermost
    per_action = jax.vmap(counts, in_axes=(None, None, 0, 0, 0))
    return jax.jit(jax.vmap(per_action, in_axes=(0, 0, 0, 0, 0)))


def compute_probability_intervals(args, model, partition, actions, coarse_partition,
                                  debug_state: int | None = None, use_coarse: bool = False):
    """Per cell id: fine intervals, successor ids, action ids, sink interval, coarse ids, coarse intervals.

    Fine entries follow the cells actions.frs_idx_lb + offset (row-major over actions.max_slice).
    """
    logger.info('=== Start transition probability intervals computation ===')
    t_start = time.time()

    if actions.coarse_factor != coarse_partition.factor:  # layers must match
        raise ValueError(
            f'Coarse factor mismatch: actions built with {actions.coarse_factor}, '
            f'coarse partition built with {coarse_partition.factor}.'
        )

    num_cells = len(partition.regions['lower_bounds'])
    all_actions = partition.regions['actions']  # (1 or num_cells, num_actions, action_dim)
    num_actions = all_actions.shape[1]

    # fine cells + sink (+ coarse blocks) per (cell, action)
    intervals_per_pair = int(np.prod(actions.max_slice)) + 1
    if use_coarse:
        intervals_per_pair += int(np.prod(actions.max_slice_coarse))
    num_intervals = num_cells * num_actions * intervals_per_pair
    beta_per_interval = args.pac_beta / num_intervals
    logger.info(f'Union bound over {num_intervals} intervals: per interval beta = {beta_per_interval:.3e}')
    cp_lb, cp_ub = compute_cp_table(args.num_samples, beta_per_interval, args.floatprecision)

    lipschitz = jnp.asarray(np.atleast_2d(model.lipschitz), dtype=args.floatprecision)
    counts_kernel = _make_counts_kernel(model.step, model.noise.sample_jax, args.num_samples,
                                        partition, lipschitz, actions.max_slice,
                                        coarse_partition, actions.max_slice_coarse, use_coarse)

    float_dtype = args.floatprecision
    action_ids = np.arange(num_actions, dtype=np.int32)  # all actions enabled
    P_full, S_id, A_id, P_absorbing, Q_id, P_coarse = {}, {}, {}, {}, {}, {}

    # new samples independent of the ones that picked the FRS slice (see forward_reachability.py)
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    starts, ends = create_batches(num_cells, args.batch_size)

    for start, end in tqdm(zip(starts, ends), total=len(starts), desc='Probability intervals'):
        batch_size = end - start
        rng, batch_rng = jax.random.split(rng)  # new keys per batch
        rngs = jax.random.split(batch_rng, batch_size * num_actions)
        rngs = rngs.reshape(batch_size, num_actions, 2)  # one key per (cell, action)
        action_batch = jnp.broadcast_to(all_actions, (batch_size, *all_actions.shape[1:]))

        (inside, overlap, valid, succ_ids, leaves,
         block_inside, block_overlap, block_valid, block_ids) = jax.device_get(counts_kernel(
            partition.regions['lower_bounds'][start:end], partition.regions['upper_bounds'][start:end],
            action_batch, actions.frs_idx_lb[start:end], rngs,
        ))

        # lower: boxes fully inside; upper: boxes overlap the cell
        prob = np.stack([np.where(valid, cp_lb[inside], 0.0), np.where(valid, cp_ub[overlap], 0.0)], axis=-1)
        # sink lower bound always zero
        prob_sink = np.stack([np.zeros_like(cp_ub[leaves]), cp_ub[leaves]], axis=-1)
        if use_coarse:
            prob_block = np.stack([np.where(block_valid, cp_lb[block_inside], 0.0),
                                   np.where(block_valid, cp_ub[block_overlap], 0.0)], axis=-1)

        for i in range(batch_size):
            s = int(start + i)
            S_id[s] = succ_ids[i].astype(np.int32)
            A_id[s] = action_ids
            P_full[s] = prob[i].astype(float_dtype)
            P_absorbing[s] = prob_sink[i].astype(float_dtype)
            if use_coarse:
                Q_id[s] = block_ids[i].astype(np.int32)
                P_coarse[s] = prob_block[i].astype(float_dtype)

    if debug_state is not None and debug_state in P_full:
        logger.info(f'- Debug state {debug_state}:')
        logger.info(f'  Enabled actions: {A_id[debug_state]}')
        logger.info(f'  Successors shape: {S_id[debug_state].shape}')
        logger.info(f'  P_absorbing: {P_absorbing[debug_state]}')
        if use_coarse:
            logger.info(f'  Coarse successors shape: {Q_id[debug_state].shape}')

    logger.info(f'Probability intervals computed (took {time.time() - t_start:.3f} sec.)')
    return P_full, S_id, A_id, P_absorbing, Q_id, P_coarse
