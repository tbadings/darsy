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
from core.abstraction.partition import _compute_linear_strides, edge_coords
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


def _grid_index(edges_padded, x, side, number_per_dim, boundary_lb, period, wrap):
    """Unclipped cell index per coordinate (-1 / n: outside the grid; counted in laps on wrapped dims).

    A point on an edge goes to the cell below it for side='left', above it for side='right'.
    """
    laps = jnp.where(wrap, jnp.floor((x - boundary_lb) / period), 0).astype(jnp.int32)
    method = 'compare_all' if edges_padded.shape[-1] <= 32 else 'scan'  # few edges: compare all at once
    search = jax.vmap(lambda e, v: jnp.searchsorted(e, v, side=side, method=method),
                      in_axes=(0, -1), out_axes=-1)
    local = search(edges_padded, x - laps * period) - 1
    return (laps * number_per_dim + local).astype(jnp.int32)


def _window_counts(lo, hi, a, b, origin, window_shape, grid_shape, wrap):
    """How many sample boxes fall in / overlap each window cell (row-major over window_shape).

    Each box overlaps the cells lo..hi and lies inside a single cell iff a == b (indices from _grid_index).
    Overlaps are counted with a difference array: +-1 at the corners of each index box, then a prefix
    sum along every dim.
    """
    window = tuple(int(w) for w in window_shape)
    grid = jnp.asarray(grid_shape, dtype=jnp.int32)
    lo, hi, a, b, origin = (jnp.asarray(v, dtype=jnp.int32) for v in (lo, hi, a, b, origin))
    num, dim = lo.shape

    # inside: the box's single cell, if that cell is in the grid and in the window
    pos = jnp.where(wrap, jnp.mod(a - origin, grid), a - origin)
    in_grid = wrap | ((a >= 0) & (a < grid))
    single = jnp.all((a == b) & in_grid & (pos >= 0) & (pos < jnp.asarray(window, dtype=jnp.int32)), axis=-1)
    ids = jnp.sum(jnp.where(single[:, None], pos, 0) * _compute_linear_strides(window).astype(np.int32), axis=-1)
    inside = jnp.zeros(int(np.prod(window)), jnp.int32).at[ids].add(single.astype(jnp.int32))

    # overlap: window range per dim, split in two where a wrapped range passes the grid end
    shape = tuple(w + 1 for w in window)  # one extra slot per dim for the closing corner
    strides = _compute_linear_strides(shape)
    flat, sign = 0, 1
    for k in range(dim):
        if wrap[k]:
            start = jnp.mod(lo[:, k] - origin[k], grid[k])
            end = start + hi[:, k] - lo[:, k]
            full = end - start + 1 >= grid[k]  # covers the whole circle
            seg_lo = jnp.stack([jnp.where(full, 0, start), jnp.zeros_like(start)], axis=-1)
            seg_hi = jnp.stack([jnp.where(full, window[k] - 1, jnp.minimum(end, grid[k] - 1)),
                                jnp.where(full, -1, end - grid[k])], axis=-1)
        else:
            seg_lo = (jnp.maximum(lo[:, k], 0) - origin[k])[:, None]
            seg_hi = (jnp.minimum(hi[:, k], grid[k] - 1) - origin[k])[:, None]
        seg_lo, seg_hi = jnp.maximum(seg_lo, 0), jnp.minimum(seg_hi, window[k] - 1)
        present = (seg_lo <= seg_hi)[..., None]
        corners = jnp.where(present, jnp.stack([seg_lo, seg_hi + 1], axis=-1), 0)  # (num, segments, 2)
        expand = (num,) + (1, 1) * k + corners.shape[1:] + (1, 1) * (dim - k - 1)
        flat = flat + (corners * int(strides[k])).reshape(expand)
        sign = sign * jnp.where(present, jnp.array([1, -1], dtype=jnp.int32), 0).reshape(expand)
    diff = jnp.zeros(int(np.prod(shape)), jnp.int32).at[flat.reshape(-1)].add(sign.reshape(-1))
    overlap = diff.reshape(shape)
    for k in range(dim):
        overlap = jnp.cumsum(overlap, axis=k, dtype=jnp.int32)
    overlap = overlap[tuple(slice(0, w) for w in window)].reshape(-1)
    return inside, overlap


def _make_counts_kernel(step_fn, noise_sample_fn, num_samples, partition, lipschitz, max_slice,
                        coarse_partition, max_slice_coarse, use_coarse):
    """JIT kernel: how many sample boxes fall in / overlap each candidate cell (and coarse block).

    All actions of a cell share its samples (states and noise); only the step differs per action.
    """
    number_per_dim = jnp.asarray(partition.number_per_dim)
    edges = partition.edges_padded
    boundary_lb = jnp.asarray(partition.boundary_lb)
    boundary_ub = jnp.asarray(partition.boundary_ub)
    wrap = partition.wrap
    period = boundary_ub - boundary_lb

    offsets = jnp.asarray(_window_offsets(max_slice))
    strides = jnp.asarray(_compute_linear_strides(partition.number_per_dim))
    # slice covers the whole circle so nothing can leave
    whole_circle = wrap & (np.asarray(max_slice) >= np.asarray(partition.number_per_dim))

    if use_coarse:
        factor = coarse_partition.factor
        coarse_offsets = jnp.asarray(_window_offsets(max_slice_coarse))
        coarse_strides = jnp.asarray(coarse_partition.coarse_strides)
        coarse_shape = jnp.asarray(coarse_partition.coarse_shape)

    def edge(k):
        return edge_coords(edges, k, number_per_dim, period, wrap)

    # window edges: of cells idx_lb + j for j = 0..max_slice per dim (inf past the window)
    steps = np.arange(max(max_slice) + 1)[:, None]
    past_window = steps > np.asarray(max_slice)
    wrapped = np.flatnonzero(wrap)

    def cell_index(x, idx_lb, window_edges):
        """Cell index of each coordinate for side='right' and side='left' (see _grid_index).

        Unwrapped dims are searched on the window edges only, so indices outside the window are clipped
        to one cell beyond it, which leaves all window counts unchanged. Wrapped dims are searched on the
        whole grid, since a box there can leave the window and re-enter it from the other side.
        """
        right = idx_lb + jnp.sum(window_edges <= x[:, None, :], axis=1) - 1
        left = idx_lb + jnp.sum(window_edges < x[:, None, :], axis=1) - 1
        if len(wrapped):
            grid = (number_per_dim[wrapped], boundary_lb[wrapped], period[wrapped], True)
            right = right.at[:, wrapped].set(_grid_index(edges[wrapped], x[:, wrapped], 'right', *grid))
            left = left.at[:, wrapped].set(_grid_index(edges[wrapped], x[:, wrapped], 'left', *grid))
        return right.astype(jnp.int32), left.astype(jnp.int32)

    def counts(state_min, state_max, action, idx_lb, rng):
        states, next_states = sample_next_states(step_fn, noise_sample_fn, num_samples,
                                                 state_min, state_max, action, rng)

        # every x in the cell maps into this box around its sampled next state
        inflation = jnp.dot(jnp.maximum(states - state_min, state_max - states), lipschitz.T)

        # cells each box overlaps (lo..hi, closed cells), and whether it lies in a single cell (a == b)
        box_lb, box_ub = next_states - inflation, next_states + inflation
        window_edges = jnp.where(past_window, jnp.inf, edge(idx_lb + steps))
        a, lo = cell_index(box_lb, idx_lb, window_edges)
        hi, b = cell_index(box_ub, idx_lb, window_edges)

        # cells in the FRS
        cell_coords = idx_lb + offsets
        valid = jnp.all((cell_coords < number_per_dim) | wrap, axis=-1)
        inside, overlap = _window_counts(lo, hi, a, b, idx_lb, max_slice, number_per_dim, wrap)
        succ_ids = _linear_ids(cell_coords, strides, number_per_dim, valid, wrap)

        # sink: inflated box out of the FRS
        slice_lb = edge(idx_lb)
        slice_ub = edge(idx_lb + jnp.asarray(max_slice))

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

        # block of each fine cell index; every lap of the fine grid holds coarse_shape blocks
        def block(k):
            return (k // number_per_dim) * coarse_shape + (k % number_per_dim) // factor

        block_inside, block_overlap = _window_counts(block(lo), block(hi), block(a), block(b), first_block,
                                                     max_slice_coarse, coarse_shape, wrap)
        block_ids = _linear_ids(block_coords, coarse_strides, coarse_shape, block_valid, wrap)

        return inside, overlap, valid, succ_ids, leaves, block_inside, block_overlap, block_valid, block_ids

    # vmap: actions innermost (one key per cell, so its samples are drawn once), cells outermost
    per_action = jax.vmap(counts, in_axes=(None, None, 0, 0, None))
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

    # Get shapes of cells and actions
    num_cells = len(partition.regions['lower_bounds'])
    all_actions = partition.regions['actions']  # (1 or num_cells, num_actions, action_dim)
    num_actions = all_actions.shape[1]

    # Fine cells + sink (+ coarse blocks) per (cell, action)
    intervals_per_pair = int(np.prod(actions.max_slice)) + 1
    if use_coarse:
        intervals_per_pair += int(np.prod(actions.max_slice_coarse))
    num_intervals = num_cells * num_actions * intervals_per_pair
    beta_per_interval = args.pac_beta / num_intervals
    logger.info(f'Union bound over {num_intervals} intervals: per interval beta = {beta_per_interval:.3e}')
    cp_lb, cp_ub = compute_cp_table(args.num_samples, beta_per_interval, args.floatprecision)

    # For multiplicative noise, model.lipschitz must bound the Jacobian
    # uniformly over the *entire* noise support. For the radially truncated
    # Gaussian benchmark, this is (1 + model.noise.radius) * abs(model.A).
    if getattr(model.noise, 'radius', None) is not None:
        required = (1.0 + float(model.noise.radius)) * np.abs(np.asarray(model.A))
        if np.any(np.asarray(model.lipschitz) + 1e-12 < required):
            raise ValueError('Lipschitz matrix does not cover truncated noise support')
    lipschitz = jnp.asarray(np.atleast_2d(model.lipschitz), dtype=args.floatprecision)
    counts_kernel = _make_counts_kernel(model.step, model.noise.sample_jax, args.num_samples,
                                        partition, lipschitz, actions.max_slice,
                                        coarse_partition, actions.max_slice_coarse, use_coarse)

    # Initialize output dictionaries
    float_dtype = args.floatprecision
    action_ids = np.arange(num_actions, dtype=np.int32)  # all actions enabled
    P_full, S_id, A_id, P_absorbing, Q_id, P_coarse = {}, {}, {}, {}, {}, {}

    # new samples independent of the ones that picked the FRS slice (see forward_reachability.py)
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    starts, ends = create_batches(num_cells, args.batch_size)

    # For every batch of cells
    for start, end in tqdm(zip(starts, ends), total=len(starts), desc='Probability intervals'):
        batch_size = end - start
        rng, batch_rng = jax.random.split(rng)  # new keys per batch
        rngs = jax.random.split(batch_rng, batch_size)  # one key per cell, shared by its actions
        action_batch = jnp.broadcast_to(all_actions, (batch_size, *all_actions.shape[1:]))

        # inside:           int32 (batch_size, num_actions, prod(actions.max_slice)), # boxes inside each window cell
        # overlap:          int32 (batch_size, num_actions, prod(actions.max_slice)), # boxes overlapping each window cell
        # valid:            bool  (batch_size, num_actions, prod(actions.max_slice)), # window cell lies in the grid
        # succ_ids:         int32 (batch_size, num_actions, prod(actions.max_slice)), # window cell ids (0 if invalid)
        # leaves:           int32 (batch_size, num_actions), boxes leaving the window (sink)
        # block_inside:     int32 (batch_size, num_actions, prod(actions.max_slice_coarse)), or None without use_coarse
        # block_overlap:    int32 (batch_size, num_actions, prod(actions.max_slice_coarse)), or None without use_coarse
        # block_valid:      bool  (batch_size, num_actions, prod(actions.max_slice_coarse)), or None without use_coarse
        # block_ids:        int32 (batch_size, num_actions, prod(actions.max_slice_coarse)), or None without use_coarse
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
