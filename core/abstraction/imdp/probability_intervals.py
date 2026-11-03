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


_MAX_BUCKETS = 1 << 20  # larger bucket tables (very thin cells) fall back to a binary search


def _make_edge_search(edges):
    """Search on fixed, strictly increasing edges: x -> (#(edges <= x) - 1, #(edges < x) - 1).

    A point on an edge goes to the cell above it in the first index, below it in the second. Uses a
    table of buckets no wider than the narrowest cell, holding per bucket the last edge half a bucket
    below it: up to rounding of the bucket index, a point lies less than 1.5 buckets above that edge, so
    at most 2 further edges can be at or below it.
    """
    edges = np.asarray(edges)
    if np.any(np.diff(edges) <= 0):
        raise ValueError('Grid edges must be strictly increasing.')
    width = np.min(np.diff(edges))
    num_buckets = int(np.ceil((edges[-1] - edges[0]) / width)) + 2  # with one bucket past either end

    if num_buckets > _MAX_BUCKETS:
        sorted_edges = jnp.asarray(edges)

        def binary_search(x):
            e = jax.lax.optimization_barrier(sorted_edges)  # else XLA constant-folds the first step over all of x
            right = jnp.searchsorted(e, x, side='right', method='scan_unrolled').astype(jnp.int32) - 1
            left = right - ((right >= 0) & (e[jnp.maximum(right, 0)] == x))  # x is the edge just below it
            return right, left
        return binary_search

    bucket_lb = edges[0] + (np.arange(num_buckets) - 0.5) * width
    starts = jnp.asarray(np.searchsorted(edges, bucket_lb, side='right') - 1, dtype=jnp.int32)
    padded = jnp.asarray(np.append(edges, np.full(3, np.inf, dtype=edges.dtype)))  # 3 candidates past the end

    def search(x):
        bucket = jnp.clip(jnp.floor((x - edges[0]) / width), 0, num_buckets - 1).astype(jnp.int32)
        start = starts[bucket]
        candidates = [padded[start + t] for t in (1, 2, 3)]
        right = start + sum((c <= x).astype(jnp.int32) for c in candidates)
        left = start + sum((c < x).astype(jnp.int32) for c in candidates)
        return right, left
    return search


def _grid_index(search, x, number_per_dim, boundary_lb, period):
    """Cell index of each coordinate on a wrapped dim, counted in laps, for side='right' and side='left'."""
    laps = jnp.floor((x - boundary_lb) / period).astype(jnp.int32)
    right, left = search(x - laps * period)  # search from _make_edge_search
    shift = laps * number_per_dim
    return shift + right, shift + left


def _contract(factors):
    """Per window cell, the number of samples n with all factors[k][n, j_k] set (row-major over j).

    A sum over samples of outer products, computed as a matrix product: the dims are split into two
    groups of about equal window size, and each group is combined into one (samples, cells) matrix.
    """
    num = factors[0].shape[0]
    if len(factors) == 1:
        factors = [jnp.ones((num, 1), dtype=bool), *factors]  # matrix product here too: faster than a sum
    sizes = [f.shape[1] for f in factors]
    split = min(range(1, len(factors)), key=lambda m: abs(np.log(np.prod(sizes[:m]) / np.prod(sizes[m:]))))

    def combine(group):
        out = group[0]
        for f in group[1:]:
            out = (out[:, :, None] & f[:, None, :]).reshape(num, -1)
        return out

    # 0/1 entries and integer sums below 2^24 are exact in float32
    dtype = jnp.float32 if num < 2 ** 24 else jnp.float64
    left, right = (combine(g).astype(dtype) for g in (factors[:split], factors[split:]))
    counts = jnp.dot(left.T, right, precision=jax.lax.Precision.HIGHEST)
    return counts.reshape(-1).astype(jnp.int32)


def _window_counts(lo, hi, a, b, origin, window_shape, grid_shape, wrap):
    """How many sample boxes fall in / overlap each window cell (row-major over window_shape).

    Each box overlaps the cells lo..hi and lies inside a single cell iff a == b (indices from _grid_index).
    Boxes and cells are boxes, so a box overlaps a cell iff it does so in every dim; the counts are sums
    over samples of the per-dim indicators (see _contract).
    """
    window = tuple(int(w) for w in window_shape)
    grid = jnp.asarray(grid_shape, dtype=jnp.int32)
    lo, hi, a, b, origin = (jnp.asarray(v, dtype=jnp.int32) for v in (lo, hi, a, b, origin))
    num, dim = lo.shape

    inside, overlap = [], []
    for k in range(dim):
        cells = origin[k] + jnp.arange(window[k], dtype=jnp.int32)  # grid cell of each window position
        lo_k, hi_k, a_k = lo[:, k, None], hi[:, k, None], a[:, k, None]
        single = (a == b)[:, k, None]
        if wrap[k]:
            # a range covers a cell iff it starts at most its length before it, or covers the whole circle
            span = hi_k - lo_k
            overlap.append((span + 1 >= grid[k]) | (jnp.mod(cells - lo_k, grid[k]) <= span))
            inside.append(single & (jnp.mod(a_k - cells, grid[k]) == 0))
        else:
            in_grid = cells < grid[k]
            overlap.append((lo_k <= cells) & (cells <= hi_k) & in_grid)
            inside.append(single & (a_k == cells) & in_grid)
    return _contract(inside), _contract(overlap)


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
    # whole-grid search per wrapped dim, on its edges without padding
    grid_sizes = [int(n) for n in np.asarray(partition.number_per_dim)]
    searches = {k: _make_edge_search(np.asarray(edges[k])[:grid_sizes[k] + 1]) for k in np.flatnonzero(wrap)}

    def cell_index(x, idx_lb, window_edges):
        """Cell index of each coordinate for side='right' and side='left'.

        Unwrapped dims are searched on the window edges only, so indices outside the window are clipped
        to one cell beyond it, which leaves all window counts unchanged. Wrapped dims are searched on the
        whole grid (see _grid_index), since a box there can leave the window and re-enter it from the
        other side.
        """
        right, left = [], []
        for k in range(len(wrap)):
            if wrap[k]:
                r, l = _grid_index(searches[k], x[:, k], grid_sizes[k], boundary_lb[k], period[k])
            else:
                r = idx_lb[k] + jnp.sum(window_edges[:, k] <= x[:, k, None], axis=1, dtype=jnp.int32) - 1
                l = idx_lb[k] + jnp.sum(window_edges[:, k] < x[:, k, None], axis=1, dtype=jnp.int32) - 1
            right.append(r)
            left.append(l)
        return jnp.stack(right, axis=-1).astype(jnp.int32), jnp.stack(left, axis=-1).astype(jnp.int32)

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
        # computed once: else XLA fuses the search into each window cell's indicator and repeats it there
        a, lo, hi, b = jax.lax.optimization_barrier((a, lo, hi, b))

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
    all_actions = partition.regions['actions']  # (1, num_actions, action_dim)
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
