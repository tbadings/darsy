from functools import partial, reduce
import logging

from core.utils import create_batches
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Note: The following implementation supports Gaussian and Triangular noise distributions.

def wrap_pad(V, size):
    '''
    Repeat the first `size` entries of V at the end, so that dynslice can never read past the array.

    :param V: Vector to pad.
    :param size: Slice length dynslice will be called with (must be <= len(V)).
    :return: V with its first `size` entries appended.
    '''
    return jnp.concatenate([V, V[:size]])

def dynslice(V, idx_low, size):
    '''
    Given a vector of indices, keep only those starting at position idx_low and of length size,
    wrapping around the end of the (unpadded) vector.

    V must be wrap-padded by `wrap_pad` with this same size, which makes the slice equal to
    jnp.roll(V, -idx_low)[:size] on the unpadded vector while reading only `size` elements instead
    of the whole vector.

    :param V: Vector of indices, wrap-padded by `wrap_pad`.
    :param idx_low: Index to start slice at.
    :param size: Number of elements to keep in slice.
    :return: Slice of V.
    '''
    return jax.lax.dynamic_slice(V, (idx_low,), (size,))

def interval_distribution(i_lb, mean_lb, mean_ub, *,
                          n, max_slice, wrap, wrap_array, decimals,
                          number_per_dim, per_dim_lb, per_dim_ub,
                          state_space_lb, state_space_ub,
                          region_linear_idx, region_linear_state,
                          region_linear_strides, missing_state,
                          unsafe_states, noise, dense):
    '''
    For a given state-action pair, compute the probability intervals over all successor states.

    High-level steps:
      1. Compute probability bounds: slice partition boundaries around the reachable region,
         query the noise distribution per dimension, and form joint bounds via outer product.
      2. Resolve state IDs: map the flat n-D grid indices to 1-D partition state IDs; on a dense
         partition the linear cell key is already the state ID, otherwise it is looked up by a
         sparse binary search and cells absent from the partition get a sentinel ID.
      3. Build the interval distribution: accumulate probability mass for absent (missing)
         cells as absorbing failure, filter to nonzero successor cells, enforce a minimum
         lower bound for graph-preservation, and compute the total absorbing probability.
      4. Decide whether to keep this state-action pair based on a safety threshold.
      5. Pack outputs: stably move nonzero entries to the front (O(N)) and zero the tail.
    '''

    # --- Step 1: Compute probability bounds ---
    # Extract slices of per-dimension partition boundaries centred on the reachable region.
    x_lb = [dynslice(per_dim_lb[i], i_lb[i], max_slice[i]) for i in range(n)]
    x_ub = [dynslice(per_dim_ub[i], i_lb[i], max_slice[i]) for i in range(n)]
    # Grid indices of those cells in each dimension.
    prob_idx = [jnp.arange(max_slice[i]) + i_lb[i] for i in range(n)]

    # Spatial bounding box of the slice (first/last cell boundary per dimension).
    slice_lb = jnp.stack([xl[0]  for xl in x_lb])
    slice_ub = jnp.stack([xu[-1] for xu in x_ub])

    # Switch explicitly on noise type to avoid silently accepting unsupported distributions.
    noise_type = noise['type']
    if noise_type in ('Gaussian', 'Triangular'):
        _, prob_low, prob_high = noise.prob_minmax_per_dim(n, wrap, x_lb, x_ub, mean_lb, mean_ub, state_space_ub - state_space_lb)
        prob_in_slice = noise.prob_minmax(slice_lb, slice_ub, mean_lb, mean_ub, wrap_array)
    else:
        raise ValueError(f'Unsupported noise type: {noise_type}. Expected Gaussian or Triangular.')

    # Joint probability bounds via outer product across dimensions.
    # After this, prob_low_prod and prob_high_prod are flat arrays of length prod(max_slice) containing the lower and upper bounds, 
    # respectively, for each cell in the n-D slice defined by max_slice. The ordering matches that of the flattened n-D grid indices 
    # in prob_idx after meshgrid with indexing='ij'.
    prob_low_prod  = jnp.round(reduce(lambda a, b: a[..., None] * b, prob_low).reshape(-1),  decimals)
    prob_high_prod = jnp.round(reduce(lambda a, b: a[..., None] * b, prob_high).reshape(-1), decimals)

    # Per-dimension contribution of each cell to its linear key, and whether it falls outside the
    # grid. Clip before linearizing; out-of-range cells are filtered via in_partition_bounds below.
    key_per_dim = [jnp.clip(idx, 0, number_per_dim[i] - 1).astype(region_linear_strides.dtype) * region_linear_strides[i]
                   for i, idx in enumerate(prob_idx)]
    oob_per_dim = [(idx < 0) | (idx >= number_per_dim[i]) for i, idx in enumerate(prob_idx)]

    # Combine across dimensions the same way as the probability bounds above (outer sum / outer OR
    # rather than outer product), so the ordering matches and no [prod(max_slice), n] index array is
    # ever built. After this, for each cell in the n-D slice defined by max_slice we have:
    # - prob_low_prod, prob_high_prod: the lower and upper bounds of the probability (shape: [prod(max_slice)])
    # - linear_prob_idx: its linear key within the overall partition grid (shape: [prod(max_slice)])
    # - in_partition_bounds: whether it lies inside that grid at all (shape: [prod(max_slice)])
    linear_prob_idx     =  reduce(lambda a, b: a[..., None] + b, key_per_dim).reshape(-1)
    in_partition_bounds = ~reduce(lambda a, b: a[..., None] | b, oob_per_dim).reshape(-1)

    # --- Step 2: Resolve state IDs ---
    # prob_id is an array of shape [prod(max_slice)] and gives the partition state ID for each cell in
    # the n-D slice, or missing_state if that cell is absent from the (sparse) partition.
    if dense:
        # Dense rectangular grid: cells are stored in row-major key order (enforced by the
        # 'Dense partition cells are not in row-major order' check in the partition), so the sorted
        # keys are exactly 0..S-1 and region_linear_state is the identity -- the key *is* the state
        # ID, with neither a binary search nor a gather. Keys are clipped to the grid above, so
        # nothing can be missing; cells outside the grid are dropped by in_partition_bounds below.
        prob_id = linear_prob_idx
    else:
        # Binary search in the sorted sparse key array, then verify exact match.
        pos = jnp.searchsorted(region_linear_idx, linear_prob_idx, side='left')
        pos_clip = jnp.minimum(pos, region_linear_idx.shape[0] - 1)
        valid = (pos < region_linear_idx.shape[0]) & (region_linear_idx[pos_clip] == linear_prob_idx)
        # Cells absent from the sparse partition map to a sentinel ID and are handled in step 3.
        prob_id = jnp.where(valid, region_linear_state[pos_clip], missing_state)

    p_lowest = 10 ** -decimals

    # --- Step 3: Build the interval distribution ---
    # Cells within grid bounds but absent from the sparse partition are treated as absorbing
    # failure (rather than silently discarding their probability mass).
    missing_cell_mask = (prob_high_prod > p_lowest) & in_partition_bounds & (prob_id == missing_state)
    missing_absorbing = jnp.array([
        jnp.sum(prob_low_prod  * missing_cell_mask),
        jnp.sum(prob_high_prod * missing_cell_mask),
    ])

    # Keep only cells with nonzero probability; enforce a small minimum lower bound on
    # active cells to ensure the IMDP is graph-preserving.
    prob_nonzero = (prob_high_prod > p_lowest) & in_partition_bounds & (prob_id != missing_state)
    prob_low_prod = jnp.maximum(p_lowest * prob_nonzero, prob_low_prod)
    prob_high_prod = jnp.maximum(p_lowest * prob_nonzero, prob_high_prod)
    # Pack lower and upper bounds: prob[s] = [lb, ub].
    prob = jnp.stack([prob_low_prod, prob_high_prod]).T

    # Absorbing probability = P(NOT in slice), covering both P(outside state space) and
    # P(in state space but outside the slice window). Invert prob_in_slice ([lb_in, ub_in])
    # and add the missing-cell contribution.
    prob_absorbing = jnp.round(1 - prob_in_slice[::-1], decimals)
    prob_absorbing = jnp.maximum(p_lowest * (prob_absorbing[1] > 0), prob_absorbing)
    prob_absorbing = prob_absorbing + missing_absorbing

    # --- Step 4: Decide whether to keep this state-action pair ---
    # Discard if both the lower bound on reaching safe successors is below 1-threshold AND
    # the upper bound on reaching unsafe/absorbing states exceeds threshold.
    # Only count present partition cells (prob_nonzero) — missing cells (absent from the sparse
    # partition but within grid bounds) have already had their mass moved to prob_absorbing via
    # missing_absorbing, so counting them here as safe successors would double-count their mass.
    threshold = 0.1
    unsafe_states_slice = unsafe_states[prob_id]
    keep = ~(((jnp.sum(prob[:, 0] * prob_nonzero * ~unsafe_states_slice)) < 1 - threshold) &
             ((prob_absorbing[1] + jnp.sum(prob[:, 1] * prob_nonzero * unsafe_states_slice)) > threshold))

    number_nonzero = jnp.sum(prob_nonzero)

    # --- Step 5: Pack outputs ---
    # Stably move nonzero entries to the front in O(N) (not O(N log N)).
    idx        = jnp.arange(prob_nonzero.shape[0], dtype=jnp.int32)
    true_pos   = jnp.cumsum(prob_nonzero)  - 1
    false_pos  = jnp.cumsum(~prob_nonzero) - 1
    target_pos = jnp.where(prob_nonzero, true_pos, number_nonzero + false_pos)
    sorted_idx = jnp.zeros_like(idx).at[target_pos].set(idx)
    prob         = prob[sorted_idx]
    prob_id      = prob_id[sorted_idx]
    prob_nonzero = prob_nonzero[sorted_idx]

    # Zero out padded tail entries to prevent invalid successor IDs from appearing with
    # nonzero probability after batch-level truncation in the caller.
    prob    = jnp.where(prob_nonzero[:, None], prob,    jnp.zeros_like(prob))
    prob_id = jnp.where(prob_nonzero,          prob_id, jnp.zeros_like(prob_id))

    return prob, prob_id, prob_nonzero, prob_absorbing, keep, number_nonzero, missing_absorbing

def compute_probability_intervals(args, model, partition, actions, vectorized=True, debug_state=None):
    '''
    Compute probability intervals for all states and actions of the IMDP.

    :param args: Argument object.
    :param model: Model object.
    :param partition: Partition object.
    :param frs: Forward reachable sets.
    :param max_slice: Array where each element is the maximum number of partition elements to consider in each dimension.
    :param debug_state: If provided, log detailed per-action diagnostics for this state index when it has no enabled actions.
    :return:
        - prob: Probability intervals per state-action pair
        - prob_id: Successor states associated with these probability intervals per state-action pair
        - prob_absorbing: Probability interval of reaching the absorbing state per state-action pair
    '''

    logger.info('Compute probability intervals for all state-action pairs...')
    logger.info('- Size of the successor state slice to consider per dimension: %s', actions.max_slice)

    frs_lb = actions.frs_lb
    frs_ub = actions.frs_ub
    frs_idx_lb = actions.frs_idx_lb
    model_wrap_tuple = tuple(np.array(model.wrap))

    interval_distribution_fixed = partial(
        interval_distribution,
        n=model.n,
        max_slice=actions.max_slice,
        wrap=model_wrap_tuple,
        wrap_array=model.wrap,
        decimals=args.decimals,
        number_per_dim=partition.number_per_dim,
        # Wrap-padded so dynslice can take a fixed-length slice without reading past the array.
        per_dim_lb=[wrap_pad(partition.regions_per_dim['lower_bounds'][i], actions.max_slice[i]) for i in range(model.n)],
        per_dim_ub=[wrap_pad(partition.regions_per_dim['upper_bounds'][i], actions.max_slice[i]) for i in range(model.n)],
        state_space_lb=jax.device_put(partition.boundary_lb),
        state_space_ub=jax.device_put(partition.boundary_ub),
        region_linear_idx=jax.device_put(partition.region_linear_idx),
        region_linear_state=jax.device_put(partition.region_linear_state),
        region_linear_strides=jax.device_put(partition.region_linear_strides),
        missing_state=partition.missing_state,
        unsafe_states=jax.device_put(jnp.concatenate((partition.critical['bools'], jnp.array([False])))),
        noise=model.noise,
        dense=bool(partition.rectangular),
    )

    # vmap over the 3 per-action args only; all constants are captured in the closure
    vmap_interval_distribution = jax.jit(
        jax.vmap(interval_distribution_fixed, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0, 0, 0, 0)))

    action_labels = {}
    interval_matrix = {}
    successor_id = {}
    interval_absorbing = {}

    actions_id = np.asarray(actions.id)

    nrA = partition.regions['actions'].shape[1]
    if vectorized:

        starts, ends = create_batches(len(partition.regions['idxs']), batch_size=args.batch_size)

        for i, j in tqdm(zip(starts, ends), total=len(starts)):
            
            # Reshape frs_idx_lb from S x A x n to (S x A) rows and n columns
            frs_idx_lb_2D = frs_idx_lb[i:j].reshape(-1, model.n)
            frs_lb_2D = frs_lb[i:j].reshape(-1, model.n)
            frs_ub_2D = frs_ub[i:j].reshape(-1, model.n)

            p, s_id, _, p_abs, keep_actions, number_nonzero, missing_absorbing = vmap_interval_distribution(
                                                                                    frs_idx_lb_2D,
                                                                                    frs_lb_2D,
                                                                                    frs_ub_2D)

            # Transfer outputs from device in one call to reduce synchronization overhead.
            # jax.device_get already returns numpy arrays, so np.asarray is not needed.
            p, s_id, p_abs, keep_actions, number_nonzero, missing_absorbing = jax.device_get((p, s_id, p_abs, keep_actions, number_nonzero, missing_absorbing))

            if partition.rectangular:
                assert np.all(missing_absorbing == 0), (
                    f"missing_absorbing must be zero for a rectangular partition, got max={missing_absorbing.max()}")
            max_nonzero = int(np.max(number_nonzero))

            # Reshape once to avoid expensive global masking/cumsum splitting.
            batch_states = j - i
            keep_actions = keep_actions.reshape(batch_states, nrA)
            p = p[:, :max_nonzero].reshape(batch_states, nrA, max_nonzero, 2)
            s_id = s_id[:, :max_nonzero].reshape(batch_states, nrA, max_nonzero)
            # Keep raw p_abs for debug logging before pAbs_min clamp
            p_abs_raw = np.round(p_abs, args.decimals).reshape(batch_states, nrA, 2)
            p_abs = np.maximum(args.pAbs_min, p_abs_raw)

            # --- Debug: log diagnostics when the target state has no enabled actions ---
            if debug_state is not None and i <= debug_state < j:
                dbg_idx = debug_state - i
                if not np.any(keep_actions[dbg_idx]):
                    logger.warning(
                        'State %d (debug_state) has NO enabled actions (batch [%d, %d), threshold=%.2f):',
                        debug_state, i, j, 0.1,
                    )
                    for a in range(nrA):
                        ma = missing_absorbing[dbg_idx * nrA + a]
                        pa_raw = p_abs_raw[dbg_idx, a]
                        logger.warning(
                            '  Action %3d: missing_absorbing=[%.4f, %.4f]  p_abs_raw=[%.4f, %.4f]  kept=%s',
                            actions_id[a], ma[0], ma[1], pa_raw[0], pa_raw[1], keep_actions[dbg_idx, a],
                        )
                else:
                    logger.info(
                        'State %d (debug_state) has %d enabled actions (batch [%d, %d)).',
                        debug_state, int(np.sum(keep_actions[dbg_idx])), i, j,
                    )

            for idx, s in enumerate(range(i, j)):
                keep_mask = keep_actions[idx]
                action_labels[s] = actions_id[keep_mask]
                interval_matrix[s] = p[idx, keep_mask]
                successor_id[s] = s_id[idx, keep_mask]
                interval_absorbing[s] = p_abs[idx, keep_mask]

            del p, s_id, p_abs, keep_actions, number_nonzero, missing_absorbing

    else:

        # For all states
        for s in tqdm(range(len(partition.regions['idxs']))):

            p, s_id, _, p_abs, keep_actions, number_nonzero, missing_absorbing = vmap_interval_distribution(
                                                                                frs_idx_lb[s].reshape(-1, model.n),
                                                                                frs_lb[s].reshape(-1, model.n),
                                                                                frs_ub[s].reshape(-1, model.n))

            p, s_id, p_abs, keep_actions, number_nonzero, missing_absorbing = jax.device_get((p, s_id, p_abs, keep_actions, number_nonzero, missing_absorbing))

            if partition.rectangular:
                assert np.all(missing_absorbing == 0), (
                    f"missing_absorbing must be zero for a rectangular partition, got max={missing_absorbing.max()}")
            max_nonzero = int(np.max(number_nonzero))

            # --- Debug: log diagnostics when the target state has no enabled actions ---
            if debug_state is not None and s == debug_state:
                if not np.any(keep_actions):
                    logger.warning(
                        'State %d (debug_state) has NO enabled actions (threshold=%.2f):',
                        debug_state, 0.1,
                    )
                    for a in range(nrA):
                        ma = missing_absorbing[a]
                        pa_raw = np.round(p_abs[a], args.decimals)
                        logger.warning(
                            '  Action %3d: missing_absorbing=[%.4f, %.4f]  p_abs_raw=[%.4f, %.4f]  kept=%s',
                            actions_id[a], ma[0], ma[1], pa_raw[0], pa_raw[1], keep_actions[a],
                        )
                else:
                    logger.info(
                        'State %d (debug_state) has %d enabled actions.',
                        debug_state, int(np.sum(keep_actions)),
                    )

            # k=True are the action indices that are to be kept (i.e., those with nonzero probabilities and for which the absorbing state probability is less than threshold)
            # p_nonzero=True means that the upper bound of the probability interval is greater than the minimum probability threshold
            # Evaluate p_nonzero over each columns to get the successor states that we should keep
            if np.any(keep_actions):
                action_labels[s] = actions_id[keep_actions]
                # Trim the trailing dimension before fancy-indexing to avoid a large intermediate copy.
                interval_matrix[s] = p[:, :max_nonzero][keep_actions]
                successor_id[s] = s_id[:, :max_nonzero][keep_actions]
                interval_absorbing[s] = np.maximum(args.pAbs_min, np.round(p_abs[keep_actions], args.decimals))
            del p, s_id, p_abs, missing_absorbing

    logger.info('-- Number of times function was compiled: %d', vmap_interval_distribution._cache_size())

    return interval_matrix, successor_id, action_labels, interval_absorbing