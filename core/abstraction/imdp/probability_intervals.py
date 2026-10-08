"""PAC transition probability intervals for IMDPs."""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from tqdm import tqdm

from core.abstraction.partition import _compute_linear_strides
from core.utils import create_batches

logger = logging.getLogger(__name__)


def compute_cp_table(num_samples: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Clopper-Pearson bounds for every count 0..num_samples."""
    k = np.arange(num_samples + 1)
    cp_table_lb = np.zeros(num_samples + 1, dtype=np.float64)  # count 0 -> lower 0
    cp_table_ub = np.ones(num_samples + 1, dtype=np.float64)  # count N -> upper 1

    # two-sided beta quantiles for the interior counts
    cp_table_lb[1:] = scipy.stats.beta.ppf(beta / 2.0, k[1:], num_samples - k[1:] + 1)
    cp_table_ub[:-1] = scipy.stats.beta.ppf(1.0 - beta / 2.0, k[:-1] + 1, num_samples - k[:-1])

    return cp_table_lb, cp_table_ub


def _window_offsets(window_shape):
    """Grid index offsets of all cells in a window, one row per cell."""
    dim = len(window_shape)
    return np.indices(window_shape).reshape(dim, -1).T  # row-major order


def _to_interval_columns(prob_lb, prob_ub):
    """Stack lower and upper bounds into (n, 2) interval pairs."""
    return np.stack([prob_lb, prob_ub], axis=-1)  # last axis is [lb, ub]


def _count_contained(box_lb, box_ub, ref_lb, ref_ub):
    """Per-reference-box count of sample boxes fully inside it."""
    # broadcast samples against reference boxes
    inside = jnp.all(
        (box_lb[:, None, :] >= ref_lb[None, :, :]) & (box_ub[:, None, :] <= ref_ub[None, :, :]),
        axis=-1,
    )
    return jnp.sum(inside, axis=0)  # one count per reference box


def _count_intersecting(box_lb, box_ub, ref_lb, ref_ub):
    """Per-reference-box count of sample boxes overlapping it."""
    # broadcast samples against reference boxes
    overlaps = jnp.all(
        (box_lb[:, None, :] <= ref_ub[None, :, :]) & (box_ub[:, None, :] >= ref_lb[None, :, :]),
        axis=-1,
    )
    return jnp.sum(overlaps, axis=0)  # one count per reference box


def _cells_in_grid(coords, grid_shape):
    """Mask of cells whose coords lie inside the grid."""
    in_bounds_dim = (coords >= 0) & (coords < grid_shape)  # per dim
    return jnp.all(in_bounds_dim, axis=-1)  # all dims in bounds


def _linear_ids(coords, strides, num_cells, valid):
    """Row-major cell ids; invalid cells get id 0."""
    clipped = jnp.clip(coords, 0, num_cells - 1)  # keep ids in range
    ids = jnp.sum(clipped * strides, axis=-1).astype(jnp.int32)  # row-major id
    return jnp.where(valid, ids, 0)  # invalid cells -> 0


def _make_batch_kernel(
    step_fn,
    noise_sample_fn,
    num_samples: int,
    offsets: jnp.ndarray,
    strides: jnp.ndarray,
    number_per_dim: jnp.ndarray,
    cell_width: jnp.ndarray,
    boundary_lb: jnp.ndarray,
    boundary_ub: jnp.ndarray,
    lipschitz: jnp.ndarray,
    max_slice: jnp.ndarray,
    coarse_offsets: jnp.ndarray,
    coarse_strides: jnp.ndarray,
    coarse_shape: jnp.ndarray,
    factor: int,
):
    """JIT-compiled vmapped kernel; coarse_* describe the coarse block window."""

    def single_action_counts(state_min, state_max, action, idx_lb, rng):
        # two rng streams: one for states, one for noise
        rng_state, rng_noise = jax.random.split(rng)

        # sample successor states
        states = jax.random.uniform(
            rng_state,
            shape=(num_samples, state_min.shape[0]),
            minval=state_min,
            maxval=state_max,
        )
        noise = noise_sample_fn(rng_noise, shape=(num_samples,))
        next_states = jax.vmap(step_fn, in_axes=(0, None, 0))(states, action, noise)

        # inflate samples by Lipschitz * distance to the state border
        dist_to_border = jnp.maximum(states - state_min, state_max - states)
        inflation = jnp.dot(dist_to_border, lipschitz.T)
        box_lb = next_states - inflation  # sample box, lower corner
        box_ub = next_states + inflation  # sample box, upper corner

        # fine successor cells of the FRS window
        grid_coords = idx_lb + offsets  # one row per cell
        cell_lb = grid_coords * cell_width + boundary_lb
        cell_ub = cell_lb + cell_width
        cell_valid = _cells_in_grid(grid_coords, number_per_dim)

        # lower count = contained boxes, upper count = intersecting boxes
        counts_lb = (_count_contained(box_lb, box_ub, cell_lb, cell_ub) * cell_valid).astype(jnp.int32)
        counts_ub = (_count_intersecting(box_lb, box_ub, cell_lb, cell_ub) * cell_valid).astype(jnp.int32)

        # absorbing state: continuous FRS box, clipped to the partition
        frs_cont_lb = idx_lb * cell_width + boundary_lb
        frs_cont_ub = (idx_lb + max_slice) * cell_width + boundary_lb
        frs_valid_lb = jnp.maximum(frs_cont_lb, boundary_lb)
        frs_valid_ub = jnp.minimum(frs_cont_ub, boundary_ub)

        # lower bound is 0 (built on the host); here only the upper count
        # every sample box leaving the FRS adds one to the upper count
        escapes_dim = (box_lb < frs_valid_lb) | (box_ub > frs_valid_ub)  # per dim
        count_abs_ub = jnp.sum(jnp.any(escapes_dim, axis=-1)).astype(jnp.int32)

        # fine successor ids
        succ_ids = _linear_ids(grid_coords, strides, number_per_dim, cell_valid)

        # coarse blocks of the FRS window
        block_coords = (idx_lb // factor) + coarse_offsets  # block grid coords
        block_valid = _cells_in_grid(block_coords, coarse_shape)

        # block boxes from the fine index range
        block_start = block_coords * factor  # first fine cell of the block
        block_end = jnp.minimum(block_start + factor, number_per_dim)  # shorter at the edge
        block_lb = boundary_lb + block_start * cell_width
        block_ub = boundary_lb + block_end * cell_width

        # clip blocks to the valid FRS box and drop empty ones
        block_lb = jnp.maximum(block_lb, frs_valid_lb)
        block_ub = jnp.minimum(block_ub, frs_valid_ub)
        block_valid = block_valid & jnp.all(block_lb < block_ub, axis=-1)

        # out-of-grid blocks get garbage boxes, but block_valid zeroes their counts
        coarse_counts_lb = (_count_contained(box_lb, box_ub, block_lb, block_ub) * block_valid).astype(jnp.int32)
        coarse_counts_ub = (_count_intersecting(box_lb, box_ub, block_lb, block_ub) * block_valid).astype(jnp.int32)

        # coarse successor ids
        coarse_ids = _linear_ids(block_coords, coarse_strides, coarse_shape, block_valid)

        return (counts_lb, counts_ub, succ_ids, cell_valid, count_abs_ub,
                coarse_counts_lb, coarse_counts_ub, coarse_ids, block_valid)

    # batched over states and then over actions
    return jax.jit(jax.vmap(
        jax.vmap(single_action_counts, in_axes=(None, None, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    ))


def compute_probability_intervals(
    args,
    model,
    partition,
    actions,
    coarse_partition,
    debug_state: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray],
          dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Transition probability intervals via sampling and Clopper-Pearson bounds."""
    logger.info('=== Start transition probability intervals computation ===')
    t_start = time.time()

    # problem sizes
    num_regions = len(partition.regions['lower_bounds'])
    num_actions = partition.regions['actions'].shape[1]

    # sampling and confidence settings
    num_samples = args.num_samples
    pac_beta = args.pac_beta
    float_dtype = args.floatprecision
    batch_size = args.batch_size

    # count -> probability lookup table
    cp_table_lb, cp_table_ub = compute_cp_table(num_samples, pac_beta)

    # fine layer offsets and strides
    offsets_np = _window_offsets(actions.max_slice)
    strides_np = _compute_linear_strides(partition.number_per_dim)
    lipschitz_np = np.asarray(np.atleast_2d(model.lipschitz), dtype=np.float32)

    # coarse layer offsets and strides
    coarse_factor = coarse_partition.factor
    if actions.coarse_factor != coarse_factor:  # layers must match
        raise ValueError(
            f'Coarse factor mismatch: actions built with {actions.coarse_factor}, '
            f'coarse partition built with {coarse_factor}.'
        )
    coarse_offsets_np = _window_offsets(actions.max_slice_coarse)
    coarse_strides_np = np.asarray(coarse_partition.coarse_strides)
    coarse_shape_np = np.asarray(coarse_partition.coarse_shape)

    # outputs, keyed by region id
    P_full: dict[int, np.ndarray] = {}  # fine intervals
    S_id: dict[int, np.ndarray] = {}  # fine successor ids
    A_id: dict[int, np.ndarray] = {}  # enabled action ids
    P_absorbing: dict[int, np.ndarray] = {}  # absorbing intervals
    Q_id: dict[int, np.ndarray] = {}  # coarse successor ids
    P_coarse: dict[int, np.ndarray] = {}  # coarse intervals

    action_ids = np.arange(num_actions, dtype=np.int32)  # all actions enabled
    actions_slice = partition.regions['actions']  # shared over regions
    lower_bounds = partition.regions['lower_bounds']
    upper_bounds = partition.regions['upper_bounds']

    master_rng = jax.random.PRNGKey(args.seed)  # one stream over all batches

    # jitted (states x actions) counting kernel
    batch_kernel = _make_batch_kernel(
        step_fn=model.step,
        noise_sample_fn=model.noise.sample_jax,
        num_samples=num_samples,
        offsets=jax.device_put(offsets_np),
        strides=jax.device_put(strides_np),
        number_per_dim=jax.device_put(partition.number_per_dim),
        cell_width=jax.device_put(partition.cell_width),
        boundary_lb=jax.device_put(partition.boundary_lb),
        boundary_ub=jax.device_put(partition.boundary_ub),
        lipschitz=jax.device_put(lipschitz_np),
        max_slice=jax.device_put(jnp.asarray(actions.max_slice, dtype=jnp.int32)),
        coarse_offsets=jax.device_put(coarse_offsets_np),
        coarse_strides=jax.device_put(coarse_strides_np),
        coarse_shape=jax.device_put(coarse_shape_np),
        factor=coarse_factor,
    )

    # one pass over the regions in batches
    starts, ends = create_batches(num_regions, batch_size)
    pbar = tqdm(zip(starts, ends), total=len(starts), desc="Probability intervals")

    for b_start, b_end in pbar:
        cur_batch_size = b_end - b_start

        # batch slices; the actions are shared and get broadcast
        lb_batch = lower_bounds[b_start:b_end]
        ub_batch = upper_bounds[b_start:b_end]
        frs_lb_batch = actions.frs_idx_lb[b_start:b_end]
        act_batch = jnp.broadcast_to(actions_slice, (cur_batch_size, *actions_slice.shape[1:]))

        # fresh rng key per (state, action) of the batch
        master_rng, subkey = jax.random.split(master_rng)
        rngs = jax.random.split(subkey, cur_batch_size * num_actions).reshape(cur_batch_size, num_actions, 2)

        # run the kernel and move results to host
        results = jax.device_get(batch_kernel(lb_batch, ub_batch, act_batch, frs_lb_batch, rngs))
        (counts_lb, counts_ub, succ_ids, cell_valid, count_abs_ub,
         coarse_counts_lb, coarse_counts_ub, coarse_ids, coarse_valid) = results

        # counts -> probability intervals; invalid cells stay 0
        prob_lb = np.where(cell_valid, cp_table_lb[counts_lb], 0.0)
        prob_ub = np.where(cell_valid, cp_table_ub[counts_ub], 0.0)
        # absorbing interval: lower bound 0, upper from the count
        p_abs_ub = cp_table_ub[count_abs_ub]
        p_abs_lb = np.zeros_like(p_abs_ub)
        # same lookup for the coarse blocks
        c_prob_lb = np.where(coarse_valid, cp_table_lb[coarse_counts_lb], 0.0)
        c_prob_ub = np.where(coarse_valid, cp_table_ub[coarse_counts_ub], 0.0)

        # store one entry per region
        for i, s in enumerate(range(b_start, b_end)):
            s_int = int(s)
            S_id[s_int] = succ_ids[i].astype(np.int32)  # fine successors
            A_id[s_int] = action_ids  # all actions enabled
            P_full[s_int] = _to_interval_columns(prob_lb[i], prob_ub[i]).astype(float_dtype)
            P_absorbing[s_int] = _to_interval_columns(p_abs_lb[i], p_abs_ub[i]).astype(float_dtype)
            Q_id[s_int] = coarse_ids[i].astype(np.int32)  # coarse successors
            P_coarse[s_int] = _to_interval_columns(c_prob_lb[i], c_prob_ub[i]).astype(float_dtype)

    # debug output for one region
    if debug_state is not None and debug_state in P_full:
        logger.info(f'- Debug state {debug_state}:')
        logger.info(f'  Enabled actions: {A_id[debug_state]}')
        logger.info(f'  Successors shape: {S_id[debug_state].shape}')
        logger.info(f'  P_absorbing: {P_absorbing[debug_state]}')
        logger.info(f'  Coarse successors shape: {Q_id[debug_state].shape}')

    logger.info(f'Probability intervals computed (took {time.time() - t_start:.3f} sec.)')
    return P_full, S_id, A_id, P_absorbing, Q_id, P_coarse
