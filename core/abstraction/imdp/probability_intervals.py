"""
Computation of transition probability intervals for Interval Markov Decision Processes (IMDPs).

Given a partitioned continuous state space and forward reachable sets (FRS), this module
generates empirical transition samples, inflates them via the system's Lipschitz matrix,
counts containment and intersection against FRS successor boxes, and constructs
Probably Approximately Correct (PAC) intervals via Clopper-Pearson bounds.
"""

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
    """
    Compute Clopper-Pearson confidence interval lookup table for binomial counts.

    :param num_samples: Total number of samples N.
    :param beta: Error probability / significance level (confidence level is 1 - beta).
    :return: Tuple of (cp_table_lb, cp_table_ub) arrays of length num_samples + 1.
    """
    k = np.arange(num_samples + 1)
    cp_table_lb = np.zeros(num_samples + 1, dtype=np.float64)
    cp_table_ub = np.ones(num_samples + 1, dtype=np.float64)

    cp_table_lb[1:] = scipy.stats.beta.ppf(beta / 2.0, k[1:], num_samples - k[1:] + 1)
    cp_table_ub[:-1] = scipy.stats.beta.ppf(1.0 - beta / 2.0, k[:-1] + 1, num_samples - k[:-1])

    return cp_table_lb, cp_table_ub


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
    """Create a JIT-compiled vmapped kernel to evaluate sample containment and intersections.

    The coarse_* arguments describe the block window of the coarse layer.
    """

    def single_action_counts(state_min, state_max, action, idx_lb, rng):
        rng_state, rng_noise = jax.random.split(rng)
        states = jax.random.uniform(
            rng_state,
            shape=(num_samples, state_min.shape[0]),
            minval=state_min,
            maxval=state_max,
        )
        noise = noise_sample_fn(rng_noise, shape=(num_samples,))
        next_states = jax.vmap(step_fn, in_axes=(0, None, 0))(states, action, noise)

        # Inflate next_states using max distance to state border and Lipschitz matrix
        dist_to_border = jnp.maximum(states - state_min, state_max - states)
        inflation = jnp.dot(dist_to_border, lipschitz.T)

        b_lb = next_states - inflation
        b_ub = next_states + inflation

        # Successor grid cells from FRS slice
        grid_coords = idx_lb + offsets
        succ_lb = grid_coords * cell_width + boundary_lb
        succ_ub = succ_lb + cell_width

        # In-bounds mask for successor cells
        in_bounds_dim = (grid_coords >= 0) & (grid_coords < number_per_dim)
        cell_valid = jnp.all(in_bounds_dim, axis=-1)

        # Check containment and intersections against FRS cells
        inside = jnp.all(
            (b_lb[:, None, :] >= succ_lb[None, :, :]) & (b_ub[:, None, :] <= succ_ub[None, :, :]),
            axis=-1,
        )
        intersects = jnp.all(
            (b_lb[:, None, :] <= succ_ub[None, :, :]) & (b_ub[:, None, :] >= succ_lb[None, :, :]),
            axis=-1,
        )

        counts_lb = (jnp.sum(inside, axis=0) * cell_valid).astype(jnp.int32)
        counts_ub = (jnp.sum(intersects, axis=0) * cell_valid).astype(jnp.int32)

        # Absorbing (unsafe) state counts:
        # Continuous bounds of the FRS clipped to valid partition boundaries
        frs_cont_lb = idx_lb * cell_width + boundary_lb
        frs_cont_ub = (idx_lb + max_slice) * cell_width + boundary_lb
        frs_valid_lb = jnp.maximum(frs_cont_lb, boundary_lb)
        frs_valid_ub = jnp.minimum(frs_cont_ub, boundary_ub)

        # For unsafe/absorbing, lower bound is zero; only compute upper bound
        count_abs_lb = jnp.int32(0)

        # All boxes that intersect with outside of the FRS count towards the upper bound of absorbing
        not_inside_frs_dim = (b_lb < frs_valid_lb) | (b_ub > frs_valid_ub)
        box_intersects_abs = jnp.any(not_inside_frs_dim, axis=-1)
        count_abs_ub = jnp.sum(box_intersects_abs).astype(jnp.int32)

        # Successor state IDs
        clipped_grid = jnp.clip(grid_coords, 0, number_per_dim - 1)
        succ_ids = jnp.sum(clipped_grid * strides, axis=-1).astype(jnp.int32)
        succ_ids = jnp.where(cell_valid, succ_ids, 0)

        # Coarse (block) layer, using the same sample boxes b_lb / b_ub.
        c_coords = (idx_lb // factor) + coarse_offsets                    # [Mc, dim]
        c_valid = jnp.all((c_coords >= 0) & (c_coords < coarse_shape), axis=-1)  # [Mc]

        # Block boxes from the fine index range
        c_fine_start = c_coords * factor
        c_fine_end = jnp.minimum(c_fine_start + factor, number_per_dim)   # shorter at the edge
        c_lb = boundary_lb + c_fine_start * cell_width
        c_ub = boundary_lb + c_fine_end * cell_width

        c_lb = jnp.maximum(c_lb, frs_valid_lb)
        c_ub = jnp.minimum(c_ub, frs_valid_ub)
        c_nonempty = jnp.all(c_lb < c_ub, axis=-1)
        c_valid = c_valid & c_nonempty

        c_inside = jnp.all(
            (b_lb[:, None, :] >= c_lb[None, :, :]) & (b_ub[:, None, :] <= c_ub[None, :, :]),
            axis=-1,
        )
        c_intersects = jnp.all(
            (b_lb[:, None, :] <= c_ub[None, :, :]) & (b_ub[:, None, :] >= c_lb[None, :, :]),
            axis=-1,
        )

        # For out-of-range coarse cells c_lb/c_ub are garbage, but c_valid zeroes their counts
        c_counts_lb = (jnp.sum(c_inside, axis=0) * c_valid).astype(jnp.int32)
        c_counts_ub = (jnp.sum(c_intersects, axis=0) * c_valid).astype(jnp.int32)

        c_ids = jnp.sum(jnp.clip(c_coords, 0, coarse_shape - 1) * coarse_strides, axis=-1).astype(jnp.int32)
        c_ids = jnp.where(c_valid, c_ids, 0)

        return (counts_lb, counts_ub, succ_ids, cell_valid, count_abs_ub,
                c_counts_lb, c_counts_ub, c_ids, c_valid)

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
    """
    Compute transition probability intervals for an IMDP abstraction using data-driven sampling,
    Lipschitz-based inflation, and Clopper-Pearson confidence intervals.

    :param args: Argument namespace.
    :param model: Benchmark dynamical model.
    :param partition: State space partition.
    :param actions: RectangularForward reachability object with FRS indices.
    :param coarse_partition: CoarsePartition
    :param debug_state: Optional initial/debug state index for logging.
    :return: Tuple of (P_full, S_id, A_id, P_absorbing, Q_id, P_coarse).
    """
    logger.info('=== Start transition probability intervals computation ===')
    t_start = time.time()

    num_regions = len(partition.regions['lower_bounds'])
    num_actions = partition.regions['actions'].shape[1]
    dim = partition.dimension

    num_samples = args.num_samples
    pac_beta = args.pac_beta
    float_dtype = args.floatprecision
    batch_size = args.batch_size

    # 1. Precompute Clopper-Pearson lookup table
    cp_table_lb, cp_table_ub = compute_cp_table(num_samples, pac_beta)

    # Grid offsets for each cell in the FRS slice
    offsets_np = np.indices(actions.max_slice).reshape(dim, -1).T
    strides_np = _compute_linear_strides(partition.number_per_dim)
    lipschitz_np = np.asarray(np.atleast_2d(model.lipschitz), dtype=np.float32)

    # Coarse layer statics
    coarse_factor = coarse_partition.factor
    if actions.coarse_factor != coarse_factor:
        raise ValueError(
            f'Coarse factor mismatch: actions built with {actions.coarse_factor}, '
            f'coarse partition built with {coarse_factor}.'
        )
    coarse_offsets_np = np.indices(actions.max_slice_coarse).reshape(dim, -1).T
    coarse_strides_np = np.asarray(coarse_partition.coarse_strides)
    coarse_shape_np = np.asarray(coarse_partition.coarse_shape)

    P_full: dict[int, np.ndarray] = {}
    S_id: dict[int, np.ndarray] = {}
    A_id: dict[int, np.ndarray] = {}
    P_absorbing: dict[int, np.ndarray] = {}
    Q_id: dict[int, np.ndarray] = {}
    P_coarse: dict[int, np.ndarray] = {}

    action_ids = np.arange(num_actions, dtype=np.int32)
    actions_slice = partition.regions['actions']
    lower_bounds = partition.regions['lower_bounds']
    upper_bounds = partition.regions['upper_bounds']

    master_rng = jax.random.PRNGKey(args.seed)

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

    starts, ends = create_batches(num_regions, batch_size)
    pbar = tqdm(zip(starts, ends), total=len(starts), desc="Probability intervals")

    for b_start, b_end in pbar:
        cur_batch_size = b_end - b_start

        lb_batch = lower_bounds[b_start:b_end]
        ub_batch = upper_bounds[b_start:b_end]
        frs_lb_batch = actions.frs_idx_lb[b_start:b_end]

        act_batch = jnp.broadcast_to(actions_slice, (cur_batch_size, *actions_slice.shape[1:]))

        master_rng, subkey = jax.random.split(master_rng)
        rngs = jax.random.split(subkey, cur_batch_size * num_actions).reshape(cur_batch_size, num_actions, 2)

        (res_lb, res_ub, res_succ, res_valid, res_abs_ub,
         res_c_lb, res_c_ub, res_c_ids, res_c_valid) = batch_kernel(
            lb_batch,
            ub_batch,
            act_batch,
            frs_lb_batch,
            rngs,
        )

        (counts_lb_np, counts_ub_np, succ_ids_np, valid_np, abs_ub_np,
         c_counts_lb_np, c_counts_ub_np, c_ids_np, c_valid_np) = jax.device_get(
            (res_lb, res_ub, res_succ, res_valid, res_abs_ub,
             res_c_lb, res_c_ub, res_c_ids, res_c_valid)
        )

        # Look up probability intervals from Clopper-Pearson table for valid grid-aligned cells
        prob_lb = np.where(valid_np, cp_table_lb[counts_lb_np], 0.0)
        prob_ub = np.where(valid_np, cp_table_ub[counts_ub_np], 0.0)
        p_abs_ub = cp_table_ub[abs_ub_np]
        p_abs_lb = np.zeros_like(p_abs_ub)

        # Same Clopper-Pearson table, applied to the directly counted block counts
        c_prob_lb = np.where(c_valid_np, cp_table_lb[c_counts_lb_np], 0.0)
        c_prob_ub = np.where(c_valid_np, cp_table_ub[c_counts_ub_np], 0.0)

        for i, s in enumerate(range(b_start, b_end)):
            s_int = int(s)
            S_id[s_int] = succ_ids_np[i].astype(np.int32)
            A_id[s_int] = action_ids
            P_full[s_int] = np.stack([prob_lb[i], prob_ub[i]], axis=-1).astype(float_dtype)
            P_absorbing[s_int] = np.stack([p_abs_lb[i], p_abs_ub[i]], axis=-1).astype(float_dtype)
            Q_id[s_int] = c_ids_np[i].astype(np.int32)
            P_coarse[s_int] = np.stack([c_prob_lb[i], c_prob_ub[i]], axis=-1).astype(float_dtype)

    if debug_state is not None and debug_state in P_full:
        logger.info(f'- Debug state {debug_state}:')
        logger.info(f'  Enabled actions: {A_id[debug_state]}')
        logger.info(f'  Successors shape: {S_id[debug_state].shape}')
        logger.info(f'  P_absorbing: {P_absorbing[debug_state]}')
        logger.info(f'  Coarse successors shape: {Q_id[debug_state].shape}')

    logger.info(f'Probability intervals computed (took {time.time() - t_start:.3f} sec.)')
    return P_full, S_id, A_id, P_absorbing, Q_id, P_coarse
