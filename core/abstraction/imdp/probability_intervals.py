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
):
    """Create a JIT-compiled vmapped kernel to evaluate sample containment and intersections."""

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

        return counts_lb, counts_ub, succ_ids, cell_valid, count_abs_ub

    return jax.jit(jax.vmap(
        jax.vmap(single_action_counts, in_axes=(None, None, 0, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
    ))


def compute_probability_intervals(
    args,
    model,
    partition,
    actions,
    vectorized: bool = True,
    debug_state: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Compute transition probability intervals for an IMDP abstraction using data-driven sampling,
    Lipschitz-based inflation, and Clopper-Pearson confidence intervals.

    :param args: Argument namespace.
    :param model: Benchmark dynamical model.
    :param partition: State space partition.
    :param actions: RectangularForward reachability object with FRS indices.
    :param vectorized: Ignored, kept for API compatibility.
    :param debug_state: Optional initial/debug state index for logging.
    :return: Tuple of (P_full, S_id, A_id, P_absorbing).
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

    P_full: dict[int, np.ndarray] = {}
    S_id: dict[int, np.ndarray] = {}
    A_id: dict[int, np.ndarray] = {}
    P_absorbing: dict[int, np.ndarray] = {}

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

        res_lb, res_ub, res_succ, res_valid, res_abs_ub = batch_kernel(
            lb_batch,
            ub_batch,
            act_batch,
            frs_lb_batch,
            rngs,
        )

        counts_lb_np, counts_ub_np, succ_ids_np, valid_np, abs_ub_np = jax.device_get(
            (res_lb, res_ub, res_succ, res_valid, res_abs_ub)
        )

        # Look up probability intervals from Clopper-Pearson table for valid grid-aligned cells
        prob_lb = np.where(valid_np, cp_table_lb[counts_lb_np], 0.0)
        prob_ub = np.where(valid_np, cp_table_ub[counts_ub_np], 0.0)
        p_abs_ub = cp_table_ub[abs_ub_np]
        p_abs_lb = np.zeros_like(p_abs_ub)

        for i, s in enumerate(range(b_start, b_end)):
            s_int = int(s)
            S_id[s_int] = succ_ids_np[i].astype(np.int32)
            A_id[s_int] = action_ids
            P_full[s_int] = np.stack([prob_lb[i], prob_ub[i]], axis=-1).astype(float_dtype)
            P_absorbing[s_int] = np.stack([p_abs_lb[i], p_abs_ub[i]], axis=-1).astype(float_dtype)

    if debug_state is not None and debug_state in P_full:
        logger.info(f'- Debug state {debug_state}:')
        logger.info(f'  Enabled actions: {A_id[debug_state]}')
        logger.info(f'  Successors shape: {S_id[debug_state].shape}')
        logger.info(f'  P_absorbing: {P_absorbing[debug_state]}')

    logger.info(f'Probability intervals computed (took {time.time() - t_start:.3f} sec.)')
    return P_full, S_id, A_id, P_absorbing
