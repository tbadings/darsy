import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.utils import create_batches

logger = logging.getLogger(__name__)


def _make_forward_reach(step_fn, noise_sample_fn, num_samples, alpha,
                        number_per_dim, cell_width, boundary_lb, boundary_ub, lipschitz):
    pct_lo = 100 * alpha / 2
    pct_hi = 100 * (1 - alpha / 2)
    norm_scale = number_per_dim / (boundary_ub - boundary_lb)

    def single_action_frs(state_min, state_max, action, rng):
        rng_state, rng_noise = jax.random.split(rng)
        states = jax.random.uniform(rng_state, shape=(num_samples, state_min.shape[0]),
                                    minval=state_min, maxval=state_max)
        noise = noise_sample_fn(rng_noise, shape=(num_samples,))
        next_states = jax.vmap(step_fn, in_axes=(0, None, 0))(states, action, noise)

        frs_min = jnp.percentile(next_states, pct_lo, axis=0)
        frs_max = jnp.percentile(next_states, pct_hi, axis=0)

        # Inflate FRS by cell diameter * Lipschitz matrix
        cell_diameter = state_max - state_min
        inflation = jnp.dot(cell_diameter, lipschitz.T)
        frs_min = frs_min - inflation
        frs_max = frs_max + inflation

        idx_low = jnp.clip(jnp.floor((frs_min - boundary_lb) * norm_scale), 0, number_per_dim - 1).astype(jnp.int16)
        idx_upp = jnp.clip(jnp.floor((frs_max - boundary_lb) * norm_scale), 0, number_per_dim - 1).astype(jnp.int32)

        return idx_low, idx_upp - idx_low + 1

    return jax.jit(jax.vmap(
        jax.vmap(single_action_frs, in_axes=(None, None, 0, 0)),
        in_axes=(0, 0, 0, 0),
    ))

class RectangularForward:
    def __init__(self, args, partition, model):
        logger.info('=== Start forward reachability computations ===')
        t_total = time.time()

        lipschitz_np = np.asarray(np.atleast_2d(model.lipschitz), dtype=np.float32)

        batch_forward_reach = _make_forward_reach(
            model.step,
            model.noise.sample_jax,
            args.num_samples,
            args.alpha,
            jax.device_put(partition.number_per_dim),
            jax.device_put(partition.cell_width),
            jax.device_put(partition.boundary_lb),
            jax.device_put(partition.boundary_ub),
            jax.device_put(lipschitz_np),
        )

        num_regions = len(partition.regions['lower_bounds'])
        num_actions = partition.regions['actions'].shape[1]
        dim = partition.dimension

        self.frs_idx_lb = np.zeros((num_regions, num_actions, dim), dtype=np.int16)
        max_span = np.zeros(dim, dtype=int)

        actions_slice = partition.regions['actions']
        lower_bounds = partition.regions['lower_bounds']
        upper_bounds = partition.regions['upper_bounds']

        master_rng = jax.random.PRNGKey(args.seed)
        starts, ends = create_batches(num_regions, args.frs_batch_size)

        for batch_start, batch_end in tqdm(zip(starts, ends), total=len(starts)):
            batch_size = batch_end - batch_start
            actions_batch = jnp.broadcast_to(actions_slice, (batch_size, *actions_slice.shape[1:]))

            master_rng, batch_rng = jax.random.split(master_rng)
            rngs = jax.random.split(batch_rng, batch_size * num_actions).reshape(batch_size, num_actions, 2)

            batch_idx_lb, batch_span = batch_forward_reach(
                lower_bounds[batch_start:batch_end],
                upper_bounds[batch_start:batch_end],
                actions_batch,
                rngs,
            )

            idx_lb_np, span_np = jax.device_get((batch_idx_lb, batch_span))
            self.frs_idx_lb[batch_start:batch_end] = idx_lb_np
            np.maximum(max_span, np.max(span_np, axis=(0, 1)), out=max_span)

        self.max_slice = tuple(max_span.tolist())
        self.id = np.arange(num_actions)

        logger.info(f'- Max state-slice to compute probability intervals over: {self.max_slice}')
        logger.info(f'- Forward reachable sets computed (took {(time.time() - t_total):.3f} sec.)')
        logger.info(f'Reachability computations took {(time.time() - t_total):.3f} sec.')
