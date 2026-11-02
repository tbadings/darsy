"""Robust expectation kernels for IMDPs and 2IMDPs.

Both kernels return the worst-case (or, with maximize=True, the best-case) expected value of a
vector of successor values over all transition probabilities allowed by the intervals. The last
axis of every array lists the successors of a state-action pair (the absorbing state included);
all leading axes are batch axes.
"""

from functools import partial

import jax
import jax.numpy as jnp


# ============================================================================
# IMDP: FINE INTERVALS ONLY
# ============================================================================


def expectation_imdp(values, data, maximize):
    """min (max) of p @ values subject to lb <= p <= ub and sum(p) = 1.

    Exact greedy: start every successor at its lower bound, then hand out the remaining mass to the
    successors in order of increasing value (decreasing value when maximizing).

    :param values: Successor values, shape (..., K).
    :param data: Dict with 'lb' and 'ub', broadcastable to values.
    :param maximize: If True, compute the best case instead of the worst case.
    """
    lb = jnp.broadcast_to(data['lb'], values.shape)
    ub = jnp.broadcast_to(data['ub'], values.shape)

    order = jnp.argsort(-values if maximize else values, axis=-1)
    lo = jnp.take_along_axis(lb, order, axis=-1)
    hi = jnp.take_along_axis(ub, order, axis=-1)
    vals = jnp.take_along_axis(values, order, axis=-1)

    budget = jnp.maximum(0.0, 1.0 - jnp.sum(lo, axis=-1, keepdims=True))
    capacity = jnp.maximum(0.0, hi - lo)
    preceding = jnp.cumsum(capacity, axis=-1) - capacity  # capacity of all successors handled earlier
    extra = jnp.minimum(capacity, jnp.maximum(0.0, budget - preceding))
    return jnp.clip(jnp.sum((lo + extra) * vals, axis=-1), 0.0, 1.0)


# ============================================================================
# 2IMDP: FINE INTERVALS AND COARSE BLOCK CONSTRAINTS
# ============================================================================


def _expectation_2imdp_pair(values, lb, ub, mask, block, block_lb, block_ub, block_mask, maximize):
    """Robust expectation for one state-action pair of a 2IMDP.

    The probability of successor k lies in [lb[k], ub[k]], and the total probability of the
    successors in coarse block q lies in [block_lb[q], block_ub[q]]. Solved greedily in two phases:

    1. Start from the fine lower bounds and meet each block's lower bound with the block's own
       successors, block by block, cheapest successor first (dearest when maximizing).
    2. Hand out the remaining mass to all successors, cheapest first, within the fine and block
       upper bounds.

    :param values: Successor values, shape (K,).
    :param mask: Successors that exist (unused window slots are False), shape (K,).
    :param block: Block of each successor (index into the block_* arrays), or -1, shape (K,).
    :param block_mask: Blocks that exist, shape (Q,).
    """
    num_successors, num_blocks = values.shape[0], block_lb.shape[0]

    key = jnp.where(mask, -values if maximize else values, jnp.inf)  # handle the smallest key first
    in_block = mask & (block >= 0) & block_mask[jnp.maximum(block, 0)]
    safe_block = jnp.where(in_block, block, 0)

    # Start from the fine lower bounds
    gamma = jnp.where(mask, lb, 0.0)
    gamma_block = jnp.zeros_like(block_lb).at[safe_block].add(jnp.where(in_block, gamma, 0.0))
    remaining = 1.0 - jnp.sum(gamma)

    # Phase 1: successors grouped by block (in block order), cheapest first within each block
    required = jnp.where(block_mask, jnp.maximum(0.0, block_lb - gamma_block), 0.0)
    phase1_order = jnp.lexsort((key, jnp.where(in_block, block, num_blocks)))

    def phase1(i, carry):
        gamma, gamma_block, required, remaining = carry
        k = phase1_order[i]
        q = safe_block[k]
        allocation = jnp.minimum(required[q], jnp.minimum(jnp.maximum(0.0, ub[k] - gamma[k]), remaining))
        allocation = jnp.where(in_block[k] & (required[q] > 0.0) & (remaining > 0.0), allocation, 0.0)
        return (gamma.at[k].add(allocation), gamma_block.at[q].add(allocation),
                required.at[q].add(-allocation), remaining - allocation)

    gamma, gamma_block, _, remaining = jax.lax.fori_loop(
        0, num_successors, phase1, (gamma, gamma_block, required, remaining))

    # Phase 2: remaining mass to all successors, cheapest first
    phase2_order = jnp.argsort(key)

    def phase2(i, carry):
        gamma, gamma_block, remaining = carry
        k = phase2_order[i]
        q = safe_block[k]
        block_capacity = jnp.where(in_block[k], jnp.maximum(0.0, block_ub[q] - gamma_block[q]), jnp.inf)
        allocation = jnp.minimum(remaining, jnp.minimum(jnp.maximum(0.0, ub[k] - gamma[k]), block_capacity))
        allocation = jnp.where(mask[k] & (remaining > 0.0), allocation, 0.0)
        return (gamma.at[k].add(allocation), gamma_block.at[q].add(jnp.where(in_block[k], allocation, 0.0)),
                remaining - allocation)

    gamma, _, _ = jax.lax.fori_loop(0, num_successors, phase2, (gamma, gamma_block, remaining))
    return jnp.clip(jnp.sum(gamma * values), 0.0, 1.0)


def expectation_2imdp(values, data, maximize):
    """Robust expectation of a 2IMDP for every state-action pair in a batch.

    :param values: Successor values, shape (..., K).
    :param data: Dict with per-successor 'lb', 'ub', 'mask', 'block' (..., K) and per-block
        'block_lb', 'block_ub', 'block_mask' (..., Q), broadcastable to the batch axes of values.
    :param maximize: If True, compute the best case instead of the worst case.
    """
    if data['block_lb'].shape[-1] == 0:  # no blocks: only the fine intervals constrain
        return expectation_imdp(values, data, maximize)

    batch = values.shape[:-1]

    def flat(x):  # broadcast over the batch axes, then merge them into one
        x = jnp.broadcast_to(x, batch + x.shape[-1:])
        return x.reshape((-1,) + x.shape[-1:])

    keys = ('lb', 'ub', 'mask', 'block', 'block_lb', 'block_ub', 'block_mask')
    pair = partial(_expectation_2imdp_pair, maximize=maximize)
    return jax.vmap(pair)(flat(values), *(flat(data[k]) for k in keys)).reshape(batch)
