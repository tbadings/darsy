import logging
import time

import jax.numpy as jnp
import numpy as np


logger = logging.getLogger(__name__)


def parse_model(model, float_dtype):
    """Parse a dynamics model for abstraction: convert arrays to JAX and validate."""

    logger.info('Parsing model...')
    t = time.time()

    model.partition['boundary'] = jnp.array(model.partition['boundary']).astype(float_dtype)
    model.partition['number_per_dim'] = jnp.array(model.partition['number_per_dim']).astype(int)

    model.uMin = jnp.array(model.uMin, dtype=float_dtype)
    model.uMax = jnp.array(model.uMax, dtype=float_dtype)

    logger.info('Model parsing done (took %.3f sec.)', time.time() - t)
    return model


def identify_states_of_interest(partition, regions):
    """Map each named ROI to a mask of fully contained abstract states."""
    lower_bounds = np.asarray(partition.regions['lower_bounds'])
    upper_bounds = np.asarray(partition.regions['upper_bounds'])
    n_states = partition.size
    states_of_interest = {}
    for r, rectangles in regions.items():
        contained = np.zeros(n_states, dtype=bool)
        for rectangle in np.asarray(rectangles, dtype=float):
            region_lb, region_ub = rectangle
            contained |= (
                np.all(lower_bounds >= region_lb, axis=1)
                & np.all(upper_bounds <= region_ub, axis=1)
            )
        states_of_interest[r] = contained
    return states_of_interest


def attach_region_data(imdp, model, partition):
    """Attach region memberships and labels to a new or checkpointed IMDP."""
    if hasattr(partition, 'states_of_interest'):
        imdp.states_of_interest = {
            r: np.asarray(mask, dtype=bool).copy()
            for r, mask in partition.states_of_interest.items()
        }
    else:
        imdp.states_of_interest = identify_states_of_interest(partition, model.regions)
    if hasattr(partition, 'partial_region_cells'):
        imdp.partial_region_cells = {
            r: np.asarray(mask, dtype=bool).copy()
            for r, mask in partition.partial_region_cells.items()
        }
    if hasattr(partition, 'region_labels'):
        imdp.region_labels = {
            r: set(labels) for r, labels in partition.region_labels.items()
        }
    else:
        imdp.region_labels = {
            r: set(model.labeling_function(r)) for r in model.regions
        }
    imdp.atomic_propositions = list(
        partition.atomic_propositions
        if hasattr(partition, 'atomic_propositions')
        else model.atomic_propositions
    )
    return imdp
