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