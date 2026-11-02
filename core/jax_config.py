import argparse
import logging

import jax
import numpy as np

logger = logging.getLogger(__name__)


def configure_jax(args: argparse.Namespace) -> None:
    jax.config.update("jax_default_matmul_precision", "high")

    # Keep explicitly requested int64 arrays, including grid strides, at int64.
    jax.config.update("jax_explicit_x64_dtypes", "allow")

    # args.floatprecision is float64 (set in options.py); without x64 mode JAX truncates it to float32
    jax.config.update("jax_enable_x64", True)

    if args.gpu:
        jax.config.update('jax_platform_name', 'gpu')
        logger.info('Requested to run on GPU')
    else:
        jax.config.update('jax_platform_name', 'cpu')
        logger.info('Requested to run on CPU')

    if args.gpu_rvi:
        args.rvi_device = jax.devices('gpu')[0]
        logger.info('Requested to run RVI on GPU')
    else:
        args.rvi_device = jax.devices('cpu')[0]
        logger.info('Requested to run RVI on CPU')

    logger.info('JAX backend in use: %s', args.rvi_device.platform)
    logger.debug('JAX devices (%s): %s', args.rvi_device.platform, jax.devices(args.rvi_device.platform))

    # Report the precision in use after configuring JAX.
    logger.info('Float precision: %s | JAX x64: %s | explicit int64 dtypes: %s',
                np.dtype(args.floatprecision).name,
                'enabled' if jax.config.read('jax_enable_x64') else 'disabled',
                jax.config.jax_explicit_x64_dtypes.name.lower())
