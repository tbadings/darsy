"""Isotropic multivariate Gaussian conditioned on a Euclidean ball.

Copy TruncatedGaussianBallDistr into benchmarks/dynamics/distributions.py,
or import this module from the benchmark.
"""
import numpy as np
import jax
import jax.numpy as jnp


class TruncatedGaussianBallDistr(dict):
    def __init__(self, dimension: int, stdev: float, truncation: float = 4.0):
        if dimension <= 0 or stdev <= 0 or truncation <= 0:
            raise ValueError('dimension, stdev and truncation must be positive')
        radius = float(stdev * truncation)
        super().__init__(type='TruncatedGaussianBall',
                         mean=jnp.zeros(dimension),
                         cov=jnp.eye(dimension) * stdev**2,
                         cov_diag=jnp.full(dimension, stdev**2),
                         stdev=jnp.full(dimension, stdev),
                         support_radius=jnp.full(dimension, radius),
                         ball_radius=radius,
                         truncation=float(truncation))
        self.dimension = int(dimension)
        self.sigma = float(stdev)
        self.radius = radius

    def sample_jax(self, rng, shape=()):
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)
        full_shape = shape + (self.dimension,)
        radius2 = self.radius**2

        def draw(key):
            key, subkey = jax.random.split(key)
            return key, self.sigma * jax.random.normal(subkey, full_shape)

        rng, samples = draw(rng)
        invalid = jnp.sum(samples**2, axis=-1) > radius2

        def cond(carry):
            _, _, bad = carry
            return jnp.any(bad)

        def body(carry):
            key, current, bad = carry
            key, proposals = draw(key)
            current = jnp.where(bad[..., None], proposals, current)
            bad = jnp.sum(current**2, axis=-1) > radius2
            return key, current, bad

        _, samples, _ = jax.lax.while_loop(cond, body, (rng, samples, invalid))
        return samples

    def sample(self, size=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        out = rng.normal(0.0, self.sigma, size=shape + (self.dimension,))
        bad = np.sum(out**2, axis=-1) > self.radius**2
        while np.any(bad):
            out[bad] = rng.normal(0.0, self.sigma, size=(int(np.sum(bad)), self.dimension))
            bad = np.sum(out**2, axis=-1) > self.radius**2
        return out
