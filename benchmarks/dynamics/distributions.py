import numpy as np
import jax
import jax.numpy as jnp


class GaussianDistr(dict):
    def __init__(self, cov_diag):
        cov_diag = jnp.asarray(cov_diag, dtype=float)
        super().__init__(
            type='Gaussian',
            mean=jnp.zeros(cov_diag.shape[0], dtype=float),
            cov=jnp.diag(cov_diag),
            cov_diag=cov_diag,
            stdev=jnp.sqrt(cov_diag),
            support_radius=4 * jnp.sqrt(cov_diag), # 3 standard deviations cover 99.9% of the probability mass for a univariate Gaussian
        )

    def sample(self, size=None, rng=None):
        rng = np.random if rng is None else rng
        return rng.multivariate_normal(self['mean'], self['cov'], size=size)

    def sample_jax(self, rng, shape=()):
        mean = jnp.asarray(self['mean'])
        stdev = jnp.asarray(self['stdev'])
        if isinstance(shape, int):
            shape = (shape,)
        sample_shape = tuple(shape) + (mean.shape[0],)
        return mean + stdev * jax.random.normal(rng, shape=sample_shape)


class TriangularDistr(dict):
    def __init__(self, halfwidth):
        halfwidth = jnp.asarray(halfwidth, dtype=float)
        super().__init__(
            type='Triangular',
            mean=jnp.zeros(halfwidth.shape[0], dtype=float),
            halfwidth=halfwidth,
            support_radius=halfwidth, # Support radius of triangular distribution is equal to halfwidth
        )

    def sample(self, size=None, rng=None):
        rng = np.random if rng is None else rng
        mean = np.asarray(self['mean'])
        halfwidth = np.asarray(self['halfwidth'])
        nonzero = halfwidth > 0
        halfwidth_safe = np.where(nonzero, halfwidth, 1.0)
        if size is None:
            sample_size = mean.shape
        elif np.isscalar(size):
            sample_size = (int(size), mean.shape[0])
        else:
            sample_size = tuple(size) + (mean.shape[0],)
        sampled = rng.triangular(mean - halfwidth_safe, mean, mean + halfwidth_safe, size=sample_size)
        return np.where(nonzero, sampled, mean)

    def sample_jax(self, rng, shape=()):
        mean = jnp.asarray(self['mean'])
        halfwidth = jnp.asarray(self['halfwidth'])
        if isinstance(shape, int):
            shape = (shape,)
        sample_shape = tuple(shape) + (mean.shape[0],)
        rng1, rng2 = jax.random.split(rng)
        u1 = jax.random.uniform(rng1, shape=sample_shape)
        u2 = jax.random.uniform(rng2, shape=sample_shape)
        return mean + halfwidth * (u1 - u2)


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
        rng = np.random if rng is None else rng
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        out = rng.normal(0.0, self.sigma, size=shape + (self.dimension,))
        bad = np.sum(out**2, axis=-1) > self.radius**2
        while np.any(bad):
            out[bad] = rng.normal(0.0, self.sigma, size=(int(np.sum(bad)), self.dimension))
            bad = np.sum(out**2, axis=-1) > self.radius**2
        return out
