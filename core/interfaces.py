"""Array types and the model interfaces consumed by the framework."""

from typing import Any, Protocol

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = jax.Array | NDArray[Any]
FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]


class NoiseDistribution(Protocol):
    """Noise metadata and samplers; samples append the state dimension."""

    def __getitem__(self, key: str) -> Any: ...

    def sample(
        self, size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> FloatArray: ...

    def sample_jax(self, rng: jax.Array, shape: int | tuple[int, ...] = ()) -> jax.Array: ...


class Dynamics(Protocol):
    """Stochastic dynamics with n state and p control dimensions."""

    n: int
    p: int
    noise: NoiseDistribution
    lipschitz: ArrayLike

    def step(self, state: ArrayLike, action: ArrayLike, noise: ArrayLike) -> Array: ...


class Benchmark(Dynamics, Protocol):
    """Dynamics plus partition settings, labeled geometry, and a specification."""

    partition: dict[str, ArrayLike]
    u_min: ArrayLike
    u_max: ArrayLike
    num_actions: list[int] | IntArray
    wrap: ArrayLike
    x0: ArrayLike
    goal: FloatArray
    critical: FloatArray
    regions: dict[str, FloatArray]
    labels: dict[str, set[str]]
    atomic_propositions: list[str]
    specification: str
    plot_dimensions: list[int]
    state_variables: list[str]

    def labeling_function(self, region: str, /) -> set[str]: ...


class SafetyBenchmark(Benchmark, Protocol):
    """A benchmark with safe boxes and a finite transition horizon."""

    safe_set: FloatArray
    horizon: int
