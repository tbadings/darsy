import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from .polytope import hyperrectangles_iscontained_multi, hyperrectangles_isdisjoint_multi

EPS = 1e-4

logger = logging.getLogger(__name__)


@jax.jit
def meshgrid_jax(points, size):
    '''
    Set rectangular grid.

    :param points: Center points per dimension (list or arrays).
    :param size: Number of cells per dimension (list of ints).
    :return: Grid as 2D array.
    '''

    meshgrid = jnp.asarray(jnp.meshgrid(*points, indexing='ij'))
    grid = jnp.reshape(meshgrid, (len(size), -1)).T

    return grid


def define_grid_jax(low, high, size):
    '''
    Define a grid of the specified size, covering the box [low,high].

    :param low: Lower bound of the box to cover (array).
    :param high: Upper bound of the box to cover (array).
    :param size: Number of cells per dimension (list of ints).
    :return:
    '''
    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = meshgrid_jax(points, size)

    return grid


# Jitted vmapped kernels used during partition construction. Defined once at module level so the
# compilation cache is shared across every partition instance (re-wrapping them inside __init__ would
# force a fresh trace/compile each time).
vmap_hyperrectangles_iscontained = jax.jit(
    jax.vmap(hyperrectangles_iscontained_multi, in_axes=(0, 0, None, None), out_axes=0)
)
vmap_hyperrectangles_isdisjoint = jax.jit(
    jax.vmap(hyperrectangles_isdisjoint_multi, in_axes=(0, 0, None, None), out_axes=0)
)


def _compute_linear_strides(number_per_dim):
    # Row-major strides to map an n-D grid index [i0, ..., in-1] to a unique 1-D key.
    number_per_dim = np.asarray(number_per_dim, dtype=np.int64)
    strides = np.ones_like(number_per_dim, dtype=np.int64)
    if len(number_per_dim) > 1:
        strides[:-1] = np.cumprod(number_per_dim[1:][::-1], dtype=np.int64)[::-1]
    return strides


def _linear_key_dtype(number_per_dim):
    '''
    Integer dtype able to hold every linear key of this grid. The key indexes the *nominal* grid, so it
    scales with prod(number_per_dim) however few cells are actually kept: a sparse partition over a
    fine grid still needs the wide dtype. int32 keys that silently wrap collide two distinct cells onto
    one key, which searchsorted then resolves to the wrong state without erroring.
    '''
    max_key = int(np.prod(np.asarray(number_per_dim, dtype=np.int64))) - 1
    grid = 'x'.join(str(n) for n in np.asarray(number_per_dim).tolist())
    if max_key <= np.iinfo(np.int32).max:
        logger.info(f'- Partition cell keys: int32 (dense grid {grid} = {max_key + 1:,} cells)')
        return np.int32

    # JAX truncates int64 to int32 unless it is allowed to keep it (see core/jax_config.py), and only
    # warns while doing so. Fail loudly rather than build an index that quietly aliases cells.
    if jnp.zeros(1, dtype=jnp.int64).dtype != jnp.int64:
        raise ValueError(
            f"Grid {np.asarray(number_per_dim).tolist()} needs int64 cell keys (max key {max_key} exceeds "
            f"int32), but JAX is configured to truncate them to int32. Call core.jax_config."
            f"configure_jax first, or set jax.config.update('jax_explicit_x64_dtypes', 'allow')."
        )
    logger.info(f'- Partition cell keys: int64 (dense grid {grid} = {max_key + 1:,} cells, exceeds int32)')
    return np.int64


def _compute_goal_regions(goal_regions, lower_bounds, upper_bounds, region_idxs, size):
    '''Boolean mask + index list of partition cells fully contained in any goal region.'''
    t = time.time()
    if len(goal_regions) > 0:
        # Check which regions (hyperrectangles) are contained in the goal regions (also hyperrectangles)
        goal_lbs = goal_regions[:, 0, :]
        goal_ubs = goal_regions[:, 1, :]

        # Inflate goal regions by tiny EPS to avoid numerical issues with cells that are exactly on the boundary of a goal region.
        goal_regions_bools = vmap_hyperrectangles_iscontained(
            lower_bounds, upper_bounds, goal_lbs - EPS, goal_ubs + EPS
        )
        goal_regions_idxs = region_idxs[goal_regions_bools]
    else:
        goal_regions_bools = jnp.full(size, False, dtype=bool)
        goal_regions_idxs = jnp.array([], dtype=int)
    logger.debug(f'- Goal regions defined (took {(time.time() - t):.3f} sec.)')

    goal = {
        'bools': goal_regions_bools,
        'idxs': goal_regions_idxs.tolist(),  # TODO: Set should be more efficient here
    }
    logger.debug(f"-- Number of goal regions: {len(goal['idxs'])}")
    return goal


def _compute_critical_regions(critical_regions, lower_bounds, upper_bounds, region_idxs, size):
    '''Boolean mask + index list of partition cells overlapping any critical (unsafe) region.'''
    t = time.time()
    if len(critical_regions) > 0:
        # Check which regions (hyperrectangles) are *not* disjoint from the critical regions (also hyperrectangles)
        critical_lbs = critical_regions[:, 0, :]
        critical_ubs = critical_regions[:, 1, :]

        # Deflate critical regions by tiny EPS to avoid numerical issues with cells that are exactly on the boundary of a critical region.
        critical_regions_bools = ~vmap_hyperrectangles_isdisjoint(
            lower_bounds, upper_bounds, critical_lbs + EPS, critical_ubs - EPS
        )
        critical_regions_idxs = region_idxs[critical_regions_bools]
    else:
        critical_regions_bools = jnp.full(size, False, dtype=bool)
        critical_regions_idxs = jnp.array([], dtype=int)
    logger.debug(f'- Critical regions defined (took {(time.time() - t):.3f} sec.)')

    critical = {
        'bools': critical_regions_bools,
        'idxs': critical_regions_idxs.tolist(),  # TODO: Set should be more efficient here
    }
    logger.debug(f"-- Number of critical regions: {len(critical['idxs'])}")
    return critical


class _HyperrectangularPartition(object):
    """
    Base class for a partitioning of a state space into hyperrectangular regions (cells).

    Each cell is defined by its center, bounds, and vertices. Subclasses only decide *which* cells
    are kept and *what* action vectors each cell carries (via `_cells_and_actions`); the index maps,
    bounds, goal/critical detection, and coordinate lookup are all common and built here.
    """

    # Set by each subclass: True for the dense grid, False for a pruned sparse grid. Read by
    # downstream code (forward_reachability / successor_ids / probability_intervals) to pick fast paths.
    rectangular: bool = None

    def __init__(self, model):
        t_total = time.time()

        self.dimension = model.n

        # Retrieve necessary data from the model object
        self.number_per_dim = model.partition['number_per_dim']
        # Host-side copy: grid_idx2state runs in plain Python, once per simulation step.
        self.grid_shape = tuple(int(n) for n in self.number_per_dim)
        partition_boundary = model.partition['boundary']
        self.boundary_lb = partition_boundary[0]
        self.boundary_ub = partition_boundary[1]
        self.wrap = np.asarray(model.wrap, dtype=bool)

        # From the partition boundary, determine where the first grid centers are placed
        self.cell_width = (partition_boundary[1] - partition_boundary[0]) / self.number_per_dim
        lb_center = partition_boundary[0] + self.cell_width * 0.5

        # Subclass-specific: unit-cube grid indices of the kept cells and their action vectors.
        # actions is (1, num_actions, dim) for the dense grid (broadcast downstream) or
        # (num_states, num_actions, dim) for the sparse grid.
        t = time.time()
        centers_unit, actions = self._cells_and_actions(model)

        # JAX-friendly sparse index map via linearized coordinates and searchsorted.
        # We store sorted linear keys and aligned state IDs so lookup can stay inside JIT.
        idx_np_dtype = _linear_key_dtype(self.number_per_dim)
        idx_jnp_dtype = jnp.int64 if idx_np_dtype == np.int64 else jnp.int32
        region_linear_strides = _compute_linear_strides(self.number_per_dim).astype(idx_np_dtype)

        # Linear key for each kept cell index tuple.
        centers_np = np.asarray(centers_unit, dtype=idx_np_dtype)
        region_linear_idx = np.sum(centers_np * region_linear_strides, axis=1, dtype=idx_np_dtype)
        order = np.argsort(region_linear_idx)

        # region_linear_idx = the sorted linear keys of the kept cells (might contain gaps if the partition is sparse)
        # region_linear_state = the state ID of each kept cell (never contains gaps, always just a range)
        # region_linear_strides = the row-major strides to map an n-D grid index to a linear key
        self.region_linear_idx = jnp.asarray(region_linear_idx[order], dtype=idx_jnp_dtype)
        self.region_linear_state = jnp.asarray(np.arange(len(centers_np), dtype=np.int32)[order], dtype=jnp.int32)
        self.region_linear_strides = jnp.asarray(region_linear_strides, dtype=idx_jnp_dtype)

        # A dense grid emits its cells in row-major order, so the linear key of a cell *is* its state
        # ID. Both grid_idx2state and the DP's successor-ID fast path (successor_ids.box_to_ids_single)
        # depend on that; check it rather than assume it, since a violation would silently resolve
        # every lookup to the wrong state.
        if self.rectangular and not np.array_equal(region_linear_idx, np.arange(len(centers_np))):
            raise ValueError('Dense partition cells are not in row-major order.')

        # Unit-cube index tuple for each state (used to map a state ID back to grid coordinates).
        self.state_id2idx = centers_unit

        # Now scale the unit-cube partition to physical coordinates.
        centers_shifted = centers_unit * self.cell_width + lb_center

        region_idxs = jnp.arange(len(centers_shifted))
        lower_bounds = centers_shifted - self.cell_width / 2
        upper_bounds = centers_shifted + self.cell_width / 2

        logger.debug(f'- Grid points defined (took {(time.time() - t):.3f} sec.)')

        self.regions = {
            'centers': jnp.array(centers_shifted, dtype=float),
            'idxs': region_idxs,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'actions': actions,
        }
        self.size = len(centers_shifted)

        # Geometry-only region membership. No DFA dependency.
        self.regions_of_interest = {
            name: np.asarray(boxes, dtype=float).copy()
            for name, boxes in getattr(model, 'regions', {}).items()
        }
        self.region_labels = {
            name: frozenset(model.labeling_function(name))
            for name in self.regions_of_interest
        }
        self.atomic_propositions = tuple(getattr(model, 'atomic_propositions', ()))
        self._identify_regions_of_interest()

        # Sentinel state id used by JAX kernels for coordinates that are not present.
        self.missing_state = int(self.size)

        # Also store the partition bounds per dimension
        idx_per_dim = [jnp.arange(num) for num in self.number_per_dim]
        centers_per_dim = [idx_per_dim[i] * self.cell_width[i] + lb_center[i] for i in range(self.dimension)]
        lower_bounds_per_dim = [jnp.array(centers_per_dim[i] - self.cell_width[i] / 2) for i in range(self.dimension)]
        upper_bounds_per_dim = [jnp.array(centers_per_dim[i] + self.cell_width[i] / 2) for i in range(self.dimension)]

        self.regions_per_dim = {
            'centers': centers_per_dim,
            'idxs': idx_per_dim,
            'lower_bounds': lower_bounds_per_dim,
            'upper_bounds': upper_bounds_per_dim,
        }

        self.goal = _compute_goal_regions(
            model.goal, self.regions['lower_bounds'], self.regions['upper_bounds'], region_idxs, self.size
        )
        self.critical = _compute_critical_regions(
            model.critical, self.regions['lower_bounds'], self.regions['upper_bounds'], region_idxs, self.size
        )

        logger.debug(f'Partitioning took {(time.time() - t_total):.3f} sec.')

        logger.info(f"(Number of states: {len(self.regions['idxs'])})")
        logger.info(f"(Number of actions: {actions.shape[1]})")

    def _identify_regions_of_interest(self):
        """Identify cells intersecting each region of interest.

        A cell belongs to a region if it has positive-volume intersection
        with at least one hyperrectangle defining that region. Boundary-only
        contact does not count. The optional partial_region_cells masks are
        retained for diagnostics but do not restrict membership or labeling.
        """
        lbs = np.asarray(self.regions['lower_bounds'])
        ubs = np.asarray(self.regions['upper_bounds'])
        n, dim = lbs.shape
        self.states_of_interest = {}
        self.partial_region_cells = {}

        for name, boxes in self.regions_of_interest.items():
            boxes = np.asarray(boxes, dtype=float)
            if boxes.size == 0:
                boxes = np.empty((0, 2, dim), dtype=float)
            if boxes.ndim != 3 or boxes.shape[1:] != (2, dim):
                raise ValueError(f'Invalid region geometry for {name}')

            intersects = np.zeros(n, dtype=bool)
            contained = np.zeros(n, dtype=bool)
            for lo, hi in boxes:
                if np.any(lo > hi):
                    raise ValueError(f'Invalid bounds in region {name}')

                # Positive-volume intersection in every dimension.
                intersects |= np.all(lbs < hi, axis=1) & np.all(ubs > lo, axis=1)
                contained |= np.all(lbs >= lo, axis=1) & np.all(ubs <= hi, axis=1)

            # Include both fully contained and partially overlapping cells.
            self.states_of_interest[name] = intersects
            # Diagnostic only; not used to exclude cells from labels.
            self.partial_region_cells[name] = intersects & ~contained

    def _cells_and_actions(self, model):
        '''Return (centers_unit, actions): the kept unit-cube grid indices and their action vectors.'''
        raise NotImplementedError

    def x2state(self, x):
        '''
        Return the state ID for a given point x in the continuous state space.

        :param x: Point in the continuous state space.
        :return: (state ID, exists). exists is False (and the ID is the missing-state sentinel) when x
            is outside the partition boundary or falls in a cell absent from the (sparse) partition.
        '''
        # Discard points outside of partition
        in_partition = np.all((x >= self.boundary_lb) * (x <= self.boundary_ub))
        if not in_partition:
            return self.size, False

        # Normalize points to unit-cube grid indices
        x_norm = np.array(((x - self.boundary_lb) / (self.boundary_ub - self.boundary_lb) * self.number_per_dim) // 1, dtype=int)
        return self.grid_idx2state(x_norm)

    def grid_idx2state(self, grid_idx):
        '''Map a unit-cube grid index tuple to its (state ID, exists).'''
        
        key = 0
        for value, num in zip(grid_idx, self.grid_shape):
            value = int(value)
            if not 0 <= value < num:
                return self.size, False
            key = key * num + value

        if self.rectangular:
            # Dense grid: cells are stored in key order, so the key is the state ID.
            return key, True

        # Sparse grid: only some cells are kept, so the key has to be looked up among them.
        keys = np.asarray(self.region_linear_idx)
        pos = int(np.searchsorted(keys, key))
        if pos < len(keys) and keys[pos] == key:
            return int(np.asarray(self.region_linear_state)[pos]), True
        return self.size, False


class DensePartition(_HyperrectangularPartition):
    """Dense partition: every cell of the nominal grid is kept, all sharing one action set."""

    rectangular = True

    def __init__(self, model, verbose=False):
        logger.info('Define dense partition...')
        super().__init__(model)

        print('')

    def _cells_and_actions(self, model):
        # Full dense grid where each region is a unit cube.
        lb_unit = jnp.zeros(self.dimension, dtype=int)
        ub_unit = jnp.array(self.number_per_dim - 1, dtype=int)
        centers_unit = jnp.array(define_grid_jax(lb_unit, ub_unit, self.number_per_dim), dtype=int)

        # Discrete action grid: Cartesian product of the per-dimension action grids.
        discrete_actions_per_dimension = [
            np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
            for i in range(len(model.num_actions))
        ]
        action_array = jnp.array(list(itertools.product(*discrete_actions_per_dimension)))
        # Store as (1, num_actions, action_dim) — broadcast in forward_reachability to avoid
        # materializing a (num_states, num_actions, action_dim) array for large rectangular partitions.
        actions = action_array[None, :, :].astype(float)
        return centers_unit, actions


class SparsePartition(_HyperrectangularPartition):
    """Sparse partition: only the given active cells are kept, each with its own action set.

    Currently has no producer in the repo: the Main scripts build a `DensePartition`. Kept for
    callers that can supply their own `active_states` / `active_actions`.
    """

    rectangular = False

    def __init__(self, model, active_states, active_actions, verbose=False):
        logger.info('=== Define sparse partition ===')
        self._active_states = active_states
        self._active_actions = active_actions
        super().__init__(model)

        print('')

    def _cells_and_actions(self, model):
        # The sparse partition is defined directly by the given active states.
        centers_unit = jnp.array(self._active_states, dtype=int)

        # Gather each state's enabled action vectors from the per-state action map.
        keys = np.asarray(centers_unit, dtype=int).tolist()
        example = self._active_actions[tuple(keys[0])]
        actions = np.zeros((len(keys), example.shape[0], example.shape[1]), dtype=float)
        for i, key in enumerate(keys):
            a = self._active_actions[tuple(key)]
            assert a.shape == example.shape, (
                f"State {i} has action shape {a.shape}, expected {example.shape}. "
                "All states must have the same number of actions."
            )
            actions[i] = a
        return centers_unit, jnp.array(actions, dtype=float)


class CoarsePartition:
    def __init__(self, partition, factor):
        self.factor = int(factor)

        # Fine grid geometry
        self.fine_shape = np.asarray(partition.number_per_dim, dtype=np.int64)

        # Coarse grid
        self.coarse_shape = (self.fine_shape + self.factor - 1) // self.factor
        self.coarse_strides = _compute_linear_strides(self.coarse_shape)

        # parents: fine state id -> coarse cell id (blocks tile the grid, so every fine cell has one)
        fine_idx = np.asarray(partition.state_id2idx, dtype=np.int64)
        self.parents = ((fine_idx // self.factor) @ self.coarse_strides).astype(np.int32)

        # children: coarse cell id -> fine state ids (ascending, possibly empty)
        num_coarse = int(np.prod(self.coarse_shape))
        order = np.argsort(self.parents, kind='stable').astype(np.int32)
        splits = np.searchsorted(self.parents[order], np.arange(1, num_coarse))
        self.children = np.split(order, splits)

        logger.info(
            f"- Coarse partition: {'x'.join(str(n) for n in self.coarse_shape)} blocks (factor {self.factor}) "
            f"over fine grid {'x'.join(str(n) for n in self.fine_shape)}"
        )