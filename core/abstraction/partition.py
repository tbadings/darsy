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


def subdivide_interval_edges(edges, max_width):
    '''Split each interval into equal pieces no wider than max_width'''
    refined = [edges[:1]]
    for left, right in zip(edges[:-1], edges[1:]):
        # Small tolerance so e.g. 0.2 / 0.1 counts as 2 pieces, not 3.
        num_pieces = int(np.ceil((right - left) / max_width - 1e-9))
        refined.append(np.linspace(left, right, num_pieces + 1)[1:])
    return np.concatenate(refined)


def grid_edges(boundary_lb, boundary_ub, region_boxes, max_cell_width):
    '''
    Grid edges per dimension
    :param region_boxes: Boxes whose faces become grid edges, shape (K, 2, dim).
    :param max_cell_width: Maximum cell width per dimension.
    '''
    edges = []
    for d in range(len(boundary_lb)):
        lb = boundary_lb[d]
        ub = boundary_ub[d]
        faces = region_boxes[:, :, d].ravel()
        inner_faces = faces[(faces > lb) & (faces < ub)]
        coarse_edges = np.unique(np.concatenate([[lb, ub], inner_faces]))
        edges.append(subdivide_interval_edges(coarse_edges, max_cell_width[d]))
    return edges


def _region_boxes(model):
    '''All boxes that should align with the grid'''
    boxes = [
        np.asarray(model.goal, dtype=float).reshape(-1, 2, model.n),
        np.asarray(model.critical, dtype=float).reshape(-1, 2, model.n),
    ]
    for region in getattr(model, 'regions', {}).values():
        boxes.append(np.asarray(region, dtype=float).reshape(-1, 2, model.n))
    return np.concatenate(boxes)


def _pad_edges(edges):
    max_len = max(len(e) for e in edges)
    padded = np.full((len(edges), max_len), np.inf)
    for d, e in enumerate(edges):
        padded[d, :len(e)] = e
    return padded


def edge_coords(edges_padded, k, number_per_dim, period, wrap):
    laps = jnp.where(wrap, k // number_per_dim, 0)
    local = jnp.clip(k - laps * number_per_dim, 0, number_per_dim)
    dims = jnp.arange(edges_padded.shape[0])
    return edges_padded[dims, local] + laps * period


def coords_to_grid(edges_padded, x, number_per_dim, boundary_lb, period, wrap):
    laps = jnp.where(wrap, jnp.floor((x - boundary_lb) / period), 0).astype(jnp.int32)
    x_local = x - laps * period
    # Search each dimension's edges (axis 0 of edges_padded) for that dimension's coordinates (last axis of x)
    search = jax.vmap(lambda e, v: jnp.searchsorted(e, v, side='right'), in_axes=(0, -1), out_axes=-1)
    local = jnp.clip(search(edges_padded, x_local) - 1, 0, number_per_dim - 1)
    return laps * number_per_dim + local


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


class DensePartition:
    """Full hyperrectangular grid with one shared discrete action set."""

    def __init__(self, model, float_dtype, verbose=False):
        logger.info('Define dense partition...')
        t_total = time.time()
        self.float_dtype = float_dtype

        self.dimension = model.n

        # Retrieve necessary data from the model object
        partition_boundary = model.partition['boundary']
        self.boundary_lb = partition_boundary[0]
        self.boundary_ub = partition_boundary[1]
        self.wrap = np.asarray(model.wrap, dtype=bool)
        lb_np = np.asarray(self.boundary_lb, dtype=float_dtype)
        ub_np = np.asarray(self.boundary_ub, dtype=float_dtype)

        # Maximum cell width per dimension
        widths = (ub_np - lb_np) / np.asarray(model.partition['number_per_dim'])

        # Grid edges per dimension
        self.edges = grid_edges(lb_np, ub_np, _region_boxes(model), widths)
        self.edges_padded = jnp.asarray(_pad_edges(self.edges), dtype=float_dtype)
        self.number_per_dim = jnp.array([len(e) - 1 for e in self.edges], dtype=int)
        logger.info(f'- Partition with max cell width {widths.tolist()}')
        self.grid_shape = tuple(int(n) for n in self.number_per_dim)

        # for plots
        self.cell_width = (partition_boundary[1] - partition_boundary[0]) / self.number_per_dim

        # Every grid cell shares actions of shape (1, num_actions, action_dim).
        t = time.time()
        centers_unit, actions = self._cells_and_actions(model)

        # State IDs and transition kernels both rely on row-major cell order.
        centers_np = np.asarray(centers_unit, dtype=np.int64)
        linear_ids = centers_np @ _compute_linear_strides(self.number_per_dim)
        if not np.array_equal(linear_ids, np.arange(len(centers_np))):
            raise ValueError('Dense partition cells are not in row-major order.')

        # Unit-cube index tuple for each state (used to map a state ID back to grid coordinates).
        self.state_id2idx = centers_unit

        # Now map the unit-cube grid indices to physical cell bounds via the grid edges.
        idx_np = np.asarray(centers_unit, dtype=int)
        lower_np = np.zeros(idx_np.shape, dtype=float_dtype)
        upper_np = np.zeros(idx_np.shape, dtype=float_dtype)
        for d in range(self.dimension):
            lower_np[:, d] = self.edges[d][idx_np[:, d]]
            upper_np[:, d] = self.edges[d][idx_np[:, d] + 1]

        lower_bounds = jnp.asarray(lower_np)
        upper_bounds = jnp.asarray(upper_np)
        centers_shifted = (lower_bounds + upper_bounds) / 2
        region_idxs = jnp.arange(len(centers_shifted))

        logger.debug(f'- Grid points defined (took {(time.time() - t):.3f} sec.)')

        self.regions = {
            'centers': jnp.array(centers_shifted, dtype=self.float_dtype),
            'idxs': region_idxs,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'actions': actions,
        }
        self.size = len(centers_shifted)

        # Geometry-only region membership. No DFA dependency.
        self.regions_of_interest = {
            name: np.asarray(boxes, dtype=self.float_dtype).copy()
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
            boxes = np.asarray(boxes, dtype=self.float_dtype)
            if boxes.size == 0:
                boxes = np.empty((0, 2, dim), dtype=self.float_dtype)
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

    def x2state(self, x):
        '''
        Return the state ID for a given point x in the continuous state space.

        :param x: Point in the continuous state space.
        :return: (state ID, exists). exists is False (and the ID is the missing-state sentinel) when x
            does not fall in a grid cell within the partition boundary.
        '''
        # Discard points outside of partition
        in_partition = np.all((x >= self.boundary_lb) * (x <= self.boundary_ub))
        if not in_partition:
            return self.size, False

        # Find the grid index per dimension by searching the grid edges
        grid_idx = []
        for d in range(self.dimension):
            grid_idx.append(np.searchsorted(self.edges[d], x[d], side='right') - 1)
        return self.grid_idx2state(grid_idx)

    def grid_idx2state(self, grid_idx):
        '''Map a unit-cube grid index tuple to its (state ID, exists).'''
        
        key = 0
        for value, num in zip(grid_idx, self.grid_shape):
            value = int(value)
            if not 0 <= value < num:
                return self.size, False
            key = key * num + value

        # Cells are stored in row-major order, so the key is the state ID.
        return key, True

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
        actions = action_array[None, :, :].astype(self.float_dtype)
        return centers_unit, actions


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