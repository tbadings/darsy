"""Build a nonuniform Cartesian partition from regions of interest and find FRS intersections."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import LineCollection, PolyCollection


def partition_from_regions(regions, domain):
    """Build grid edges from ROI rectangles and the domain.

    regions: (K, 2, 2), each entry [[x_min,y_min], [x_max,y_max]].
    domain: (2, 2), same convention.
    All ROI edges must lie inside the domain.
    """
    regions = np.asarray(regions, dtype=float)
    domain = np.asarray(domain, dtype=float)
    if regions.ndim != 3 or regions.shape[1:] != (2, 2):
        raise ValueError('regions must have shape (K, 2, 2)')
    if domain.shape != (2, 2) or np.any(domain[0] >= domain[1]):
        raise ValueError('domain must be [[x_min,y_min], [x_max,y_max]]')
    if np.any(regions[:, 0] >= regions[:, 1]):
        raise ValueError('ROI lower bounds must be below upper bounds')
    if np.any(regions[:, 0] < domain[0]) or np.any(regions[:, 1] > domain[1]):
        raise ValueError('ROI rectangles must be contained in domain')
    return tuple(np.unique(np.r_[domain[:, d], regions[:, :, d].ravel()]) for d in range(2))


def subdivide_partition(boundaries, max_side_length):
    """Refine each initial grid interval until every side is strictly < maximum.

    Original ROI-induced edges are preserved. max_side_length can be a scalar
    or a two-element sequence specifying the maximum per dimension.
    """
    limits = np.broadcast_to(np.asarray(max_side_length, dtype=float), (2,))
    if np.any(~np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError('max_side_length must be finite and positive')

    refined = []
    for edges, limit in zip(boundaries, limits):
        pieces = []
        for left, right in zip(edges[:-1], edges[1:]):
            # Strict inequality: intervals exactly equal to the limit split too.
            count = int(np.floor((right - left) / limit)) + 1
            pieces.append(np.linspace(left, right, count + 1)[:-1])
        refined.append(np.r_[*pieces, edges[-1]])
    return tuple(refined)


def intersecting_cells(boundaries, box_lb, box_ub):
    """Indices of grid cells with positive-area intersection with a box."""
    ranges = []
    for edges, lo, hi in zip(boundaries, box_lb, box_ub):
        if lo >= hi or hi <= edges[0] or lo >= edges[-1]:
            return np.empty((0, 2), dtype=int)
        first = max(0, np.searchsorted(edges, lo, side='right') - 1)
        last = min(len(edges) - 2, np.searchsorted(edges, hi, side='left') - 1)
        if first > last:
            return np.empty((0, 2), dtype=int)
        ranges.append(np.arange(first, last + 1))
    return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(2, -1).T


def main():
    # 1. Specify the regions of interest through their lower/upper corners.
    #    Their edges (plus the domain edges) DEFINE the nonuniform partition.
    domain = np.array([[-5.0, -4.0], [5.0, 4.0]])
    regions_of_interest = np.array([
        [[-3.5, -2.5], [-0.8,  1.2]],
        [[ 0.0, -1.0], [ 3.0,  2.8]],
        [[-2.0,  0.5], [ 1.7,  2.0]],
    ])
    # First construct the partition using only ROI and domain edges.
    initial_boundaries = partition_from_regions(regions_of_interest, domain)

    # Then subdivide each interval without moving any original ROI edge.
    max_side_length = np.array([0.3, 0.3])
    boundaries = subdivide_partition(initial_boundaries, max_side_length)
    x_edges, y_edges = boundaries

    # 2. Example rectangular overapproximation of a forward reachable set.
    frs_lb = np.array([-2.6, -1.6])
    frs_ub = np.array([2.3, 2.2])

    # 3. Find intersecting grid cells without enumerating the entire partition.
    selected = intersecting_cells(boundaries, frs_lb, frs_ub)
    print('Initial partition:', tuple(len(b)-1 for b in initial_boundaries))
    print('Maximum side lengths:', max_side_length)
    print(f'Refined partition: {(len(x_edges)-1)} x {(len(y_edges)-1)} cells')
    assert all(np.all(np.diff(b) < limit) for b, limit in zip(boundaries, max_side_length))
    print(f'FRS-intersecting cells: {len(selected)}')

    # Draw filled cells and grid lines in batches for fast plotting.
    fig, ax = plt.subplots(figsize=(10, 7))
    if len(selected):
        i, j = selected.T
        xl, xr = x_edges[i], x_edges[i + 1]
        yl, yr = y_edges[j], y_edges[j + 1]
        vertices = np.stack([
            np.stack([xl, yl], axis=-1),
            np.stack([xr, yl], axis=-1),
            np.stack([xr, yr], axis=-1),
            np.stack([xl, yr], axis=-1),
        ], axis=1)
        ax.add_collection(PolyCollection(vertices, facecolors='lightskyblue', edgecolors='none'))

    segments = ([((x, y_edges[0]), (x, y_edges[-1])) for x in x_edges]
                + [((x_edges[0], y), (x_edges[-1], y)) for y in y_edges])
    ax.add_collection(LineCollection(segments, colors='0.7', linewidths=0.45, rasterized=True))

    for roi in regions_of_interest:
        ax.add_patch(Rectangle(roi[0], *(roi[1]-roi[0]), fill=False,
                               edgecolor='darkorange', linewidth=2, linestyle=':'))
    ax.add_patch(Rectangle(frs_lb, *(frs_ub-frs_lb), fill=False,
                           edgecolor='crimson', linewidth=2.7, linestyle='--'))
    ax.set(xlabel='$x_1$', ylabel='$x_2$',
           title='ROI-induced partition refined by maximum cell side length')
    ax.set_xlim(domain[0, 0] - .3, domain[1, 0] + .3)
    ax.set_ylim(domain[0, 1] - .3, domain[1, 1] + .3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(handles=[
        Patch(facecolor='lightskyblue', label='FRS-intersecting cells'),
        Patch(facecolor='none', edgecolor='darkorange', linestyle=':', label='Regions of interest'),
        Patch(facecolor='none', edgecolor='crimson', linestyle='--', label='FRS bounding box'),
    ], loc='upper left')
    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / 'nonuniform_frs_fixed.png'
    fig.savefig(output_path, dpi=180)
    print(f'Figure saved to: {output_path}')
    plt.show()


if __name__ == '__main__':
    main()