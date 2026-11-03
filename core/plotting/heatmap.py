import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from pathlib import Path

from core.plotting.utils import PNG_DPI


def heatmap(args, stamp, idx_show, slice_values=None, partition=None,
            results=None, filename='heatmap', overlay=None, **kwargs):
    """Plot a partition heatmap, optionally overlaying physical-coordinate geometry.

    `overlay(ax)` is called before saving. The callback can use
    `ax.roi_physical_transform` as the transform for patches specified in
    physical state coordinates (e.g. Rectangle(..., transform=...)).
    """
    mpl.rc('font', size=10)
    sns.set_style('white', {
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })

    idx = np.atleast_1d(np.asarray(idx_show, dtype=int))
    if idx.size == 1:
        i1, i2 = int(idx[0]), None
    elif idx.size == 2:
        i1, i2 = int(idx[0]), int(idx[1])
    else:
        raise ValueError(f'Expected one or two plot dimensions, got {idx_show}.')

    nx = int(partition.number_per_dim[i1])
    ny = int(partition.number_per_dim[i2]) if i2 is not None else 1
    idxs = np.asarray(partition.state_id2idx, dtype=int)
    num_states = len(idxs)
    res = np.asarray(results)[:num_states]
    if res.ndim != 1:
        raise ValueError('heatmap requires one scalar value per partition cell')

    sum_vals = np.zeros((ny, nx), dtype=float)
    counts = np.zeros((ny, nx), dtype=int)
    valid = ~np.isnan(res)
    x_coords = idxs[valid, i1]
    y_coords = (idxs[valid, i2] if i2 is not None
                else np.zeros(np.count_nonzero(valid), dtype=int))
    np.add.at(sum_vals, (y_coords, x_coords), res[valid].astype(float))
    np.add.at(counts, (y_coords, x_coords), 1)
    values = np.full((ny, nx), np.nan, dtype=float)
    mask = counts > 0
    values[mask] = sum_vals[mask] / counts[mask]

    # Draw in physical coordinates to keep the size of the cells
    x_edges = np.asarray(partition.edges[i1])
    if i2 is not None:
        y_edges = np.asarray(partition.edges[i2])
    else:
        y_edges = np.array([0.0, 1.0])

    fig, ax = plt.subplots()
    try:
        mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values), cmap=sns.cm.rocket)
        fig.colorbar(mesh, ax=ax)

        if not args.plot_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if i2 is None:
            ax.set_yticks([])
        if args.plot_title:
            ax.set_title(f'Heatmap for {args.model} ({filename})')

        ax.roi_physical_transform = ax.transData
        if overlay is not None:
            if i2 is None:
                raise ValueError('Rectangle overlays require two plot dimensions')
            overlay(ax)

        output_dir = Path(getattr(args, 'output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(output_dir / f'{filename}_{stamp}.png', format='png', bbox_inches='tight', dpi=PNG_DPI)
    finally:
        plt.close(fig)
