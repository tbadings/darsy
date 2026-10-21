import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.transforms import Affine2D
from pathlib import Path


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

    X = np.asarray(partition.regions_per_dim['centers'][i1])
    Y = (np.asarray(partition.regions_per_dim['centers'][i2])
         if i2 is not None else np.zeros(1))
    DF = pd.DataFrame(values[::-1, :], index=Y[::-1], columns=X)

    fig, ax = plt.subplots()
    try:
        sns.heatmap(DF, xticklabels=False, yticklabels=False, ax=ax)
        missing_rows, missing_cols = np.where(np.isnan(DF.to_numpy()))
        for row, col in zip(missing_rows, missing_cols):
            ax.plot([col, col + 1], [row, row + 1], color='red',
                    linewidth=0.4, alpha=1.0, zorder=5)
            ax.plot([col, col + 1], [row + 1, row], color='red',
                    linewidth=0.4, alpha=1.0, zorder=5)

        if args.plot_ticks:
            fmt = lambda v: f'{v:.4g}'
            x_idx = np.unique(np.linspace(0, len(X)-1, min(6, len(X)), dtype=int))
            y_idx = np.unique(np.linspace(0, len(Y)-1, min(6, len(Y)), dtype=int))
            ax.set_xticks(x_idx + 0.5)
            ax.set_xticklabels([fmt(X[i]) for i in x_idx])
            if i2 is None:
                ax.set_yticks([])
            else:
                ax.set_yticks(y_idx + 0.5)
                ax.set_yticklabels([fmt(Y[::-1][i]) for i in y_idx])
            ax.tick_params(axis='x', labelsize=8, rotation=0)
            ax.tick_params(axis='y', labelsize=8, rotation=0)

        if args.plot_title:
            ax.set_title(f'Heatmap for {args.model} ({filename})')

        # Seaborn draws in cell-index coordinates, not physical coordinates.
        # x: [physical_lb, physical_ub] -> [0, nx]
        # y: [physical_lb, physical_ub] -> [ny, 0] (reversed rows).
        if i2 is not None:
            lb = np.asarray(partition.boundary_lb, dtype=float)
            ub = np.asarray(partition.boundary_ub, dtype=float)
            sx = nx / (ub[i1] - lb[i1])
            sy = ny / (ub[i2] - lb[i2])
            ax.roi_physical_transform = (
                Affine2D().translate(-lb[i1], -ub[i2]).scale(sx, -sy)
                + ax.transData
            )

        if overlay is not None:
            if i2 is None:
                raise ValueError('Rectangle overlays require two plot dimensions')
            overlay(ax)

        output_dir = Path(getattr(args, 'output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(output_dir / f'{filename}_{stamp}.png', format='png', bbox_inches='tight')
    finally:
        plt.close(fig)