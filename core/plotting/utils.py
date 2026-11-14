from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from core.interfaces import Benchmark, FloatArray, IntArray

# Resolution of all saved PNG figures (PDFs are vector graphics and unaffected)
PNG_DPI = 600

def plot_boxes(
    ax: Axes, model: Benchmark, plot_dimensions: list[int] | IntArray | None = None,
    labels: bool = False, latex: bool = False, size: float = 12,
) -> None:
    """Add goal/unsafe boxes to ax, selecting dimensions [0, 1] by default.

    Labels use size + 4 points and mathematical notation when latex=True.
    """
    if plot_dimensions is None:
        plot_dimensions = [0, 1]
    label_size = size + 4

    # Plot target set
    for box in model.goal:
        low = box[0]
        high = box[1]
        width, height = (high - low)[plot_dimensions]
        ax.add_patch(Rectangle(low[plot_dimensions], width, height, facecolor='green', alpha=0.3, edgecolor='green'))
        midpoint = (high + low)[plot_dimensions] / 2
        if labels:
            if latex:
                text = r'$\mathcal{X}_T$'
            else:
                text = 'X_T'
            ax.annotate(text, midpoint, color='green', fontsize=label_size, ha='center', va='center')

    # Plot unsafe set
    for box in model.critical:
        low = box[0]
        high = box[1]
        width, height = (high - low)[plot_dimensions]
        ax.add_patch(Rectangle(low[plot_dimensions], width, height, facecolor='red', alpha=0.3, edgecolor='red'))
        midpoint = (high + low)[plot_dimensions] / 2
        if labels:
            if latex:
                text = r'$\mathcal{X}_U$'
            else:
                text = 'X_U'
            ax.annotate(text, midpoint, color='red', fontsize=label_size, ha='center', va='center')

    return


def plot_grid(
    ax: Axes, state_min: FloatArray, state_max: FloatArray,
    size: Sequence[float] | FloatArray | None = None,
) -> None:
    """Add grid lines to ax using (2,) bounds and spacing [1, 1] by default."""
    if size is None:
        size = [1, 1]
    min_floor = state_min // np.array(size)
    max_ceil = state_max // np.array(size) + 1
    x_positions = np.arange(min_floor[0], max_ceil[0], size[0])
    y_positions = np.arange(min_floor[1], max_ceil[1], size[1])

    # Vertical lines
    ax.hlines(y_positions, xmin=state_min[0] - 1, xmax=state_max[0] + 1, color='gray', linewidth=0.5)
    ax.vlines(x_positions, ymin=state_min[1] - 1, ymax=state_max[1] + 1, color='gray', linewidth=0.5)


def set_plot_ticks(ax: Axes, state_min: FloatArray, state_max: FloatArray, width: FloatArray) -> None:
    regions = np.round((state_max - state_min) / width)

    # Plot at most x ticks per axis, but at least skip every 4 regions
    skip_x = max(1, regions[0] // 5, 4)
    skip_y = max(1, regions[1] // 10, 4)
    major_ticks_x = np.arange(state_min[0] + 0.5 * width[0], state_max[0] - 0.5 * width[0], skip_x * width[0])
    major_ticks_y = np.arange(state_min[1] + 0.5 * width[1], state_max[1] - 0.5 * width[1], skip_y * width[1])

    # minor_ticks_x = np.arange(state_min[0] + 0.5 * width[0], state_max[0] - 0.5 * width[0], width[0])
    # minor_ticks_y = np.arange(state_min[1] + 0.5 * width[1], state_max[1] - 0.5 * width[1], width[1])
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    # ax.set_xticks(minor_ticks_x, minor=True)
    # ax.set_yticks(minor_ticks_y, minor=True)
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)

    # plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)
def set_plot_lims(ax: Axes, state_min: FloatArray, state_max: FloatArray) -> None:
    # Goal x-y limits
    ax.set_xlim(state_min[0], state_max[0])
    ax.set_ylim(state_min[1], state_max[1])
