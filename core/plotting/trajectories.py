#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from matplotlib.patches import Rectangle
from pathlib import Path

from core.utils import cm2inch
from core.plotting.utils import PNG_DPI, plot_boxes, plot_grid, set_plot_lims, set_plot_ticks

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Whitespace (in inches) kept around the tight bounding box of saved trajectory plots
TIGHT_PAD = 0.01


def _format_state_label_math(var_name):
    if '_' in var_name:
        head, tail = var_name.split('_', 1)
        return f'${head}_{{{tail}}}$'
    return f'${var_name}$'


def _plot_labeled_regions_2d(ax, model, dimensions, alpha=0.24):
    """Plot ROI rectangles with colored faces and a legend of proposition labels."""
    from matplotlib.patches import Patch

    if not hasattr(model, "regions"):
        return
    i1, i2 = map(int, dimensions)
    cmap = plt.get_cmap("tab10")
    handles = []
    for j, (name, boxes) in enumerate(model.regions.items()):
        boxes = np.asarray(boxes, dtype=float)
        if boxes.size == 0:
            continue
        color = cmap(j % 10)
        propositions = (model.labeling_function(name)
                        if hasattr(model, "labeling_function")
                        else getattr(model, "labels", {}).get(name, ()))
        label = ", ".join(sorted(propositions)) or name
        for box in boxes:
            lo, hi = box
            ax.add_patch(Rectangle(
                (lo[i1], lo[i2]), hi[i1] - lo[i1], hi[i2] - lo[i2],
                facecolor=color, edgecolor=color, alpha=alpha,
                linewidth=1.4, zorder=1))
        handles.append(Patch(facecolor=color, edgecolor=color,
                             alpha=alpha, label=label))
    if handles:
        old_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=old_handles + handles, loc="best", fontsize=8)


def _save_figure(args, stamp, filename):
    """Write the current figure to the run output directory, as both PDF and PNG."""
    output_dir = Path(getattr(args, 'output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{filename}_{stamp}.pdf', format='pdf', bbox_inches='tight', pad_inches=TIGHT_PAD)
    plt.savefig(output_dir / f'{filename}_{stamp}.png', format='png', bbox_inches='tight', pad_inches=TIGHT_PAD, dpi=PNG_DPI)


def _plot_trajectories_1d(args, stamp, i1, partition, model, trajectories, num_trajectories=10, filename="trajectories", show_ticks=None):
    """Plot trajectories of a 1D model as the state against the time step (there is no second state axis)."""
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5), dpi=300)
    font = {'size': 10}
    mpl.rc('font', **font)
    ax.set_xlabel('$k$', labelpad=2)
    ax.set_ylabel(_format_state_label_math(model.state_variables[i1]), labelpad=2)

# Goal and unsafe sets become intervals on the state axis.
    for boxes, color in ((model.goal, 'green'), (model.critical, 'red')):
        for box in boxes:
            ax.axhspan(float(box[0][i1]), float(box[1][i1]), facecolor=color, alpha=0.3, edgecolor=color)

    ax.set_ylim(float(partition.boundary_lb[i1]), float(partition.boundary_ub[i1]))

# Add trajectories
    max_k = 1
    i = 0
    for trajectory in trajectories.values():
        states = np.array(trajectory['x'])[:, i1]

# Only show trajectory if there are at least two points
        if len(states) < 2:
            continue
        else:
            i += 1

# Stop at desired number of trajectories
        if i > num_trajectories:
            break

        steps = np.arange(len(states))
        max_k = max(max_k, int(steps[-1]))
        ax.plot(steps, states, '-o', color="blue", markersize=1, linewidth=1, alpha=0.6)

    ax.set_xlim(0, max_k)

# show_ticks overrides args.plot_ticks when explicitly set
    _show_ticks = args.plot_ticks if show_ticks is None else show_ticks

    if _show_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3g}'))
        ax.tick_params(axis='both', which='major', labelsize=6, direction='in', length=3, top=False, right=False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

# Set tight layout
    fig.tight_layout()

    if args.plot_title:
        ax.set_title(f"Simulation for {args.model}")

    _save_figure(args, stamp, filename)
    plt.close(fig)


def plot_trajectories(args, stamp, idx_show, partition, model, trajectories, line=True, num_trajectories=10, add_unsafe_box=True, filename="trajectories", show_ticks=None, labeled_regions=False):
    idx = np.atleast_1d(np.asarray(idx_show, dtype=int))
    if idx.size == 1:
        return _plot_trajectories_1d(args, stamp, int(idx[0]), partition, model, trajectories,
                               num_trajectories=num_trajectories, filename=filename, show_ticks=show_ticks)
    if idx.size != 2:
        raise ValueError(f'Expected one or two plot dimensions to show, got {idx.size}: {idx_show}.')
    i1, i2 = int(idx[0]), int(idx[1])
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5), dpi=300)
    font = {'size': 10}
    mpl.rc('font', **font)
    plt.xlabel(_format_state_label_math(model.state_variables[i1]), labelpad=2)
    plt.ylabel(_format_state_label_math(model.state_variables[i2]), labelpad=2)

    if add_unsafe_box:
        expand = 1
    else:
        expand = 0

# show_ticks overrides args.plot_ticks when explicitly set
    _show_ticks = args.plot_ticks if show_ticks is None else show_ticks

    if _show_ticks == 'nice' or show_ticks is True:
# Keep a small number of readable ticks and avoid scientific-offset labels.
        formatter = FuncFormatter(lambda x, pos: f'{x:.3g}')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', which='major', labelsize=6, direction='in', length=3, top=False, right=False)
    elif _show_ticks:
        set_plot_ticks(ax,
                  state_min=np.array(partition.boundary_lb)[[i1, i2]] - expand,
                  state_max=np.array(partition.boundary_ub)[[i1, i2]] + expand,
                  width=np.array(partition.cell_width)[[i1, i2]])
        formatter = FuncFormatter(lambda x, pos: f'{x:.3g}')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', which='major', labelsize=6, direction='in', length=3, top=False, right=False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    set_plot_lims(ax,
                  state_min=np.array(partition.boundary_lb)[[i1, i2]] - expand,
                  state_max=np.array(partition.boundary_ub)[[i1, i2]] + expand)

# Plot grid
    if args.plot_grid:
        plot_grid(ax,
                state_min=np.array(partition.boundary_lb)[[i1, i2]],
                state_max=np.array(partition.boundary_ub)[[i1, i2]],
                size=[1, 1])

# Plot goal/unsafe regions
    if labeled_regions:
        _plot_labeled_regions_2d(ax, model, (i1, i2))
    else:
        plot_boxes(ax, model, plot_dimensions=[i1, i2])

# Plot boundary of unsafe regions if requested
    if add_unsafe_box:
        state_lb = np.array(partition.boundary_lb)
        state_ub = np.array(partition.boundary_ub)
        LOWS = [np.array([state_lb[i1] - expand, state_lb[i2] - expand]),
                np.array([state_lb[i1], state_lb[i2] - expand]),
                np.array([state_lb[i1], state_ub[i2]]),
                np.array([state_ub[i1], state_lb[i2] - expand])
                ]
        HIGHS = [np.array([state_lb[i1], state_ub[i2] + expand]),
                 np.array([state_ub[i1], state_lb[i2]]),
                 np.array([state_ub[i1], state_ub[i2] + expand]),
                 np.array([state_ub[i1] + expand, state_ub[i2] + expand]),
                 ]

        for low, high in zip(LOWS, HIGHS):
            width, height = (high - low)
            ax.add_patch(Rectangle(low, width, height, facecolor='red', alpha=0.3, edgecolor='red'))

# Add trajectories
    i = 0
    for trajectory in trajectories.values():
        states = np.array(trajectory['x'])[:, [i1, i2]]

# Only show trajectory if there are at least two points
        if len(states) < 2:
            continue
        else:
            i += 1

# Stop at desired number of trajectories
        if i > num_trajectories:
            break

# Connect consecutive states with a thin line
        if line:
            plt.plot(*states.T, '-', color="black", linewidth=0.3, alpha=0.6)

# Plot precise points
        plt.plot(*states.T, 'o', markersize=1, color="black");

# plt.gca().set_aspect('equal')
# Set tight layout
    fig.tight_layout()

    if args.plot_title:
        ax.set_title(f"Simulation for {args.model}")

# Save figure
    _save_figure(args, stamp, filename)
    plt.close(fig)
