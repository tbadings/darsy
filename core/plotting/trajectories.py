#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


def _cuboid_faces(low, high):
    x0, y0, z0 = low
    x1, y1, z1 = high

    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _plot_cuboid(ax, low, high, facecolor, edgecolor, alpha):
    faces = _cuboid_faces(low, high)
    collection = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=0.4,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


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


def _plot_labeled_regions_3d(ax, model, dimensions, alpha=0.24):
    """Plot ROI cuboids and proposition labels in the 3D plot legend."""
    from matplotlib.patches import Patch

    if not hasattr(model, "regions"):
        return
    dims = list(map(int, dimensions))
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
            _plot_cuboid(ax, lo[dims], hi[dims], facecolor=color,
                         edgecolor=color, alpha=alpha)
        handles.append(Patch(facecolor=color, edgecolor=color,
                             alpha=alpha, label=label))
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8)


def plot_trajectories_3d(args, stamp, idx_show, partition, model, trajectories, num_trajectories=10, filename="trajectories_3d", show_ticks=None, camera_angles=None, labeled_regions=False):
    """Plot 3D trajectories.

    New params:
    - grid_linewidth: float or None -> line width for grid (very thin if None defaults)
    - camera_angles: list of (elev, azim) tuples. If provided, saves one file per angle.
    """
    fig = plt.figure(figsize=cm2inch(7.5, 6.5), dpi=1000)
    ax = fig.add_subplot(111, projection='3d')

    font = {'size': 10}
    mpl.rc('font', **font)

    i1, i2, i3 = np.array(idx_show, dtype=int)

# ax.set_xlabel(_format_state_label_math(model.state_variables[i1]), labelpad=4)
# ax.set_ylabel(_format_state_label_math(model.state_variables[i2]), labelpad=4)
# ax.set_zlabel(_format_state_label_math(model.state_variables[i3]), labelpad=4)

    state_lb = np.array(partition.boundary_lb)[[i1, i2, i3]]
    state_ub = np.array(partition.boundary_ub)[[i1, i2, i3]]

    if labeled_regions:
        _plot_labeled_regions_3d(ax, model, (i1, i2, i3))
    else:
        for set_ in model.goal:
            low = np.array(set_[0])[[i1, i2, i3]]
            high = np.array(set_[1])[[i1, i2, i3]]
            _plot_cuboid(ax, low, high, facecolor='green', edgecolor='none', alpha=0.48)

        for set_ in model.critical:
            low = np.array(set_[0])[[i1, i2, i3]]
            high = np.array(set_[1])[[i1, i2, i3]]
            _plot_cuboid(ax, low, high, facecolor='red', edgecolor='none', alpha=0.44)

    for trajectory in trajectories.values():
        states = np.array(trajectory['x'])[:, [i1, i2, i3]]
        ax.plot(states[:, 0], states[:, 1], states[:, 2], '-', color='black', linewidth=0.3, alpha=0.6)
        ax.scatter(states[:, 0], states[:, 1], states[:, 2], s=1, color='black', edgecolors='none', alpha=0.7)

    ax.set_xlim(state_lb[0], state_ub[0])
    ax.set_ylim(state_lb[1], state_ub[1])
    ax.set_zlim(state_lb[2], state_ub[2])
    try:
        ax.set_box_aspect((state_ub - state_lb).tolist())
    except Exception:
        pass

    widths = np.array(partition.cell_width)[[i1, i2, i3]]
    lowers = np.array(partition.boundary_lb)[[i1, i2, i3]]
    uppers = np.array(partition.boundary_ub)[[i1, i2, i3]]
    tick_sets = []
    for lower, upper, width in zip(lowers, uppers, widths):
        ticks = np.arange(lower, upper + width, width)
        tick_sets.append(ticks)

    formatter = FuncFormatter(lambda value, pos: f'{value:.3g}')
    ax.set_xticks(tick_sets[0])
    ax.set_yticks(tick_sets[1])
    ax.set_zticks(tick_sets[2])
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', which='major', labelsize=0, direction='in', length=3, pad=0)

# Use a very thin default grid line width unless overridden
    ax.grid(True, linestyle='--', linewidth=1, opacity=0.5)

# Default to one sensible view if no camera angles provided
    if camera_angles is None:
        camera_angles = [(60, 15)]
    if args.plot_title:
        ax.set_title(f"Simulation for {args.model}")

    output_dir = Path(getattr(args, 'output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)

# Save one file per camera angle
    for vi, (elev, azim) in enumerate(camera_angles):
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()

        pdf_name = output_dir / f'{filename}_{stamp}_view{vi}_{stamp}.pdf'
        png_name = output_dir / f'{filename}_{stamp}_view{vi}_{stamp}.png'
        plt.savefig(pdf_name, format='pdf', bbox_inches='tight', pad_inches=TIGHT_PAD)
        plt.savefig(png_name, format='png', bbox_inches='tight', pad_inches=TIGHT_PAD, dpi=max(PNG_DPI, fig.dpi))

    plt.close(fig)


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
