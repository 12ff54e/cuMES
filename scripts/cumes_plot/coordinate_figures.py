"""Six-panel PEST and Boozer magnetic-coordinate figures."""

import os
import time

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from .config import validate_figure_config
from .coordinates import PERIOD, interpolate_zeta, make_boozer_grid
from .output_paths import figure_path


def _surface_indices(count, wanted, start=0):
    if count < 1:
        raise ValueError("a coordinate plot needs at least one flux surface")
    start = min(max(start, 0), count - 1)
    return np.unique(np.rint(np.linspace(start, count - 1,
                                         min(wanted, count))).astype(int))


def _periodic_indices(count, wanted):
    """Select uniformly spaced samples without duplicating the periodic seam."""
    if count < 1:
        raise ValueError("a periodic coordinate needs at least one sample")
    selected = min(max(wanted, 1), count)
    return np.arange(selected, dtype=int) * count // selected


def _slice_angles(config):
    return PERIOD * np.arange(config.slice_panel_count) / \
        config.slice_panel_count


def _set_slice_limits(axes, grid, config):
    radial = _surface_indices(
        grid.s.size, config.slice_limit_surface_count)
    r_values = grid.r[radial]
    z_values = grid.z[radial]
    r_min, r_max = float(np.min(r_values)), float(np.max(r_values))
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))
    span = max(r_max - r_min, z_max - z_min)
    margin = config.slice_limit_margin_fraction * span if span > 0.0 \
        else config.slice_limit_zero_span_margin
    for ax in axes:
        ax.set_xlim(r_min - margin, r_max + margin)
        ax.set_ylim(z_min - margin, z_max + margin)
        ax.set_aspect(config.slice_aspect,
                      adjustable=config.slice_aspect_adjustable)


def _format_slice_axes(axes, angles, grid, nfp, config):
    for panel, (ax, toroidal_angle) in enumerate(zip(axes, angles)):
        physical_angle = np.degrees(toroidal_angle / nfp)
        symbol = r"\zeta_b" if grid.coordinate.lower() == "boozer" \
            else r"\varphi"
        ax.set_title(rf"${symbol}={physical_angle:g}^\circ$",
                     fontsize=config.panel_title_fontsize)
        ax.tick_params(labelsize=config.tick_label_fontsize)
        if panel % config.panel_columns == 0:
            ax.set_ylabel("Z (m)", fontsize=config.axis_label_fontsize)
        if panel >= (config.panel_rows - 1) * config.panel_columns:
            ax.set_xlabel("R (m)", fontsize=config.axis_label_fontsize)


def _save(fig, path, config):
    fig.savefig(path, dpi=config.save_dpi, bbox_inches=config.save_bbox,
                facecolor=config.save_face_color)
    plt.close(fig)
    print(f"saved {path}", flush=True)


def _fill_axis_hole(ax, r, z, b, cmap, norm, config):
    """Fill the region inside the first non-axis surface smoothly.

    The exported magnetic-coordinate grids start at the first non-axis flux
    surface.  Its poloidal averages estimate the degenerate magnetic-axis
    point and the single-valued axis field.  A triangular fan then interpolates
    those axis values to the innermost surface without adding a fictitious
    finite-area surface to the data model.
    """
    inner_r = r[0]
    inner_z = z[0]
    inner_b = b[0]
    count = inner_r.size
    boundary = np.arange(1, count + 1)
    triangles = np.column_stack((
        np.zeros(count, dtype=int),
        boundary,
        np.roll(boundary, -1),
    ))
    return ax.tripcolor(
        np.concatenate(([np.mean(inner_r)], inner_r)),
        np.concatenate(([np.mean(inner_z)], inner_z)),
        triangles,
        np.concatenate(([np.mean(inner_b)], inner_b)),
        cmap=cmap,
        norm=norm,
        shading=config.field_shading,
        rasterized=config.field_rasterized,
    )


def render_coordinate_slices(base, grid, nfp, title, config):
    """Six toroidal R-Z cuts with flux and poloidal coordinate lines."""
    print(f"rendering {grid.coordinate} coordinate slices ...", flush=True)
    angles = _slice_angles(config)
    fig, axes = plt.subplots(
        config.panel_rows, config.panel_columns,
        figsize=config.coordinate_figure_size,
        sharex=config.share_panel_x, sharey=config.share_panel_y,
        constrained_layout=config.use_constrained_layout)
    axes = axes.ravel()
    radial = _surface_indices(
        grid.s.size, config.coordinate_radial_line_count)
    poloidal = _periodic_indices(
        grid.theta.size, config.coordinate_theta_line_count)
    for ax, zeta in zip(axes, angles):
        r = interpolate_zeta(grid.r, zeta)
        z = interpolate_zeta(grid.z, zeta)
        for index in radial:
            ax.plot(np.append(r[index], r[index, 0]),
                    np.append(z[index], z[index, 0]),
                    color=config.coordinate_flux_color,
                    linewidth=config.coordinate_edge_flux_linewidth
                    if index == radial[-1]
                    else config.coordinate_inner_flux_linewidth)
        for index in poloidal:
            ax.plot(r[:, index], z[:, index],
                    color=config.coordinate_theta_color,
                    linewidth=config.coordinate_theta_linewidth,
                    alpha=config.coordinate_theta_alpha)
    _set_slice_limits(axes, grid, config)
    _format_slice_axes(axes, angles, grid, nfp, config)
    fig.suptitle(f"{title} — {grid.coordinate} coordinate mesh",
                 fontsize=config.figure_title_fontsize)
    path = figure_path(base, f"{grid.coordinate.lower()}_coordinate_slices")
    _save(fig, path, config)
    return path


def render_field_slices(base, grid, nfp, title, norm, cmap, config):
    """Six physical-space R-Z cuts colored by magnetic-field magnitude."""
    print("rendering physical |B| slices ...", flush=True)
    angles = _slice_angles(config)
    fig, axes = plt.subplots(
        config.panel_rows, config.panel_columns,
        figsize=config.field_slice_figure_size,
        sharex=config.share_panel_x, sharey=config.share_panel_y,
        constrained_layout=config.use_constrained_layout)
    axes = axes.ravel()
    artist = None
    for ax, zeta in zip(axes, angles):
        r = interpolate_zeta(grid.r, zeta)
        z = interpolate_zeta(grid.z, zeta)
        b = interpolate_zeta(grid.b, zeta)
        _fill_axis_hole(ax, r, z, b, cmap, norm, config)
        r = np.concatenate([r, r[:, :1]], axis=1)
        z = np.concatenate([z, z[:, :1]], axis=1)
        b = np.concatenate([b, b[:, :1]], axis=1)
        artist = ax.pcolormesh(r, z, b, cmap=cmap, norm=norm,
                               shading=config.field_shading,
                               rasterized=config.field_rasterized)
    _set_slice_limits(axes, grid, config)
    _format_slice_axes(axes, angles, grid, nfp, config)
    cbar = fig.colorbar(artist, ax=axes.tolist(),
                        shrink=config.colorbar_shrink,
                        pad=config.colorbar_pad)
    cbar.set_label(config.colorbar_label)
    fig.suptitle(f"{title} — magnetic-field slices",
                 fontsize=config.figure_title_fontsize)
    path = figure_path(base, "field_slices")
    _save(fig, path, config)
    return path


def render_field_contours(base, grid, title, norm, cmap, config):
    """Six radial flux surfaces as |B|(poloidal angle, toroidal angle)."""
    print(f"rendering {grid.coordinate} flux-surface |B| contours ...",
          flush=True)
    fig, axes = plt.subplots(
        config.panel_rows, config.panel_columns,
        figsize=config.field_contour_figure_size,
        sharex=config.share_panel_x, sharey=config.share_panel_y,
        constrained_layout=config.use_constrained_layout)
    axes = axes.ravel()
    # Avoid packing several panels into the tiny near-axis interval when many
    # surfaces are available, while retaining the innermost exported surface
    # for small manufactured/test grids.
    start = int(round(
        config.field_contour_inner_fraction * (grid.s.size - 1)))
    radial = _surface_indices(
        grid.s.size, config.field_contour_surface_count, start=start)
    theta = np.append(grid.theta, PERIOD) / PERIOD
    zeta = np.append(grid.zeta, PERIOD) / PERIOD
    levels = np.linspace(
        norm.vmin, norm.vmax, config.field_contour_level_count)
    artist = None
    for ax, surface in zip(axes, radial):
        b = grid.b[surface]
        b = np.concatenate([b, b[:1, :]], axis=0)
        b = np.concatenate([b, b[:, :1]], axis=1)
        artist = ax.contourf(theta, zeta, b, levels=levels, cmap=cmap,
                             norm=norm, extend=config.field_contour_extend)
        ax.contour(theta, zeta, b,
                   levels=levels[::config.field_contour_line_stride],
                   colors=config.field_contour_line_color,
                   linewidths=config.field_contour_linewidth,
                   alpha=config.field_contour_line_alpha)
        ax.set_title(f"s = {grid.s[surface]:.3f}",
                     fontsize=config.panel_title_fontsize)
        ax.tick_params(labelsize=config.tick_label_fontsize)
    for ax in axes[len(radial):]:
        ax.set_visible(False)
    for panel, ax in enumerate(axes):
        if panel % config.panel_columns == 0:
            ax.set_ylabel(grid.toroidal_label + r" $/\ 2\pi$",
                          fontsize=config.axis_label_fontsize)
        if panel >= (config.panel_rows - 1) * config.panel_columns:
            ax.set_xlabel(grid.angle_label + r" $/\ 2\pi$",
                          fontsize=config.axis_label_fontsize)
        ax.set_xlim(*config.normalized_angle_limits)
        ax.set_ylim(*config.normalized_angle_limits)
    cbar = fig.colorbar(artist, ax=axes.tolist(),
                        shrink=config.colorbar_shrink,
                        pad=config.colorbar_pad)
    cbar.set_label(config.colorbar_label)
    fig.suptitle(f"{title} — {grid.coordinate} flux-surface |B| contours",
                 fontsize=config.figure_title_fontsize)
    path = figure_path(base, f"{grid.coordinate.lower()}_field_contours")
    _save(fig, path, config)
    return path


def render_coordinate_grids(grids, output_path, nfp, title_name, config):
    """Render coordinate-specific plots and one physical field-slice plot."""
    base = os.path.splitext(output_path)[0]
    config = validate_figure_config(config)
    title = f"{config.figure_title_prefix} — {title_name}"
    if not grids:
        raise ValueError("at least one coordinate grid is required")
    bmin = min(float(np.min(grid.b)) for grid in grids)
    bmax = max(float(np.max(grid.b)) for grid in grids)
    if bmax <= bmin:
        bmax = bmin + max(abs(bmin), 1.0) * np.finfo(float).eps
    norm = mcolors.Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap(config.field_colormap)
    paths = []
    started = time.perf_counter()
    for grid in grids:
        paths.append(render_coordinate_slices(
            base, grid, nfp, title, config))
        paths.append(render_field_contours(
            base, grid, title, norm, cmap, config))
    # R-Z cuts live in physical space and therefore do not have distinct PEST
    # and Boozer variants. Prefer the native-state PEST reconstruction when it
    # is available; its geometry has not undergone Boozer spectral truncation.
    field_grid = next(
        (grid for grid in grids if grid.coordinate.lower() == "pest"),
        grids[0],
    )
    paths.append(render_field_slices(
        base, field_grid, nfp, title, norm, cmap, config))
    print(f"coordinate figures completed in {time.perf_counter() - started:.1f}s",
          flush=True)
    return paths


def render_coordinate_figures(data, output_path, config):
    """Convenience wrapper for the Boozer figures from a Boozer container."""
    # The public field grid may intentionally be compact (for example 16x36
    # for a low-mode equilibrium). Fourier synthesis plus periodic linear
    # interpolation give publication-quality curves without inventing modes.
    config = validate_figure_config(config)
    grid = make_boozer_grid(
        data, max(config.min_render_theta, data.ntheta),
        max(config.min_render_zeta, data.nzeta))
    title_name = os.path.splitext(os.path.basename(
        data.source_path or output_path))[0]
    return render_coordinate_grids(
        [grid], output_path, data.nfp, title_name, config)
