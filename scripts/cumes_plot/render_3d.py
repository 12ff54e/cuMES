"""Matplotlib and PyVista renderers for equilibrium overview figures."""

import time

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .coils import tube_mesh
from .equilibrium import periodic_close, surface_intensity, to_cartesian
from .image_processing import trim_white
from .output_paths import figure_path


def _render_machinery(S):
    """Per-process color machinery (each render process builds its own —
    matplotlib state is not shared across processes)."""
    config = S["figure_parameters"]
    cmap = plt.get_cmap(config.field_colormap)
    norm = mcolors.Normalize(vmin=S["bmin"], vmax=S["bmax"])
    light_dir = mcolors.LightSource(
        azdeg=config.light_azimuth_degrees,
        altdeg=config.light_altitude_degrees).direction
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return cmap, norm, light_dir, mappable


def require_pyvista():
    """Import the optional real-3-D backend with an actionable error."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit(
            "error: --backend pyvista requires the optional PyVista/VTK "
            "renderer; install it with 'python -m pip install pyvista'") \
            from exc
    return pv


def mesh_quads(X, Y, Z):
    """Convert a structured surface mesh into uniformly shaped quads."""
    points = np.stack([X, Y, Z], axis=-1)
    return np.stack(
        [points[:-1, :-1], points[1:, :-1],
         points[1:, 1:], points[:-1, 1:]], axis=2).reshape(-1, 4, 3)


def quad_face_colors(vertex_colors):
    """Average structured RGBA vertex colors onto the corresponding quads."""
    return (vertex_colors[:-1, :-1] + vertex_colors[1:, :-1] +
            vertex_colors[1:, 1:] + vertex_colors[:-1, 1:]).reshape(
                -1, 4) * 0.25


def shade_quads(quads, color, light_dir):
    """Apply Matplotlib-compatible diffuse shading to uniformly colored
    quads before they enter the shared depth-sorted collection."""
    normals = np.cross(quads[:, 0] - quads[:, 1],
                       quads[:, 1] - quads[:, 2])
    magnitudes = np.linalg.norm(normals, axis=1)
    shade = np.zeros(len(quads), dtype=float)
    valid = magnitudes > 0.0
    shade[valid] = normals[valid] @ light_dir / magnitudes[valid]
    # Axes3D._shade_colors maps a [-1, 1] dot product onto [0.3, 1].
    intensity = 0.65 + 0.35 * np.clip(shade, -1.0, 1.0)
    rgba = np.tile(mcolors.to_rgba(color), (len(quads), 1))
    rgba[:, :3] *= intensity[:, None]
    return rgba


def draw_scene(ax, vw, S, cmap, norm, light_dir):
    """Draw plasma and coils as one globally face-sorted collection.

    Matplotlib has no depth buffer: separate Poly3DCollection objects receive
    one painter-order depth each. Combining every plasma and coil face lets
    its internal polygon sorter interleave the two geometries by view depth.
    """
    geometry = []
    config = S["figure_parameters"]
    colors = []
    antialiased = []
    for s in S["surf_3d"]:  # a single opaque surface (the plasma boundary)
        X, Y, Z = to_cartesian(s["r"], s["z"], S["nfp"])
        B = periodic_close(np.concatenate([s["bmag"]] * S["nfp"], axis=1))
        lam = surface_intensity(
            X, Y, Z, light_dir, lo=config.surface_intensity_floor)
        hsv = mcolors.rgb_to_hsv(cmap(norm(B))[:, :, :3])
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * lam, 0.0, 1.0)
        face = np.concatenate(
            [mcolors.hsv_to_rgb(hsv), np.ones(hsv.shape[:2] + (1,))], axis=2)
        # Match the former plot_surface cstride=2 while keeping uniform quads.
        plasma_quads = mesh_quads(X[:, ::2], Y[:, ::2], Z[:, ::2])
        geometry.append(plasma_quads)
        colors.append(quad_face_colors(face[:, ::2]))
        antialiased.append(np.zeros(len(plasma_quads), dtype=bool))
    for filament in S["coils"]:
        X, Y, Z = tube_mesh(filament, S["coil_radius"])
        coil_quads = mesh_quads(X, Y, Z)
        geometry.append(coil_quads)
        colors.append(shade_quads(
            coil_quads, config.coil_color, light_dir))
        antialiased.append(np.ones(len(coil_quads), dtype=bool))
    if geometry:
        surfaces = Poly3DCollection(
            np.concatenate(geometry),
            facecolors=np.concatenate(colors), edgecolors="none",
            linewidths=0.0, antialiaseds=np.concatenate(antialiased),
            zsort="average")
        ax.add_collection3d(surfaces)
    for (x, y, z) in S["lines"]:
        ax.plot(x, y, z, color=config.curve_color,
                linewidth=config.field_line_width,
                alpha=config.field_line_alpha, zorder=50)
    ax.plot(S["axx"], S["axy"], S["axz"], color=config.curve_color,
            linewidth=config.axis_line_width, zorder=60)
    # Data-derived limits hug the plasma so the torus fills the frame
    # instead of floating in white space.
    lim = S["lim"]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-S["zlim"], S["zlim"])
    ax.set_box_aspect(S["box_aspect"])
    ax.set_axis_off()
    ax.view_init(**vw)


def pyvista_surface(pv, X, Y, Z, scalars):
    """Create a quad PolyData surface without relying on VTK array-order
    conventions.  Points and scalar values both use NumPy's C order."""
    n0, n1 = X.shape
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n0 - 1), np.arange(n1 - 1),
                       indexing="ij")
    corner = (i * n1 + j).ravel()
    faces = np.column_stack([
        np.full(corner.size, 4, dtype=np.int64),
        corner, corner + n1, corner + n1 + 1, corner + 1,
    ])
    mesh = pv.PolyData(points, faces)
    mesh.point_data["B"] = np.asarray(scalars).ravel()
    return mesh


def pyvista_polyline(pv, points, close=False):
    """Create one connected VTK polyline, optionally closing the loop."""
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    points = points[keep]
    if close and len(points) > 1:
        if np.linalg.norm(points[-1] - points[0]) <= 1e-12:
            points = points[:-1]
        points = np.vstack([points, points[0]])
    line = pv.PolyData(points)
    line.lines = np.concatenate(
        [np.array([len(points)], dtype=np.int64),
         np.arange(len(points), dtype=np.int64)])
    return line


def add_pyvista_scene(plotter, pv, S, section_angles=()):
    """Populate one VTK renderer with opaque, depth-tested scene actors."""
    config = S["figure_parameters"]
    for s in S["surf_3d"]:
        X, Y, Z = to_cartesian(s["r"], s["z"], S["nfp"])
        B = periodic_close(np.concatenate([s["bmag"]] * S["nfp"], axis=1))
        mesh = pyvista_surface(pv, X, Y, Z, B)
        plotter.add_mesh(
            mesh, scalars="B", cmap=config.field_colormap,
            clim=(S["bmin"], S["bmax"]), show_scalar_bar=False,
            smooth_shading=True, ambient=0.25, diffuse=0.75,
            specular=0.08, specular_power=12.0)

    for filament in S["coils"]:
        centerline = pyvista_polyline(pv, filament, close=True)
        tube = centerline.tube(radius=S["coil_radius"], n_sides=12,
                               capping=False)
        plotter.add_mesh(tube, color=config.coil_color, smooth_shading=True,
                         ambient=0.22, diffuse=0.72, specular=0.28,
                         specular_power=24.0)

    for x, y, z in S["lines"]:
        line = pyvista_polyline(pv, np.column_stack([x, y, z]))
        plotter.add_mesh(line, color=config.curve_color, line_width=1.5,
                         render_lines_as_tubes=True, lighting=False)
    axis = pyvista_polyline(
        pv, np.column_stack([S["axx"], S["axy"], S["axz"]]))
    plotter.add_mesh(axis, color=config.curve_color, line_width=2.0,
                     render_lines_as_tubes=True, lighting=False)

    # These guides belong only to the top-view panel in the slices figure.
    # They remain ordinary depth-tested actors: sections passing behind the
    # plasma are hidden instead of being painted over it.
    for zeta in section_angles:
        phi = zeta / S["nfp"]
        guide = pyvista_polyline(
            pv, [[0.0, 0.0, 0.0],
                 [S["lim"] * np.cos(phi), S["lim"] * np.sin(phi), 0.0]])
        plotter.add_mesh(guide, color=config.curve_color, line_width=1.2,
                         render_lines_as_tubes=True, lighting=False)


def set_pyvista_camera(plotter, vw, S):
    """Match the two established Matplotlib viewpoints in VTK."""
    elev = np.radians(vw["elev"])
    azim = np.radians(vw["azim"])
    direction = np.array([np.cos(elev) * np.cos(azim),
                          np.cos(elev) * np.sin(azim), np.sin(elev)])
    distance = 4.5 * max(S["lim"], S["zlim"])
    if abs(vw["elev"]) > 89.0:
        # A defined up-vector avoids the singular roll at the north pole.
        direction = np.array([0.0, 0.0, 1.0])
        view_up = (0.0, 1.0, 0.0)
    else:
        view_up = (0.0, 0.0, 1.0)
    plotter.camera_position = [tuple(distance * direction),
                               (0.0, 0.0, 0.0), view_up]
    if abs(vw["elev"]) > 89.0:
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 1.04 * S["lim"]
    else:
        plotter.camera.parallel_projection = False
        plotter.camera.view_angle = 30.0
    plotter.reset_camera_clipping_range()


def pyvista_scene_image(S, vw, section_angles=()):
    """Rasterize one scene through VTK's z-buffer and return an RGB image."""
    pv = require_pyvista()
    config = S["figure_parameters"]
    plotter = pv.Plotter(off_screen=True,
                         window_size=config.pyvista_window_size,
                         lighting="light kit")
    plotter.set_background("white")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except ValueError:  # older PyVista releases without SSAA support
        plotter.enable_anti_aliasing()
    add_pyvista_scene(plotter, pv, S, section_angles)
    set_pyvista_camera(plotter, vw, S)
    try:
        return plotter.screenshot(return_img=True)
    finally:
        plotter.close()


def render_single_view(base, suffix, label, vw, S):
    """One 3-D view per file (perspective / top); runs in its own process."""
    print(f"rendering {label} ...", flush=True)
    t0 = time.perf_counter()
    config = S["figure_parameters"]
    cmap, norm, light_dir, mappable = _render_machinery(S)
    # constrained_layout: 3-D projections reserve wide margins for
    # rotation; constrained_layout + bbox_inches='tight' mitigates them
    # (the rest is handled by trim_white).
    fig = plt.figure(figsize=config.overview_single_figure_size,
                     dpi=config.overview_working_dpi,
                     constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax, vw, S, cmap, norm, light_dir)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, aspect=20, pad=0.01)
    cbar.set_label(config.overview_colorbar_label,
                   fontsize=config.overview_colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=config.overview_colorbar_tick_fontsize)
    # The title is re-centered onto the torus by trim_white (the 3-D
    # box is not centered in the axes window, so matplotlib-side
    # placement cannot do it).
    fig.suptitle(f"{S['title']} ({label})",
                 fontsize=config.overview_title_fontsize, y=1.01)
    out_png = figure_path(base, suffix)
    fig.savefig(out_png, dpi=config.overview_save_dpi,
                bbox_inches=config.save_bbox,
                facecolor=config.save_face_color)
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def render_single_view_image(base, suffix, label, image, S):
    """Lay out a VTK-rendered RGB scene with Matplotlib annotations."""
    print(f"composing {label} ...", flush=True)
    t0 = time.perf_counter()
    config = S["figure_parameters"]
    _, _, _, mappable = _render_machinery(S)
    fig = plt.figure(figsize=config.overview_single_figure_size,
                     dpi=config.overview_working_dpi,
                     constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_axis_off()
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, aspect=20, pad=0.01)
    cbar.set_label(config.overview_colorbar_label,
                   fontsize=config.overview_colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=config.overview_colorbar_tick_fontsize)
    fig.suptitle(f"{S['title']} ({label})",
                 fontsize=config.overview_title_fontsize, y=0.99)
    out_png = figure_path(base, suffix)
    fig.savefig(out_png, dpi=config.pyvista_save_dpi,
                bbox_inches=config.save_bbox,
                facecolor=config.save_face_color)
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def render_combined(base, S):
    """Two side-by-side views with one shared horizontal colorbar."""
    print("rendering combined view ...", flush=True)
    t0 = time.perf_counter()
    config = S["figure_parameters"]
    cmap, norm, light_dir, mappable = _render_machinery(S)
    # Plain side-by-side subplots; the figure is sized so each panel window
    # is nearly square (the top-view box then fills it and the inter-panel
    # slack vanishes). trim_white removes the remaining projection margins.
    fig = plt.figure(figsize=config.overview_combined_figure_size,
                     dpi=config.overview_working_dpi)
    axes = []
    for i, (suffix, label, vw) in enumerate(S["views"]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        axes.append(ax)
        draw_scene(ax, vw, S, cmap, norm, light_dir)
        # mplot3d ignores the title pad, so place the label manually just
        # above the rendered box top (measured ~0.62 of the window).
        label_y = 0.92 if S["coils"] else 0.70
        ax.text2D(0.5, label_y, label, transform=ax.transAxes, ha="center",
                  va="bottom", fontsize=10)
    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal",
                        shrink=0.6, aspect=24, pad=0.16)
    cbar.set_label(config.overview_colorbar_label,
                   fontsize=config.overview_colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=config.overview_colorbar_tick_fontsize)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=0.0)
    # tight_layout overrides the colorbar pad; pin the strip just below
    # the rendered 3-D box. The box's projected bottom sits at ~0.188 of
    # the figure height (mplot3d leaves a margin inside the window), so
    # the strip top = 0.188 - 60px gap.
    cbar.ax.set_position([0.30, 0.106, 0.40, 0.045])
    fig.suptitle(S["title"], fontsize=config.overview_title_fontsize, y=0.98)
    out_png = figure_path(base, "combined")
    fig.savefig(out_png, dpi=config.overview_save_dpi,
                bbox_inches=config.save_bbox,
                facecolor=config.save_face_color)
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def render_combined_images(base, S, images):
    """Lay out two VTK-rendered views with one shared scalar bar."""
    print("composing combined view ...", flush=True)
    t0 = time.perf_counter()
    config = S["figure_parameters"]
    _, _, _, mappable = _render_machinery(S)
    cropped_images = []
    for image in images:
        # VTK frames each camera in a fixed rectangular window. Crop those
        # white margins only for the combined-view copies; the standalone
        # views retain their original framing.
        rgb = image[:, :, :3]
        nonwhite = np.any(rgb < 250, axis=2)
        ys, xs = np.where(nonwhite)
        if len(xs):
            pad = 6
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1,
                                                   image.shape[1])
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1,
                                                   image.shape[0])
            image = image[y0:y1, x0:x1]
        cropped_images.append(image)
    # Match the Matplotlib combined composition: equal-width panel cells make
    # the wide perspective projection naturally shorter than the nearly square
    # top view, instead of scaling both scenes to the same height.
    fig, axes = plt.subplots(
        1, 2, figsize=config.overview_combined_figure_size,
        dpi=config.overview_working_dpi,
        gridspec_kw={"width_ratios": (0.78, 1.0)})
    # Reserve a real bottom strip before adding the colorbar. Calling
    # subplots_adjust after fig.colorbar lets the image axes expand back into
    # the automatically carved-out colorbar region, which can cover the lower
    # part of the top-view coils.
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.18,
                        wspace=0.06)
    for i, (ax, image, (_, label, _)) in enumerate(
            zip(axes, cropped_images, S["views"])):
        ax.imshow(image)
        # Aspect correction shrinks the active image box inside its subplot.
        # Anchor both boxes toward the shared edge so that slack stays outside
        # rather than reopening the middle gutter.
        ax.set_anchor("E" if i == 0 else "W")
        ax.set_axis_off()
        ax.set_title(label, fontsize=10, pad=3)
    cbar_ax = fig.add_axes([0.30, 0.14, 0.40, 0.03])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(config.overview_colorbar_label,
                   fontsize=config.overview_colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=config.overview_colorbar_tick_fontsize)
    fig.suptitle(S["title"], fontsize=config.overview_title_fontsize, y=0.98)
    out_png = figure_path(base, "combined")
    fig.savefig(out_png, dpi=config.pyvista_save_dpi,
                bbox_inches=config.save_bbox,
                pad_inches=12.0 / config.pyvista_save_dpi,
                facecolor=config.save_face_color)
    plt.close(fig)
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def render_slices(base, S, scene_image=None):
    """Top view + six RZ poloidal cross-sections (two rows of three,
    spanning one field period), each with the nested flux-surface contour
    family."""
    print("rendering top view + RZ slices ...", flush=True)
    t0 = time.perf_counter()
    config = S["figure_parameters"]
    cmap, norm, light_dir, mappable = _render_machinery(S)
    surf_slices = S["surf_slices"]
    fams, ns, nzt_r = S["fams"], S["ns"], S["nzt_r"]
    nfp, ntor = S["nfp"], S["ntor"]
    zeta_cuts = [
        (k * 2.0 * np.pi / config.slice_panel_count,
         f"φ = {np.degrees(k * 2.0 * np.pi / (config.slice_panel_count * nfp)):.0f}°")
        for k in range(config.slice_panel_count)]
    # Explicit layout: the slice block gets a bounded region that is
    # SHORTER than the 3-D panel, so the two-row slice block ends up
    # smaller than the top-view torus.
    fig = plt.figure(figsize=config.overview_slices_figure_size,
                     dpi=config.overview_working_dpi)
    gs = fig.add_gridspec(2, 6, left=0.05, right=0.44, top=0.72,
                          bottom=0.30, hspace=0.35, wspace=0.06)
    # Two rows of three slices (2 columns each).
    axs = [fig.add_subplot(gs[0, 0:2]),
           fig.add_subplot(gs[0, 2:4]),
           fig.add_subplot(gs[0, 4:6]),
           fig.add_subplot(gs[1, 0:2]),
           fig.add_subplot(gs[1, 2:4]),
           fig.add_subplot(gs[1, 4:6])]
    # Nearly square window (matches the top-view box) so the torus fills
    # it and the gap to the slice block shrinks. The PyVista backend supplies
    # an already depth-buffered image; the default backend draws into Axes3D.
    if scene_image is None:
        ax3d = fig.add_axes([0.455, 0.06, 0.48, 0.86], projection="3d")
        draw_scene(ax3d, dict(elev=config.top_view[0],
                              azim=config.top_view[1]),
                   S, cmap, norm, light_dir)
        # Mark the toroidal positions of the cross-sections on the top view
        # with radial dashed lines (one period only).
        lim = S["lim"]
        for zc, _ in zeta_cuts:
            phi = zc / nfp
            ax3d.plot([0.0, lim * np.cos(phi)], [0.0, lim * np.sin(phi)],
                      [0.0, 0.0], color=config.curve_color,
                      linestyle=(0, (5, 4)),
                      linewidth=0.9, zorder=55)
    else:
        ax3d = fig.add_axes([0.455, 0.06, 0.48, 0.86])
        ax3d.imshow(scene_image)
        ax3d.set_axis_off()
    r_all = np.concatenate([s["r"].ravel() for s in surf_slices])
    z_all = np.concatenate([s["z"].ravel() for s in surf_slices])
    Rmin, Rmax = r_all.min(), r_all.max()
    Zmax = max(abs(z_all.min()), abs(z_all.max()))
    for i, (zc, name) in enumerate(zeta_cuts):
        idx = int(round(zc / (2.0 * np.pi) * nzt_r)) % nzt_r
        ax = axs[i]
        for k, s in enumerate(surf_slices):
            R = np.append(s["r"][:, idx], s["r"][0, idx])
            Z = np.append(s["z"][:, idx], s["z"][0, idx])
            B = np.append(s["bmag"][:, idx], s["bmag"][0, idx])
            pts = np.stack([R, Z], axis=1)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm,
                                linewidths=2.4 if k == 0 else 1.0)
            lc.set_array(B[:-1])
            ax.add_collection(lc)
        Rax = sum(fams["rmncc"][nn * ns + 1] * np.cos(nn * zc)
                  for nn in range(ntor + 1))
        Zax = sum(fams["zmncs"][nn * ns + 1] * np.sin(nn * zc)
                  for nn in range(ntor + 1))
        ax.plot([Rax], [Zax], "o", color=config.curve_color, markersize=2.5)
        ax.set_xlim(Rmin - 0.05, Rmax + 0.05)
        # Mild vertical headroom; the slice plots fill their cell heights
        # either way, so this only controls their width.
        ax.set_ylim(-1.1 * Zmax, 1.1 * Zmax)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=8.5, pad=2)
        ax.tick_params(labelsize=7.5)
        # One vertical frame label per row (first slice of each row only).
        if i in (0, 3):
            ax.set_ylabel("Z (m)", fontsize=8)
        if i < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("R (m)", fontsize=8)
    cbar = fig.colorbar(mappable, ax=ax3d, shrink=0.6, aspect=20, pad=0.01)
    cbar.set_label(config.overview_colorbar_label,
                   fontsize=config.overview_colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=config.overview_colorbar_tick_fontsize)
    fig.suptitle(f"{S['title']} (top view + poloidal cross-sections)",
                 fontsize=config.overview_title_fontsize, y=1.01)
    out_png = figure_path(base, "slices")
    fig.savefig(out_png, dpi=config.overview_save_dpi,
                bbox_inches=config.save_bbox,
                facecolor=config.save_face_color)
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def render_pyvista_figures(base, S):
    """Render all 3-D panels sequentially through one real-3-D backend.

    VTK/OpenGL contexts are deliberately not forked.  The two plain views
    are reused in the single and combined figures; only the slices panel
    needs a third render carrying its six section guides.
    """
    images = []
    for _, label, vw in S["views"]:
        print(f"VTK rendering {label} ...", flush=True)
        t0 = time.perf_counter()
        images.append(pyvista_scene_image(S, vw))
        print(f"  rendered in {time.perf_counter() - t0:.1f}s", flush=True)
    for image, (suffix, label, _) in zip(images, S["views"]):
        render_single_view_image(base, suffix, label, image, S)
    render_combined_images(base, S, images)
    print("VTK rendering top view with section guides ...", flush=True)
    config = S["figure_parameters"]
    section_angles = tuple(
        k * 2.0 * np.pi / config.slice_panel_count
        for k in range(config.slice_panel_count))
    slices_image = pyvista_scene_image(
        S, dict(elev=config.top_view[0], azim=config.top_view[1]),
        section_angles=section_angles)
    render_slices(base, S, scene_image=slices_image)
    print("all figures saved", flush=True)
