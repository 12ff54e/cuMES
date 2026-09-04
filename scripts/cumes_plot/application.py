"""Command-line orchestration for equilibrium plotting."""

import argparse
import multiprocessing
import os
import sys

import numpy as np

from .coils import load_coil_filaments, resolve_coils_path
from .config import validate_figure_config
from .coordinate_figures import render_coordinate_grids
from .coordinates import load_boozer, make_boozer_grid, make_pest_grid
from .equilibrium import (
    boundary_from_params, converged_axis, eval_state, field_lines, half_grid,
    make_profiles, solve_chip,
)
from .output_paths import DEFAULT_OUTPUT_PATH, resolve_output_base
from .render_3d import (
    render_combined, render_pyvista_figures, render_single_view, render_slices,
    require_pyvista,
)
from .state_io import load_state


def main(figure_parameters):
    figure_parameters = validate_figure_config(figure_parameters)
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="figure_data/w7x_state.bin")
    output = ap.add_mutually_exclusive_group()
    output.add_argument(
        "--out", metavar="PATH",
        help="base output path; its final extension is removed and four PNGs "
             "are written as BASE_perspective.png, BASE_top.png, "
             "BASE_combined.png, and BASE_slices.png "
             "(default: figure_data/equilibrium.png)")
    output.add_argument(
        "--output-dir", metavar="DIRECTORY",
        help="output directory with no filename prefix; writes names such as "
             "perspective.png, combined.png, field_slices.png, and "
             "pest_field_contours.png")
    ap.add_argument("--field-lines", action="store_true",
                    help="also trace and draw field lines on the edge surface "
                         "(off by default: the figure stays cleaner)")
    ap.add_argument("--coils", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="for a free-boundary state, overlay the MAKEGRID "
                         "coil filaments as bronze tubes; omit PATH to use "
                         "the coils_file path recorded in the state (the coil "
                         "geometry itself is not embedded)")
    ap.add_argument("--coil-radius", type=float, default=None,
                    metavar="METERS",
                    help="coil tube radius in meters (default: 0.6%% of the "
                         "plasma's major radial extent)")
    ap.add_argument("--backend", choices=("matplotlib", "pyvista"),
                    default="matplotlib",
                    help="3-D rendering backend (default: matplotlib; "
                         "pyvista uses VTK's z-buffer and requires the "
                         "optional pyvista package)")
    ap.add_argument(
        "--boozer-state", metavar="PATH",
        help="Boozer-v3 .bin, .nc, .h5, or .hdf5 result used for the Boozer "
             "figures; PEST figures are reconstructed from native --state")
    ap.add_argument(
        "--coordinate-system", choices=("both", "pest", "boozer"),
        help="magnetic-coordinate variants to render (default: both; Boozer "
             "requires --boozer-state, PEST requires --state)")
    ap.add_argument(
        "--coordinate-only", action="store_true",
        help="render only the selected six-panel magnetic-coordinate figures")
    if len(sys.argv) == 1:
        ap.print_help()
        return
    args = ap.parse_args()

    # Fail before loading and reconstructing a potentially large state.
    if args.backend == "pyvista" and not args.coordinate_only:
        require_pyvista()

    if args.coil_radius is not None and args.coil_radius <= 0.0:
        ap.error("--coil-radius must be positive")
    if args.coil_radius is not None and args.coils is None:
        ap.error("--coil-radius requires --coils")
    if args.coordinate_only and args.coils is not None:
        ap.error("--coils is not used with --coordinate-only")
    if args.coordinate_only and args.field_lines:
        ap.error("--field-lines is not used with --coordinate-only")

    try:
        output_base = resolve_output_base(args.out, args.output_dir)
    except ValueError as exc:
        ap.error(str(exc))
    output_dir = args.output_dir if args.output_dir is not None else \
        os.path.dirname(os.path.abspath(args.out or DEFAULT_OUTPUT_PATH))
    os.makedirs(output_dir, exist_ok=True)

    coordinate_requested = args.coordinate_only or \
        args.boozer_state is not None or args.coordinate_system is not None
    coordinate_choice = args.coordinate_system or "both"
    coordinate_systems = ("pest", "boozer") \
        if coordinate_choice == "both" else (coordinate_choice,)
    needs_pest = coordinate_requested and "pest" in coordinate_systems
    needs_boozer = coordinate_requested and "boozer" in coordinate_systems
    if needs_boozer and args.boozer_state is None:
        ap.error("Boozer coordinate plots require --boozer-state")

    coordinate_data = None
    if needs_boozer:
        coordinate_data = load_boozer(args.boozer_state)
        print(f"loaded Boozer state: surfaces={coordinate_data.surface_count}, "
              f"ntheta={coordinate_data.ntheta}, "
              f"nzeta={coordinate_data.nzeta}, nfp={coordinate_data.nfp}, "
              f"|B|={coordinate_data.b.min():.3f}-"
              f"{coordinate_data.b.max():.3f} T", flush=True)
        if args.coordinate_only and not needs_pest:
            render_coordinate_grids(
                [make_boozer_grid(coordinate_data,
                                  max(figure_parameters.min_render_theta,
                                      coordinate_data.ntheta),
                                  max(figure_parameters.min_render_zeta,
                                      coordinate_data.nzeta))],
                output_base, coordinate_data.nfp,
                os.path.splitext(os.path.basename(
                    coordinate_data.source_path or args.boozer_state))[0],
                figure_parameters)
            return

    ns, mnmax, fams, params, name = load_state(args.state)
    if args.coils is not None and not params["lfreeb"]:
        ap.error("--coils is only valid for free-boundary states")
    ntor, nfp = params["ntor"], params["nfp"]
    if mnmax != params["mpol"] * (ntor + 1):
        raise SystemExit(
            f"error: mnmax {mnmax} != mpol*(ntor+1) "
            f"{params['mpol']}*{ntor + 1} — corrupt embedded input record")
    prof = make_profiles(params, ns)
    print(f"loaded state: ns={ns}, mnmax={mnmax}, mpol={params['mpol']}, "
          f"ntor={ntor}, nfp={nfp}, ntheta={params['ntheta']}, "
          f"nzeta={params['nzeta']}, ncurr={params['ncurr']}, "
          f"phiedge={params['phiedge']:g}, lfreeb={params['lfreeb']}",
          flush=True)

    # ---- self-check 1: state LCFS vs the embedded initial boundary ------
    print("self-check 1/2: state LCFS vs embedded initial rbc/zbs boundary ...",
          flush=True)
    th_ax = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    zt_ax = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    b = eval_state(fams, ns, ns - 1, th_ax, zt_ax, ntor, nfp)
    maxsc = max(np.sqrt((ns - 1) / (ns - 1)), np.sqrt(1.0 / (ns - 1)))
    R_phys = b["re"] + maxsc * b["ro"]
    Z_phys = b["ze"] + maxsc * b["zo"]
    TH_c, ZT_c = np.meshgrid(th_ax, zt_ax, indexing="ij")
    Rb, Zb = boundary_from_params(params, TH_c, ZT_c)
    err = max(np.max(np.abs(R_phys - Rb)), np.max(np.abs(Z_phys - Zb)))
    if not np.isfinite(err):
        raise SystemExit("error: non-finite LCFS boundary reconstruction")
    if params["lfreeb"]:
        print(f"  initial-to-converged max displacement = {err:.3e} "
              "(expected for free boundary)", flush=True)
    else:
        print(f"  max err = {err:.3e}", flush=True)
        assert err < 1e-8, "state LCFS does not match the embedded boundary"

    # ---- edge half-grid field data used by interpolation / field lines ---
    print(f"edge surface: solving chi' + half-grid geometry (jh = {ns - 2}) ...",
          flush=True)
    th = np.linspace(0.0, 2.0 * np.pi, params["ntheta"], endpoint=False)
    zt = np.linspace(0.0, 2.0 * np.pi, params["nzeta"], endpoint=False)
    chip = solve_chip(fams, ns, ns - 2, params, prof)
    a = [eval_state(fams, ns, j, th, zt, ntor, nfp)
         for j in (ns - 2, ns - 1)]
    edge = half_grid(a, ns, ns - 2, chip, prof["phip_avg"](ns - 2),
                     prof["lamscale"])
    print(f"  jh={ns - 2:3d}: chip={chip:+.4e}, |B| "
          f"{edge['bmag'].min():.3f}-{edge['bmag'].max():.3f} T", flush=True)

    # ---- fine render grid (full-grid geometry + interpolated |B|) ---------
    # The visible contours use integer/full radial surfaces. |B| naturally
    # lives on the staggered half grid, so interior full-grid values are the
    # average of their two bracketing half-grid values. The LCFS uses the
    # corresponding one-sided linear extrapolation from the last two halves.
    print("fine render grid: evaluating full-grid geometry + interpolated "
          "|B| on 240x120 ...", flush=True)
    nth_r, nzt_r = 240, 120
    th_r = np.linspace(0.0, 2.0 * np.pi, nth_r, endpoint=False)
    zt_r = np.linspace(0.0, 2.0 * np.pi, nzt_r, endpoint=False)
    render_states = {}
    render_halves = {}

    def render_state(j):
        if j not in render_states:
            render_states[j] = eval_state(
                fams, ns, j, th_r, zt_r, ntor, nfp)
        return render_states[j]

    def render_half(jh):
        if jh not in render_halves:
            chip_h = solve_chip(fams, ns, jh, params, prof)
            adjacent = [render_state(jh), render_state(jh + 1)]
            render_halves[jh] = half_grid(
                adjacent, ns, jh, chip_h, prof["phip_avg"](jh),
                prof["lamscale"])
        return render_halves[jh]

    def render_full(j):
        state = render_state(j)
        rho = np.sqrt(j / (ns - 1))
        R = state["re"] + rho * state["ro"]
        Z = state["ze"] + rho * state["zo"]
        if j == 0:
            bmag = 1.5 * render_half(0)["bmag"] \
                - 0.5 * render_half(1)["bmag"]
        elif j == ns - 1:
            bmag = 1.5 * render_half(ns - 2)["bmag"] \
                - 0.5 * render_half(ns - 3)["bmag"]
        else:
            bmag = 0.5 * (render_half(j - 1)["bmag"]
                          + render_half(j)["bmag"])
        return {"r": R, "z": Z, "bmag": np.maximum(bmag, 0.0), "j": j}

    j_slices = tuple(range(ns - 1, 8, -8))  # LCFS down toward s ~ 0.1
    surf_slices = [render_full(j) for j in j_slices]
    surf_3d = [surf_slices[0]]              # the full-grid LCFS
    b_all = np.concatenate([s["bmag"].ravel() for s in surf_slices])
    print(f"  3D: full-grid LCFS j={ns - 1}, slices: "
          f"{len(surf_slices)} full-grid contours, "
          f"|B| range {b_all.min():.3f} - {b_all.max():.3f} T", flush=True)

    # ---- PEST source: native R/Z/lambda + reconstructed full-grid B --------
    coordinate_grids = {}
    if needs_pest:
        print("PEST grid: reconstructing native geometry, physical lambda, "
              "and full-grid |B| ...", flush=True)
        nth_p = max(figure_parameters.min_render_theta, params["ntheta"])
        nzt_p = max(figure_parameters.min_render_zeta, params["nzeta"])
        theta_p_source = np.linspace(0.0, 2.0 * np.pi, nth_p,
                                     endpoint=False)
        zeta_p = np.linspace(0.0, 2.0 * np.pi, nzt_p, endpoint=False)
        shape = (ns - 1, nzt_p, nth_p)
        pest_r = np.empty(shape)
        pest_z = np.empty(shape)
        pest_b = np.empty(shape)
        pest_lambda = np.empty(shape)

        def pest_state(surface):
            return eval_state(fams, ns, surface, theta_p_source, zeta_p,
                              ntor, nfp)

        def pest_half(half_surface, inner, outer):
            chip_h = solve_chip(fams, ns, half_surface, params, prof)
            return half_grid([inner, outer], ns, half_surface, chip_h,
                             prof["phip_avg"](half_surface),
                             prof["lamscale"])

        def save_pest_surface(surface, state, bmag):
            rho = np.sqrt(surface / (ns - 1))
            r_native = state["re"] + rho * state["ro"]
            z_native = state["ze"] + rho * state["zo"]
            lambda_stored = state["le"] + rho * state["lo"]
            phip = prof["phip_full"](surface)
            if not np.isfinite(phip) or abs(phip) <= 1.0e-30:
                raise SystemExit(
                    "error: physical PEST lambda is undefined where "
                    "Phi-prime vanishes")
            out = surface - 1
            pest_r[out] = r_native.T
            pest_z[out] = z_native.T
            pest_b[out] = np.maximum(bmag.T, 0.0)
            pest_lambda[out] = (
                prof["lamscale"] / phip * lambda_stored).T

        inner_state = pest_state(0)
        current_state = pest_state(1)
        previous_half = pest_half(0, inner_state, current_state)
        for half_surface in range(1, ns - 1):
            outer_state = pest_state(half_surface + 1)
            current_half = pest_half(
                half_surface, current_state, outer_state)
            save_pest_surface(
                half_surface, current_state,
                0.5 * (previous_half["bmag"] + current_half["bmag"]))
            if half_surface == ns - 2:
                save_pest_surface(
                    ns - 1, outer_state,
                    1.5 * current_half["bmag"] -
                    0.5 * previous_half["bmag"])
            previous_half = current_half
            current_state = outer_state

        s_pest = np.arange(1, ns, dtype=float) / (ns - 1)
        try:
            coordinate_grids["pest"] = make_pest_grid(
                s_pest, theta_p_source, zeta_p, pest_r, pest_z, pest_b,
                pest_lambda)
        except ValueError as exc:
            raise SystemExit(f"error: could not construct PEST grid: {exc}") \
                from exc

    if needs_boozer:
        if coordinate_data.nfp != nfp:
            raise SystemExit(
                f"error: Boozer state has nfp={coordinate_data.nfp}, "
                f"native state has nfp={nfp}")
        coordinate_grids["boozer"] = make_boozer_grid(
            coordinate_data,
            max(figure_parameters.min_render_theta, coordinate_data.ntheta),
            max(figure_parameters.min_render_zeta, coordinate_data.nzeta))

    if coordinate_requested:
        render_coordinate_grids(
            [coordinate_grids[system] for system in coordinate_systems],
            output_base, nfp, name, figure_parameters)
        if args.coordinate_only:
            return

    # ---- field lines on the edge surface (opt-in) ------------------------
    lines = []
    if args.field_lines:
        print("field lines: tracing d(theta)/d(zeta) = B^theta/B^zeta (RK4) ...",
              flush=True)
        try:
            seeds = [2.0 * np.pi * fraction for fraction in
                     figure_parameters.field_line_seed_fractions]
            lines = field_lines(edge, th, zt, nfp=nfp, seeds=seeds,
                                geometry=surf_3d[0])
            print(f"  {len(lines)} lines traced", flush=True)
        except Exception as exc:  # decorative; the figure survives without them
            lines = []
            print(f"  field lines skipped: {exc}", flush=True)

    # ---- converged magnetic axis (m=0 content of the innermost surface) --
    axx, axy, axz = converged_axis(fams, ns, ntor, nfp)
    print(f"magnetic axis: R {axx.min():.3f}-{axx.max():.3f}, "
          f"Z {axz.min():.3f}-{axz.max():.3f}", flush=True)

    # ---- rendering (four figures, one process per figure) -----------------
    # matplotlib's Agg rasterizer largely holds the GIL and pyplot state is
    # not thread-safe across concurrent figures, so threads would serialize;
    # forked processes run in parallel and inherit the shared data
    # copy-on-write (no pickling of the surface arrays). High-contrast
    # viridis (bright yellow = large |B|, deep purple = small |B|) is mapped
    # over the actual data range; shading uses the geometric surface normals
    # (surface_intensity), never the |B| heightfield.
    # Data-derived framing: the render window hugs the plotted surface with
    # a small margin, so any device fills the frame.
    r_edge = surf_3d[0]["r"]
    z_edge = surf_3d[0]["z"]
    plasma_lim = float(np.max(r_edge))
    plasma_zlim = float(max(abs(z_edge.min()), abs(z_edge.max())))
    coils = []
    coil_radius = 0.0
    if args.coils is not None:
        coils_path = resolve_coils_path(args.coils, params)
        coil_periods, loaded_coils = load_coil_filaments(coils_path)
        if coil_periods != nfp:
            raise SystemExit(
                f"error: coils file has {coil_periods} field periods, state "
                f"has nfp={nfp}")
        # Idealized axis-current filaments can use million-meter closure
        # segments.  They produce the intended field but are not physical
        # coils and would collapse the useful plot scale, so omit only such
        # unmistakable far-field constructions.
        visible_extent = figure_parameters.coil_visible_extent_factor * \
            max(plasma_lim, plasma_zlim)
        coils = [filament for filament in loaded_coils
                 if np.max(np.linalg.norm(filament, axis=1)) <= visible_extent]
        skipped = len(loaded_coils) - len(coils)
        if not coils:
            raise SystemExit("error: no coil filaments lie within the useful "
                             "plot extent")
        coil_radius = args.coil_radius \
            if args.coil_radius is not None \
            else figure_parameters.coil_default_radius_fraction * plasma_lim
        print(f"coils: loaded {len(coils)} filament(s) from {coils_path}; "
              f"bronze tube radius={coil_radius:.4g} m" +
              (f"; skipped {skipped} far-field filament(s)" if skipped else ""),
              flush=True)

    coil_xy_lim = max((np.max(np.hypot(f[:, 0], f[:, 1]))
                       for f in coils), default=0.0)
    coil_zlim = max((np.max(np.abs(f[:, 2])) for f in coils), default=0.0)
    lim = figure_parameters.horizontal_limit_margin * \
        max(plasma_lim, coil_xy_lim + coil_radius)
    zlim = figure_parameters.vertical_limit_margin * \
        max(plasma_zlim, coil_zlim + coil_radius)
    perspective_view = dict(
        elev=figure_parameters.perspective_view[0],
        azim=figure_parameters.perspective_view[1])
    top_view = dict(elev=figure_parameters.top_view[0],
                    azim=figure_parameters.top_view[1])
    views = [("perspective", "perspective view", perspective_view),
             ("top", "top view", top_view)]
    S = {
        "surf_3d": surf_3d, "surf_slices": surf_slices,
        "fams": fams, "ns": ns, "nzt_r": nzt_r,
        "nfp": nfp, "ntor": ntor,
        "lines": lines, "axx": axx, "axy": axy, "axz": axz,
        "coils": coils, "coil_radius": coil_radius,
        "pyvista_field_line_tube_radius":
            figure_parameters.pyvista_field_line_tube_radius_fraction *
            plasma_lim,
        "bmin": b_all.min(), "bmax": b_all.max(),
        "views": views,
        "figure_parameters": figure_parameters,
        "title": f"cuMES converged equilibrium — {name}",
        "lim": lim, "zlim": zlim,
        "box_aspect": (1.0, 1.0, zlim / lim),
    }
    base = output_base
    if args.backend == "pyvista":
        render_pyvista_figures(base, S)
        return

    jobs = [
        (render_single_view, (base, "perspective", "perspective view",
                              perspective_view, S)),
        (render_single_view, (base, "top", "top view",
                              top_view, S)),
        (render_combined, (base, S)),
        (render_slices, (base, S)),
    ]
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # no fork on this platform: spawn pickles S instead
        ctx = multiprocessing.get_context()
    procs = [ctx.Process(target=fn, args=args) for fn, args in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise SystemExit(f"error: {len(failed)} render process(es) failed "
                         f"(exit codes {failed})")
    print("all figures saved", flush=True)
