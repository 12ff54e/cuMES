#!/usr/bin/env python3
"""Check a cuMES result and compare physical diagnostics with Fortran VMEC wout."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from cumes_plot.equilibrium import eval_state, half_grid, make_profiles, solve_chip
from cumes_plot.state_io import load_state


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vmec_boundary(reference, ntor):
    """Fold signed-n VMEC R/Z coefficients into cuMES's eight real families."""
    mpol = int(reference["mpol"])
    result = {key: np.zeros(mpol * (ntor + 1)) for key in
              ("rmncc", "rmnss", "zmnsc", "zmncs", "rmnsc", "rmncs", "zmncc", "zmnss")}
    for k, (m, xn) in enumerate(zip(reference["xm"], reference["xn"])):
        m, n = int(m), int(round(xn / int(reference["nfp"])))
        i = m * (ntor + 1) + abs(n)
        for source, first, second, sine in (
                ("rmnc", "rmncc", "rmnss", False),
                ("zmns", "zmnsc", "zmncs", True),
                ("rmns", "rmnsc", "rmncs", True),
                ("zmnc", "zmncc", "zmnss", False)):
            value = reference[source][-1, k]
            if not sine or m:
                result[first][i] += value
            if n and (sine or m):
                result[second][i] += (-1 if sine else 1) * np.sign(n) * value
    return result


def reconstruct(families, ns, params):
    """Use the existing plot reconstruction for exported browser states."""
    if any(params.get(key, "power_series") != "power_series" for key in
           ("pmass_type", "pcurr_type", "piota_type")):
        raise ValueError("these fixtures require power-series profiles")
    theta = 2 * np.pi * np.arange(params["ntheta"]) / params["ntheta"]
    zeta = 2 * np.pi * np.arange(params["nzeta"]) / params["nzeta"]
    profiles = make_profiles(params, ns)
    volume, magnetic_energy = 0.0, 0.0
    iota = []
    valid = True
    inside = eval_state(families, ns, 0, theta, zeta, params["ntor"], params["nfp"])
    for jh in range(ns - 1):
        outside = eval_state(families, ns, jh + 1, theta, zeta,
                             params["ntor"], params["nfp"])
        chip = solve_chip(families, ns, jh, params, profiles)
        fields = half_grid([inside, outside], ns, jh, chip,
                           profiles["phip_avg"](jh), profiles["lamscale"])
        g = fields["gsqrt"]
        valid &= bool(np.all(np.isfinite(g)) and np.all(g < 0))
        iota.append(chip / profiles["phip_avg"](jh))
        volume -= np.mean(g)
        magnetic_energy -= np.mean(g * fields["bmag"]**2) / 2
        inside = outside
    factor = 4 * np.pi**2 / (ns - 1)
    return np.array(iota), volume * factor, magnetic_energy * factor, valid


def inspect_native(path):
    with h5py.File(path) as f:
        residuals = np.array([f[key][()] for key in
                              ("stage_fsqr", "stage_fsqz", "stage_fsql")])
        checks = dict(
            converged=int(f.attrs["status"]) == 0 and bool(np.all(f["stage_converged"][()])),
            residuals=bool(np.all(np.isfinite(residuals))
                           and np.all(residuals <= f["stage_ftol"][()])),
            finite_output=all(np.all(np.isfinite(f[key][()])) for key in f
                              if f[key].dtype.kind == "f"))
        g = f["sqrtg"][()]
        checks["oriented_jacobian"] = bool(np.all(g < 0))
        bu, bv = f["bsupu"][()], f["bsupv"][()]
        iota = np.mean(g * bu, axis=(1, 2)) / np.mean(g * bv, axis=(1, 2))
        b2 = bu * f["bsubu"][()] + bv * f["bsubv"][()]
        factor = 4 * np.pi**2 / g.shape[0]
        volume = -np.mean(g, axis=(1, 2)).sum() * factor
        energy = -np.mean(g * b2, axis=(1, 2)).sum() * factor / 2
        return checks, iota, volume, energy, f["stage_iterations"][()].tolist()


def compare(args):
    ns, _, families, params, name = load_state(args.cumes)
    with netcdf_file(args.vmec, mmap=False) as f:
        reference = {key: np.array(value.data) for key, value in f.variables.items()
                     if value.data.dtype.kind in "fiu"}
    for key, value in (("ns", ns), ("nfp", params["nfp"]),
                       ("mpol", params["mpol"]), ("ntor", params["ntor"])):
        if int(reference[key]) != value:
            raise ValueError(f"matched-resolution comparison requires matching {key}")
    checks = dict(asymmetric=len(families) == 12 and params.get("lasym") is True,
                  fixed_boundary=not params["lfreeb"]
                  and int(reference["lfreeb__logical__"]) == 0,
                  vmec_asymmetric=int(reference["lasym__logical__"]) == 1,
                  vmec_converged=int(reference["ier_flag"]) == 0
                  and all(0 <= float(reference[k]) <= float(reference["ftolv"])
                          for k in ("fsqr", "fsqz", "fsql")))
    epsilon = np.finfo(np.float32 if params["_precision"] == "float" else np.float64).eps
    if h5py.is_hdf5(args.cumes):
        native_checks, iota, volume, energy, iterations = inspect_native(args.cumes)
        checks.update(native_checks)
        diagnostics_source = "saved CUDA scientific fields"
    else:
        if not args.browser_report:
            raise ValueError("browser binary requires --browser-report for convergence evidence")
        browser = json.loads(args.browser_report.read_text())
        last = browser["plot"]["samples"][-1]
        completed = [row for row in browser["plot"]["samples"] if row["converged"]]
        stages = params["stages"]
        checks["browser_converged"] = (browser["dataset"]["cumesWebgpu"] == "pass"
                                        and last["converged"]
                                        and len(completed) == len(stages)
                                        and all(row["stage"] == i + 1
                                                and row["tolerance"] == stage["ftol"]
                                                and len(row["fsq"]) == 3
                                                and all(np.isfinite(x) and 0 <= x <= stage["ftol"]
                                                        for x in row["fsq"])
                                                for i, (row, stage) in enumerate(zip(completed, stages))))
        if browser["dataset"]["cumesPrecision"] == "double":
            epsilon = np.finfo(np.float32).eps**2
        iota, volume, energy, checks["reconstructed_jacobian"] = reconstruct(families, ns, params)
        checks["finite_state"] = all(np.isfinite(a).all() for a in families.values())
        iterations = [row["iteration"] for row in completed]
        diagnostics_source = "reconstructed from exported browser state"
    boundary = vmec_boundary(reference, params["ntor"])
    edge_error = max(float(np.max(np.abs(families[k].reshape(-1, ns)[:, -1] - v)))
                     for k, v in boundary.items())
    edge_tolerance = 16 * epsilon * max(1, np.max(np.abs(boundary["rmncc"])))
    checks["fixed_lcfs"] = edge_error <= edge_tolerance
    if args.replay:
        replay_checks, _, _, _, replay_iterations = inspect_native(args.replay)
        checks["checkpoint_replay"] = all(replay_checks.values()) and sum(replay_iterations) <= 2
    reference_iota = reference["iotas"][1:]
    sh = (np.arange(ns - 1) + 0.5) / (ns - 1)
    angles = 2 * np.pi * np.arange(256) / 256
    phase = np.arange(params["ntor"] + 1)[:, None] * angles
    cosine, sine = np.cos(phase), np.sin(phase)
    axis_family = lambda key: families[key].reshape(-1, ns)[:params["ntor"] + 1, 0]
    axis_r = axis_family("rmncc") @ cosine + axis_family("rmncs") @ sine
    axis_z = axis_family("zmncc") @ cosine + axis_family("zmncs") @ sine
    reference_r = reference["raxis_cc"] @ cosine - reference["raxis_cs"] @ sine
    reference_z = reference["zaxis_cc"] @ cosine - reference["zaxis_cs"] @ sine
    # VMEC's wb uses its normalized angular weights (bcovar.f); restore
    # the full-torus angular factor to compare mu0 times magnetic energy.
    reference_energy = float(reference["wb"]) * 4 * np.pi**2
    checks = {key: bool(value) for key, value in checks.items()}
    record = dict(cumes=str(args.cumes.resolve()), vmec=str(args.vmec.resolve()),
                  cumes_sha256=digest(args.cumes), vmec_sha256=digest(args.vmec),
                  diagnostics_source=diagnostics_source, checks=checks,
                  iterations=iterations, max_boundary_error_m=edge_error,
                  boundary_tolerance_m=float(edge_tolerance),
                  volume_m3=float(volume), vmec_volume_m3=float(reference["volume_p"]),
                  relative_volume_error=float(volume / reference["volume_p"] - 1),
                  magnetic_energy_mu0=float(energy), vmec_magnetic_energy_mu0=reference_energy,
                  relative_magnetic_energy_error=float(energy / reference_energy - 1),
                  max_axis_distance_m=float(np.max(np.hypot(axis_r - reference_r,
                                                            axis_z - reference_z))),
                  iota_max_absolute_error=float(np.max(np.abs(iota - reference_iota))),
                  iota_axis_adjacent=float(iota[0]), vmec_iota_axis_adjacent=float(reference_iota[0]),
                  half_flux=sh.tolist(), iota=iota.tolist(), vmec_iota=reference_iota.tolist())
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axes[0].plot(sh, reference_iota, "k--", label="Fortran VMEC")
    axes[0].plot(sh, iota, color="#2466aa", label="cuMES")
    axes[0].set(xlabel="Normalized toroidal flux s", ylabel="Rotational transform ι")
    axes[0].legend()
    theta = 2 * np.pi * np.arange(512) / 512
    nz = max(48, 4 * params["ntor"] + 4)
    zeta = 2 * np.pi * np.arange(nz) / nz
    for axis, plane in zip(axes[1:], (0, nz // 3)):
        phase = reference["xm"][:, None] * theta - reference["xn"][:, None] * zeta[plane] / params["nfp"]
        for j in sorted({int(round(s * (ns - 1))) for s in (0.1, 0.25, 0.5, 0.75, 1)}):
            r = reference["rmnc"][j] @ np.cos(phase) + reference["rmns"][j] @ np.sin(phase)
            z = reference["zmns"][j] @ np.sin(phase) + reference["zmnc"][j] @ np.cos(phase)
            axis.plot(np.r_[r, r[0]], np.r_[z, z[0]], "k--", lw=1)
            state = eval_state(families, ns, j, theta, zeta, params["ntor"], params["nfp"])
            r = (state["re"] + np.sqrt(j / (ns - 1)) * state["ro"])[:, plane]
            z = (state["ze"] + np.sqrt(j / (ns - 1)) * state["zo"])[:, plane]
            axis.plot(np.r_[r, r[0]], np.r_[z, z[0]], color="#2466aa", lw=0.8)
        axis.set(xlabel="R (m)", ylabel="Z (m)",
                 title=f"φ = {np.degrees(zeta[plane] / params['nfp']):.1f}°")
        axis.set_aspect("equal")
    figure.suptitle(f"{args.title or name} · ns={ns}, mpol={params['mpol']}, ntor={params['ntor']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(args.out) + ".png", dpi=180)
    plt.close(figure)
    Path(str(args.out) + ".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in
                      ("half_flux", "iota", "vmec_iota")}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cumes", type=Path, required=True)
    parser.add_argument("--vmec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--browser-report", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--title")
    raise SystemExit(compare(parser.parse_args()))
