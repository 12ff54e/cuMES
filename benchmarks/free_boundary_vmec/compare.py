#!/usr/bin/env python3
"""Compare a native free-boundary result with original Fortran VMEC."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asymmetric_vmec.compare import digest, eval_state, inspect_native, load_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cumes", type=Path, required=True)
    parser.add_argument("--vmec", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    ns, _, families, params, _ = load_state(args.cumes)
    checks, iota, volume, energy, iterations = inspect_native(args.cumes)
    with netcdf_file(args.vmec, mmap=False) as f:
        ref = {
            k: np.array(v.data)
            for k, v in f.variables.items()
            if v.data.dtype.kind in "fiu"
        }
    for key, value in (
        ("ns", ns),
        ("nfp", params["nfp"]),
        ("mpol", params["mpol"]),
        ("ntor", params["ntor"]),
    ):
        if int(ref[key]) != value:
            raise ValueError(f"comparison requires matching {key}")
    checks["free_boundary"] = bool(params["lfreeb"] and ref["lfreeb__logical__"])
    checks["symmetry_matches"] = params.get("lasym", False) == bool(
        ref["lasym__logical__"]
    )
    checks["vmec_converged"] = int(ref["ier_flag"]) == 0 and all(
        np.isfinite(ref[k]) and 0 <= float(ref[k]) <= float(ref["ftolv"])
        for k in ("fsqr", "fsqz", "fsql")
    )
    if args.replay:
        replay, replay_iota, replay_volume, _, replay_iterations = inspect_native(
            args.replay
        )
        checks["checkpoint_replay"] = all(replay.values())
    for key, like in (("rmns", "rmnc"), ("zmnc", "zmns")):
        if key not in ref:
            ref[key] = np.zeros_like(ref[like])
    sh = (np.arange(ns - 1) + 0.5) / (ns - 1)
    ref_iota = ref["iotas"][1:]
    report = dict(
        cumes=str(args.cumes.resolve()),
        vmec=str(args.vmec.resolve()),
        cumes_sha256=digest(args.cumes),
        vmec_sha256=digest(args.vmec),
        checks={k: bool(v) for k, v in checks.items()},
        iterations=iterations,
        volume_m3=float(volume),
        vmec_volume_m3=float(ref["volume_p"]),
        relative_volume_error=float(volume / ref["volume_p"] - 1),
        relative_magnetic_energy_error=float(energy / (ref["wb"] * 4 * np.pi**2) - 1),
        iota_max_absolute_error=float(np.max(abs(iota - ref_iota))),
        half_flux=sh.tolist(),
        iota=iota.tolist(),
        vmec_iota=ref_iota.tolist(),
    )
    if args.replay:
        report.update(
            replay_iterations=replay_iterations,
            replay_iota_max_change=float(np.max(abs(replay_iota - iota))),
            replay_relative_volume_change=float(replay_volume / volume - 1),
        )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    axes[0].plot(sh, ref_iota, "k--", label="Fortran VMEC")
    axes[0].plot(sh, iota, color="#2466aa", label="cuMES CUDA double")
    axes[0].set(xlabel="Normalized toroidal flux s", ylabel="Rotational transform ι")
    axes[0].legend(fontsize=9)
    theta = 2 * np.pi * np.arange(1024) / 1024
    nz = max(48, 4 * params["ntor"] + 4)
    zeta = 2 * np.pi * np.arange(nz) / nz
    for axis, plane in zip(axes[1:], (0, nz // 3)):
        if params["ntor"] == 0 and plane:
            axis.plot(sh, iota - ref_iota, color="#2466aa")
            axis.axhline(0, color="black", lw=0.7)
            axis.set(
                xlabel="Normalized toroidal flux s",
                ylabel="cuMES − VMEC iota",
                title="Profile difference",
            )
            continue
        phase = (
            ref["xm"][:, None] * theta
            - ref["xn"][:, None] * zeta[plane] / params["nfp"]
        )
        for j in sorted({round(s * (ns - 1)) for s in (0.1, 0.25, 0.5, 0.75, 1)}):
            r = ref["rmnc"][j] @ np.cos(phase) + ref["rmns"][j] @ np.sin(phase)
            z = ref["zmns"][j] @ np.sin(phase) + ref["zmnc"][j] @ np.cos(phase)
            axis.plot(np.r_[r, r[0]], np.r_[z, z[0]], "k--", lw=1)
            state = eval_state(
                families, ns, j, theta, zeta, params["ntor"], params["nfp"]
            )
            r = (state["re"] + np.sqrt(j / (ns - 1)) * state["ro"])[:, plane]
            z = (state["ze"] + np.sqrt(j / (ns - 1)) * state["zo"])[:, plane]
            axis.plot(np.r_[r, r[0]], np.r_[z, z[0]], color="#2466aa", lw=0.8)
        axis.set(
            xlabel="R (m)",
            ylabel="Z (m)",
            title=f"φ = {np.degrees(zeta[plane] / params['nfp']):.1f}°",
            aspect="equal",
        )
    status = "converged" if all(checks.values()) else "NOT QUALIFIED"
    figure.suptitle(f"{args.title} · free boundary · {status}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(args.out) + ".png", dpi=180)
    figure.savefig(str(args.out) + ".pdf")
    plt.close(figure)
    Path(str(args.out) + ".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("half_flux", "iota", "vmec_iota")
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
