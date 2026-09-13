#!/usr/bin/env python3
"""Summarize completed QH experiments without treating missing runs as passes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

from compare import digest, inspect_native, load_state
from gauge_study import CASES, THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for name in [
        *CASES,
        "vmec-base",
        "vmec-radial101",
        "vmec-radial201",
        "vmec-angular",
        "vmec-quadrature",
    ]:
        directory = args.root / name
        path = directory / ("wout_qh.nc" if name.startswith("vmec-") else "output.h5")
        if not path.exists():
            if (directory / "run.log").exists():
                records[name] = dict(
                    converged=False,
                    output_present=False,
                    log_sha256=digest(directory / "run.log"),
                )
            continue
        if name.startswith("vmec-"):
            with netcdf_file(path, mmap=False) as f:
                value = lambda key: np.array(f.variables[key].data)
                iota = value("iotas")[1:]
                residuals = [float(value(k)) for k in ("fsqr", "fsqz", "fsql")]
                tolerance = float(value("ftolv"))
                checks = dict(
                    status=int(value("ier_flag")) == 0,
                    residuals=all(
                        np.isfinite(x) and 0 <= x <= tolerance for x in residuals
                    ),
                )
                grid = {k: int(value(k)) for k in ("ns", "mpol", "ntor", "nfp")}
                volume, iterations = float(value("volume_p")), None
        else:
            checks, iota, volume, _, iterations = inspect_native(path)
            ns, _, _, params, _ = load_state(path)
            grid = dict(
                ns=ns,
                **{k: params[k] for k in ("mpol", "ntor", "ntheta", "nzeta", "nfp")},
            )
        s = (np.arange(len(iota)) + 0.5) / len(iota)
        record = dict(
            converged=all(checks.values()),
            checks=checks,
            grid=grid,
            volume_m3=volume,
            iterations=iterations,
            output_sha256=digest(path),
            half_flux=s.tolist(),
            iota=iota.tolist(),
            iota_at_flux={
                str(x): float(np.interp(x, s, iota))
                for x in (0.01, 0.025, 0.05, 0.1, 0.5)
            },
        )
        if not name.startswith("vmec-"):
            record["gauge_threshold"] = THRESHOLDS[name.split("-", 1)[0]]
        records[name] = record
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

    def curve(axis, name, label, **kwargs):
        record = records.get(name)
        if record and record["converged"]:
            flux = np.asarray(record["half_flux"])
            visible = np.r_[flux[flux < 0.12], 0.12]
            axis.plot(
                visible, np.interp(visible, flux, record["iota"]), label=label, **kwargs
            )

    for power in (10, 12, 14, 16):
        curve(
            axes[0, 0],
            "default-cold" if power == 16 else f"default-tol{power}",
            f"FTOL = 10⁻{str(power).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}",
        )
    axes[0, 0].set(title="Stopping tolerance · same ns=51 startup", xlim=(0, 0.12))
    for name, label in (
        ("default-cold", "default gate, DELT=.9"),
        ("default-cold-step05", "default gate, DELT=.5"),
        ("later-cold", "later freeze, DELT=.9"),
        ("later-cold-step05", "later freeze, DELT=.5"),
    ):
        curve(axes[0, 1], name, label)
    curve(axes[0, 1], "vmec-base", "Fortran VMEC", color="black", linestyle="--")
    axes[0, 1].set(title="Gauge/history sensitivity · FTOL=1e-16", xlim=(0, 0.12))
    for name, label in (
        ("default-cold", "cuMES ns=51"),
        ("default-radial101", "cuMES ns=101"),
        ("default-radial201", "cuMES ns=201"),
        ("fixed-warm-radial201", "ns=201, warm/frozen gauge"),
    ):
        curve(axes[1, 0], name, label)
    curve(axes[1, 0], "vmec-radial101", "Fortran ns=101", color="black", linestyle="--")
    axes[1, 0].set(title="Radial refinement · compare at common flux", xlim=(0, 0.12))
    for name, label in (
        ("default-cold", "mpol=8, ntor=12"),
        ("default-quadrature", "same modes, 40×80 quadrature"),
        ("default-angular", "mpol=10, ntor=15"),
        ("default-angular12", "mpol=12, ntor=15"),
    ):
        curve(axes[1, 1], name, label)
    curve(
        axes[1, 1],
        "vmec-angular",
        "Fortran mpol=10, ntor=15",
        color="black",
        linestyle="--",
    )
    axes[1, 1].set(title="Angular refinement · ns=51", xlim=(0, 0.12))
    for axis in axes.flat:
        axis.set(xlabel="Normalized toroidal flux s", ylabel="Rotational transform ι")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        "QH: residual convergence does not establish a resolved near-axis profile"
    )
    fig.savefig(str(args.out) + ".png", dpi=180)
    fig.savefig(str(args.out) + ".pdf")
    plt.close(fig)
    print(f"Wrote {len(records)} run records and figures to {args.out}")


if __name__ == "__main__":
    main()
