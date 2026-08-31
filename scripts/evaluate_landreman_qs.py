#!/usr/bin/env python3
"""Evaluate the archived Landreman-Paul QS metric directly from a VMEC wout."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import netcdf_file


REQUIRED_VARIABLES = (
    "aspect", "nfp", "phi", "iotas", "bvco", "buco", "xm_nyq",
    "xn_nyq", "gmnc", "bmnc", "bsubumnc", "bsubvmnc", "bsupumnc",
    "bsupvmnc",
)


def _scalar(variable):
    return np.asarray(variable.data).item()


def _interpolate_half_grid(values, surfaces):
    """Match the archived scipy interp1d linear/extrapolate operation."""
    values = np.asarray(values, dtype=float)
    ns = values.shape[0]
    if ns < 3:
        raise ValueError("wout requires at least 3 radial grid points")
    half_grid = (np.arange(1, ns, dtype=float) - 0.5) / (ns - 1)
    return interp1d(
        half_grid, values[1:], axis=0, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )(surfaces)


def evaluate(wout_path, *, m, n, surfaces, weights, ntheta=63, nphi=64):
    """Return the no-Boozer QuasisymmetryRatioError components.

    ``n`` is the per-field-period integer accepted by the archived SIMSOPT
    class. It is multiplied by ``nfp`` below.
    """
    surfaces = np.asarray(surfaces, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if surfaces.ndim != 1 or weights.shape != surfaces.shape:
        raise ValueError("surfaces and weights must be equal-length vectors")
    if np.any((surfaces < 0.0) | (surfaces > 1.0)):
        raise ValueError("normalized toroidal-flux surfaces must lie in [0,1]")
    if np.any(weights < 0.0):
        raise ValueError("surface weights must be nonnegative")

    with netcdf_file(str(wout_path), "r", mmap=False) as wout:
        missing = [name for name in REQUIRED_VARIABLES
                   if name not in wout.variables]
        if missing:
            raise ValueError("wout is missing variables: " + ", ".join(missing))
        if "lasym__logical__" in wout.variables and bool(
                _scalar(wout.variables["lasym__logical__"])):
            raise ValueError("the archived metric supports stellarator symmetry only")

        nfp = int(_scalar(wout.variables["nfp"]))
        aspect = float(_scalar(wout.variables["aspect"]))
        phi = np.array(wout.variables["phi"].data, dtype=float, copy=True)
        xm = np.array(wout.variables["xm_nyq"].data, dtype=float, copy=True)
        xn = np.array(wout.variables["xn_nyq"].data, dtype=float, copy=True)
        radial = {
            name: _interpolate_half_grid(
                np.array(wout.variables[name].data, dtype=float, copy=True),
                surfaces)
            for name in (
                "iotas", "bvco", "buco", "gmnc", "bmnc", "bsubumnc",
                "bsubvmnc", "bsupumnc", "bsupvmnc",
            )
        }

    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = (2.0 * np.pi / nfp) * np.arange(nphi) / nphi
    angle = (xm[:, None, None] * theta[None, :, None]
             - xn[:, None, None] * zeta[None, None, :])
    cosine = np.cos(angle)
    sine = np.sin(angle)
    shape = (surfaces.size, ntheta, nphi)

    fields = {
        name: np.zeros(shape)
        for name in ("modb", "d_b_d_theta", "d_b_d_zeta", "sqrtg",
                     "bsubu", "bsubv", "bsupu", "bsupv")
    }
    coefficient_names = {
        "modb": "bmnc",
        "sqrtg": "gmnc",
        "bsubu": "bsubumnc",
        "bsubv": "bsubvmnc",
        "bsupu": "bsupumnc",
        "bsupv": "bsupvmnc",
    }
    # Keep the archived mode-by-mode accumulation order for a tight numerical
    # comparison with its logged values.
    for mode in range(xm.size):
        for field, coefficients in coefficient_names.items():
            fields[field] += radial[coefficients][:, mode, None, None] * cosine[mode]
        fields["d_b_d_theta"] += (
            -xm[mode] * radial["bmnc"][:, mode, None, None] * sine[mode])
        fields["d_b_d_zeta"] += (
            xn[mode] * radial["bmnc"][:, mode, None, None] * sine[mode])

    b_dot_grad_b = (
        fields["bsupu"] * fields["d_b_d_theta"]
        + fields["bsupv"] * fields["d_b_d_zeta"])
    d_psi_d_s = -phi[-1] / (2.0 * np.pi)
    b_cross_grad_b_dot_grad_psi = d_psi_d_s * (
        fields["bsubu"] * fields["d_b_d_zeta"]
        - fields["bsubv"] * fields["d_b_d_theta"]
    ) / fields["sqrtg"]

    dtheta = 2.0 * np.pi / ntheta
    dphi = (2.0 * np.pi / nfp) / nphi
    volume_prime = nfp * dtheta * dphi * np.sum(
        fields["sqrtg"], axis=(1, 2))
    physical_n = n * nfp
    numerator = (
        b_cross_grad_b_dot_grad_psi
        * (physical_n - radial["iotas"][:, None, None] * m)
        - b_dot_grad_b
        * (m * radial["bvco"][:, None, None]
           + physical_n * radial["buco"][:, None, None]))
    quadrature = (nfp * dtheta * dphi / volume_prime[:, None, None]
                  * fields["sqrtg"])
    if np.any(quadrature < 0.0):
        raise ValueError("wout has inconsistent Jacobian orientation")
    unweighted_residuals = (
        np.sqrt(quadrature) * numerator / fields["modb"] ** 3)
    unweighted_profile = np.sum(
        unweighted_residuals * unweighted_residuals, axis=(1, 2))
    profile = weights * unweighted_profile

    return {
        "wout": str(Path(wout_path)),
        "nfp": nfp,
        "helicity_m": int(m),
        "helicity_n_per_period": int(n),
        "helicity_n": int(physical_n),
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "surfaces": surfaces.tolist(),
        "weights": weights.tolist(),
        "aspect_ratio": aspect,
        "qs_profile": profile.tolist(),
        "qs_total": float(np.sum(profile)),
        "unweighted_qs_profile": unweighted_profile.tolist(),
        "unweighted_qs_total": float(np.sum(unweighted_profile)),
    }


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout", type=Path)
    parser.add_argument("--case", choices=("qa", "qh"), required=True)
    parser.add_argument("--weight-end", type=float)
    parser.add_argument("--target-aspect", type=float)
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    defaults = {
        "qa": {"n": 0, "weight_end": 30.0, "aspect": 6.0},
        "qh": {"n": -1, "weight_end": 2.0, "aspect": 8.0},
    }[arguments.case]
    weight_end = (defaults["weight_end"] if arguments.weight_end is None
                  else arguments.weight_end)
    target_aspect = (defaults["aspect"] if arguments.target_aspect is None
                     else arguments.target_aspect)
    surfaces = np.linspace(0.0, 1.0, 11)
    result = evaluate(
        arguments.wout, m=1, n=defaults["n"], surfaces=surfaces,
        weights=np.linspace(1.0, weight_end, surfaces.size))
    result["target_aspect_ratio"] = target_aspect
    result["aspect_residual"] = result["aspect_ratio"] - target_aspect
    result["objective"] = (
        result["qs_total"] + result["aspect_residual"] ** 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
