#!/usr/bin/env python3
"""Render 3D figures of the converged W7-X equilibrium (cuMES state).

Reads the legacy binary v0 state container (docs/output-formats.md §3),
reconstructs real-space geometry with the solver's exact conventions
(parity-split e/o arrays, odd-m scalxc regularization, staggered half-grid
metric — src/geometry_impl.cuh), solves the ncurr=1 current constraint for
chi' per plotted surface (ncurr1FinalizeKernel), and renders the full
5-period torus colored by |B| with light-source shading.

One PNG file per view: <out>_perspective.png and <out>_top.png. The
magnetic axis is the converged axis extracted from the state (m=0 content
of the innermost surface, seed_state.hpp convention: nfp-fold modulation).

Two self-checks run before rendering:
  1. the state LCFS (j = ns-1) must match the input rbc/zbs boundary;
  2. the edge |B| must land in the physical W7-X range.

Usage: plot_w7x.py [--state PATH] [--input PATH] [--out PATH.png]
       [--field-lines]
"""

import argparse
import json
import os
import struct

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

MPOL, NTOR, NFP = 12, 12, 5
NTHETA, NZETA = 30, 36          # the solver's angular grid (2*mpol+6, input)
NTHETA_RED = NTHETA // 2 + 1    # reduced-grid theta count (forward quadrature)
SIGN_J = -1.0
MU0 = 4.0e-7 * np.pi

# W7-X profile data (inputs/w7x.json)
PHIEDGE = -1.74
APHI = [1.0]
AC = [0.0, 1.0]
CURTOR = 5000.0


def load_state(path):
    """Load the v0 container: ns, mnmax, six mode-major families."""
    with open(path, "rb") as f:
        ns, mnmax = struct.unpack("<ii", f.read(8))
        n = ns * mnmax
        fams = {name: np.frombuffer(f.read(8 * n), dtype="<f8")
                for name in ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")}
    return ns, mnmax, fams


def mode_tables(mnmax):
    m = np.arange(mnmax) // (NTOR + 1)
    n = np.arange(mnmax) % (NTOR + 1)
    return m, n


def eval_state(fams, ns, j, th, zt):
    """Stored parity arrays at full-grid surface j on an arbitrary (th, zt)
    grid — the exact inverseDFT convention (fourier_impl.cuh):

      R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
      Z = zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
      λ = lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ)

    even m -> e arrays (plain), odd m -> o arrays divided by
    maxsc = max(sqrt(s_j), sqrt(ds)) (vmecpp's scalxc decomposition).
    The λ ζ-derivative slot stores -dλ/dζ (signV = -1). Derivatives are the
    analytic per-radian ones, matching the basis-table accumulation.
    """
    m, n = mode_tables(fams["rmncc"].shape[0] // ns)
    nf = n * NFP
    odd = (m % 2 == 1)
    maxsc = max(np.sqrt(j / (ns - 1)), np.sqrt(1.0 / (ns - 1)))
    fac = np.where(odd, 1.0 / maxsc, 1.0)
    assert th.ndim == 1 and zt.ndim == 1, "eval_state takes 1-D angle axes"

    def fam(name):
        mode = m * (NTOR + 1) + n     # folded mode index
        return fams[name][mode * ns + j]  # mode-major, surface j column

    rc, rs = fam("rmncc"), fam("rmnss")
    zs, zc = fam("zmnsc"), fam("zmncs")
    lsc, lcs = fam("lmnsc"), fam("lmncs")

    TH, ZT = np.meshgrid(th, zt, indexing="ij")
    mT = m[:, None, None]
    nfT = nf[:, None, None]
    cT = np.cos(mT * TH[None]) * np.cos(n[:, None, None] * ZT[None])
    sT = np.sin(mT * TH[None]) * np.sin(n[:, None, None] * ZT[None])
    sCz = np.sin(mT * TH[None]) * np.cos(n[:, None, None] * ZT[None])
    cSz = np.cos(mT * TH[None]) * np.sin(n[:, None, None] * ZT[None])

    def acc(t):
        w = t * fac[:, None, None]
        return w[~odd].sum(0), w[odd].sum(0)

    out = {"th": th, "zt": zt}
    out["re"], out["ro"] = acc(rc[:, None, None] * cT + rs[:, None, None] * sT)
    out["rue"], out["ruo"] = acc(-mT * rc[:, None, None] * sCz
                                 + mT * rs[:, None, None] * cSz)
    out["rve"], out["rvo"] = acc(-nfT * rc[:, None, None] * cSz
                                 + nfT * rs[:, None, None] * sCz)
    out["ze"], out["zo"] = acc(zs[:, None, None] * sCz + zc[:, None, None] * cSz)
    out["zue"], out["zuo"] = acc(mT * zs[:, None, None] * cT
                                 - mT * zc[:, None, None] * sT)
    out["zve"], out["zvo"] = acc(-nfT * zs[:, None, None] * sT
                                 + nfT * zc[:, None, None] * cT)
    out["le"], out["lo"] = acc(lsc[:, None, None] * sCz + lcs[:, None, None] * cSz)
    out["lue"], out["luo"] = acc(mT * lsc[:, None, None] * cT
                                 - mT * lcs[:, None, None] * sT)
    out["lve"], out["lvo"] = acc(nfT * lsc[:, None, None] * sT
                                 - nfT * lcs[:, None, None] * cT)
    return out


def half_grid(arrays, ns, jh, chip):
    """Mirror of baseGeometryKernel + magneticFieldKernel + the per-surface
    chi' solve for half-grid surface jh (geometry_impl.cuh). arrays = the
    two adjacent full-grid surfaces (stored parity values). Returns the
    half-grid geometry (r12/z12), the covariant metric, lambda derivatives,
    and |B| (Tesla — the B^u·B_u products are invariant under the angular
    coordinate scaling, so the kernel units are physical)."""
    ds = 1.0 / (ns - 1)
    s_in = np.sqrt(jh / (ns - 1))
    s_out = np.sqrt((jh + 1) / (ns - 1))
    s_h = np.sqrt((jh + 0.5) / (ns - 1))
    r, rp = arrays[0], arrays[1]

    # ---- base geometry (stored parity values, exactly as the kernel) ----
    r12 = 0.5 * ((r["re"] + rp["re"]) + s_h * (r["ro"] + rp["ro"]))
    z12 = 0.5 * ((r["ze"] + rp["ze"]) + s_h * (r["zo"] + rp["zo"]))
    ru12 = 0.5 * ((r["rue"] + rp["rue"]) + s_h * (r["ruo"] + rp["ruo"]))
    zu12 = 0.5 * ((r["zue"] + rp["zue"]) + s_h * (r["zuo"] + rp["zuo"]))
    rs_ = ((rp["re"] - r["re"]) + s_h * (rp["ro"] - r["ro"])) / ds
    zs_ = ((rp["ze"] - r["ze"]) + s_h * (rp["zo"] - r["zo"])) / ds

    tau1 = ru12 * zs_ - rs_ * zu12
    tau2 = (rp["ruo"] * rp["zo"] + r["ruo"] * r["zo"]
            - rp["zuo"] * rp["ro"] - r["zuo"] * r["ro"]
            + (rp["rue"] * rp["zo"] + r["rue"] * r["zo"]
               - rp["zue"] * rp["ro"] - r["zue"] * r["ro"]) / s_h)
    tau = tau1 + 0.25 * tau2
    gsqrt = tau * r12

    sfi2, sfo2 = s_in * s_in, s_out * s_out
    guu = 0.5 * ((r["rue"] ** 2 + r["zue"] ** 2) + (rp["rue"] ** 2 + rp["zue"] ** 2)
                 + sfi2 * (r["ruo"] ** 2 + r["zuo"] ** 2)
                 + sfo2 * (rp["ruo"] ** 2 + rp["zuo"] ** 2)) \
        + s_h * ((r["rue"] * r["ruo"] + r["zue"] * r["zuo"])
                 + (rp["rue"] * rp["ruo"] + rp["zue"] * rp["zuo"]))
    # NOTE: the sH-weighted cross terms sit INSIDE the 0.5, matching vmecpp's
    # ComputeMetricElements exactly (metric_kernel.h) — moving them outside
    # doubles the cross-term weight (caught historically by the iter-1 dump).
    guv = 0.5 * ((r["rue"] * r["rve"] + r["zue"] * r["zve"])
                 + (rp["rue"] * rp["rve"] + rp["zue"] * rp["zve"])
                 + sfi2 * (r["ruo"] * r["rvo"] + r["zuo"] * r["zvo"])
                 + sfo2 * (rp["ruo"] * rp["rvo"] + rp["zuo"] * rp["zvo"])
                 + s_h * ((r["rue"] * rp["rvo"] + r["zue"] * rp["zvo"])
                          + (rp["rue"] * rp["rvo"] + rp["zue"] * rp["zvo"])
                          + (r["rve"] * r["ruo"] + r["zve"] * r["zuo"])
                          + (rp["rve"] * rp["ruo"] + rp["zve"] * rp["zuo"])))
    gvv = 0.5 * (r["re"] ** 2 + rp["re"] ** 2
                 + sfi2 * r["ro"] ** 2 + sfo2 * rp["ro"] ** 2) \
        + s_h * (r["re"] * r["ro"] + rp["re"] * rp["ro"]) \
        + 0.5 * ((r["rve"] ** 2 + r["zve"] ** 2) + (rp["rve"] ** 2 + rp["zve"] ** 2)
                 + sfi2 * (r["rvo"] ** 2 + r["zvo"] ** 2)
                 + sfo2 * (rp["rvo"] ** 2 + rp["zvo"] ** 2)) \
        + s_h * ((r["rve"] * r["rvo"] + r["zve"] * r["zvo"])
                 + (rp["rve"] * rp["rvo"] + rp["zve"] * rp["zvo"]))

    # ---- magnetic field (magneticFieldKernel) ----
    lu_h = 0.5 * ((r["lue"] + rp["lue"]) + s_h * (r["luo"] + rp["luo"]))
    lv_h = 0.5 * ((r["lve"] + rp["lve"]) + s_h * (r["lvo"] + rp["lvo"]))
    phip = SIGN_J * PHIEDGE / (2.0 * np.pi) / sum(APHI)   # torflux(1) = 1
    lamscale = abs(phip) * np.sqrt((ns - 1) * ds)          # = sqrt(ds·Σ phipH²)
    inv = np.zeros_like(gsqrt)
    np.divide(1.0, gsqrt, out=inv,
              where=np.isfinite(gsqrt) & (np.abs(gsqrt) > 1e-30))
    bsupv = (lamscale * lu_h + phip) * inv                 # phi' constant
    bsupu = lamscale * lv_h * inv + chip * inv             # chi' from the solve
    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv
    bsq = bsupu * bsubu + bsupv * bsubv
    return {"r12": r12, "z12": z12, "guu": guu, "guv": guv, "gvv": gvv,
            "gsqrt": gsqrt, "lu_h": lu_h, "lv_h": lv_h, "bsupu": bsupu,
            "bsupv": bsupv, "bmag": np.sqrt(np.maximum(bsq, 0.0))}


def solve_chip(fams, ns, jh):
    """ncurr1FinalizeKernel: chi' = (currH - Σ(guu·B^θ_λ + guv·B^ζ)w) /
    Σ(guu/√g·w), summed over the reduced-theta trapezoid subset with
    dnorm3 = 1/(nzeta·(nThetaRed-1)) (the exact kernel quadrature)."""
    it = np.arange(NTHETA_RED)
    iz = np.arange(NZETA)
    th = 2.0 * np.pi * it / NTHETA
    zt = 2.0 * np.pi * iz / NZETA
    a = [eval_state(fams, ns, j, th, zt) for j in (jh, jh + 1)]
    h = half_grid(a, ns, jh, 0.0)   # chip = 0: the lambda-only B^θ
    w = np.full(NTHETA_RED, 1.0 / (NZETA * (NTHETA_RED - 1)))
    w[0] *= 0.5
    w[-1] *= 0.5
    jv = np.sum(w[:, None] * (h["guu"] * h["bsupu"] + h["guv"] * h["bsupv"]))
    one_over = np.zeros_like(h["gsqrt"])
    np.divide(1.0, h["gsqrt"], out=one_over, where=np.abs(h["gsqrt"]) > 1e-30)
    avg = np.sum(w[:, None] * h["guu"] * one_over)
    i1 = sum(AC[i] / (i + 1) for i in range(len(AC)))
    itor = SIGN_J * MU0 * CURTOR / (2.0 * np.pi * i1)
    s_h = np.sqrt((jh + 0.5) / (ns - 1))
    curr = itor * sum(AC[i] * s_h ** (i + 1) / (i + 1) for i in range(len(AC)))
    return (curr - jv) / avg if avg != 0.0 else 0.0


def boundary_from_input(path, th, zt):
    """Boundary R/Z from the input rbc/zbs (signed-n VMEC convention)."""
    with open(path) as f:
        doc = json.load(f)
    R = np.zeros_like(th)
    Z = np.zeros_like(th)
    for c in doc["rbc"]:
        R += c["value"] * np.cos(c["m"] * th - c["n"] * zt)
    for c in doc["zbs"]:
        Z += c["value"] * np.sin(c["m"] * th - c["n"] * zt)
    return R, Z


def periodic_close(A):
    """Close a 2-D periodic grid: append the first row (theta = 2pi) and the
    first column (zeta = 2pi) so the rendered surface mesh has no seam."""
    A = np.concatenate([A, A[:1, :]], axis=0)
    A = np.concatenate([A, A[:, :1]], axis=1)
    return A


def surface_intensity(X, Y, Z, light_dir, lo=0.30):
    """Lambertian illumination from the GEOMETRIC surface normals (cross
    product of the grid tangents), rescaled to [lo, 1] so the side facing
    away from the light stays bright. Unlike LightSource.hillshade over the
    |B| scalar this produces real 3-D relief: the |B| heightfield is so
    smooth that its normal-intensity range collapses and (with fraction<1)
    the whole map clips to black."""
    gx1, gy1, gz1 = np.gradient(X, axis=0), np.gradient(Y, axis=0), np.gradient(Z, axis=0)
    gx2, gy2, gz2 = np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)
    nx = gy1 * gz2 - gz1 * gy2
    ny = gz1 * gx2 - gx1 * gz2
    nz = gx1 * gy2 - gy1 * gx2
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = (nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]) / mag
    lam = (lam - lam.min()) / (lam.max() - lam.min())
    return lo + (1.0 - lo) * lam


def to_cartesian(R, Z, n_periods=NFP):
    """Replicate one field period into the full torus (phi = zeta/nfp),
    closing both periodic grid directions."""
    nz = R.shape[1]
    zt = np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False)
    Rfull = np.concatenate([R] * n_periods, axis=1)
    Zfull = np.concatenate([Z] * n_periods, axis=1)
    phi = np.concatenate([(zt + 2.0 * np.pi * k) / NFP for k in range(n_periods)])
    Rfull = periodic_close(Rfull)
    Zfull = periodic_close(Zfull)
    phi = np.append(phi, 2.0 * np.pi)
    return Rfull * np.cos(phi[None, :]), Rfull * np.sin(phi[None, :]), Zfull


def converged_axis(fams, ns, n=240):
    """The converged magnetic axis: the m=0 content of the innermost
    computed surface (j=1). The axis is theta-independent by construction
    (only m=0 modes enter), so the curve runs through the center of every
    cross-section and cannot leave the plasma:

        R_ax(zeta) = sum_n rmncc(0, n, j=1) cos(n zeta)
        Z_ax(zeta) = sum_n zmncs(0, n, j=1) sin(n zeta)

    with zeta the per-field-period angle, replicated nfp times around the
    full torus (phi = zeta/nfp). The W7-X axis carries genuine nfp-fold
    modulation (outboard at the beans, inboard at the triangles), matching
    the seed convention in seed_state.hpp (raxis_c/zaxis_s interpolate
    against the same folded toroidal mode n)."""
    zt = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R = np.zeros_like(zt)
    Z = np.zeros_like(zt)
    for nn in range(NTOR + 1):
        R += fams["rmncc"][nn * ns + 1] * np.cos(nn * zt)
        Z += fams["zmncs"][nn * ns + 1] * np.sin(nn * zt)
    Rfull = np.concatenate([R] * NFP)
    Zfull = np.concatenate([Z] * NFP)
    phi = np.concatenate([(zt + 2.0 * np.pi * k) / NFP for k in range(NFP)])
    X = Rfull * np.cos(phi)
    Y = Rfull * np.sin(phi)
    return np.append(X, X[0]), np.append(Y, Y[0]), np.append(Zfull, Zfull[0])


def field_lines(edge, th, zt, seeds, zeta_span=6 * 2.0 * np.pi, n_steps=2400):
    """Trace d(theta)/d(zeta) = B^theta/B^zeta on the edge half-grid surface
    (RK4, periodic in both angles), lifted onto the (r12, z12) geometry.
    `edge` is the half_grid() result on the (th, zt) solver grid."""
    ratio = edge["bsupu"] / edge["bsupv"]
    ri = RegularGridInterpolator((th, zt), ratio, method="linear",
                                 bounds_error=False, fill_value=None)
    rgi = RegularGridInterpolator((th, zt), edge["r12"], method="linear",
                                  bounds_error=False, fill_value=None)
    zgi = RegularGridInterpolator((th, zt), edge["z12"], method="linear",
                                  bounds_error=False, fill_value=None)
    lines = []
    for seed in seeds:
        zeta = np.linspace(0.0, zeta_span, n_steps)
        theta = np.empty_like(zeta)
        theta[0] = seed
        h = zeta[1] - zeta[0]
        for i in range(n_steps - 1):
            th_i, zt_i = theta[i] % (2 * np.pi), zeta[i] % (2 * np.pi)
            k1 = ri([th_i, zt_i])[0]
            k2 = ri([(theta[i] + 0.5 * h * k1) % (2 * np.pi),
                     (zeta[i] + 0.5 * h) % (2 * np.pi)])[0]
            k3 = ri([(theta[i] + 0.5 * h * k2) % (2 * np.pi),
                     (zeta[i] + 0.5 * h) % (2 * np.pi)])[0]
            k4 = ri([(theta[i] + h * k3) % (2 * np.pi),
                     (zeta[i] + h) % (2 * np.pi)])[0]
            theta[i + 1] = theta[i] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts = np.stack([theta % (2 * np.pi), zeta % (2 * np.pi)], axis=1)
        r_line = rgi(pts)
        z_line = zgi(pts)
        phi = zeta / NFP
        lines.append((r_line * np.cos(phi), r_line * np.sin(phi), z_line))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="figure_data/w7x_state.bin")
    ap.add_argument("--input", default="inputs/w7x.json")
    ap.add_argument("--out", default="figure_data/w7x_equilibrium_3d.png")
    ap.add_argument("--field-lines", action="store_true",
                    help="also trace and draw field lines on the edge surface "
                         "(off by default: the figure stays cleaner)")
    args = ap.parse_args()

    ns, mnmax, fams = load_state(args.state)
    assert (ns, mnmax) == (99, 156), f"expected the W7-X state (99, 156), got {(ns, mnmax)}"
    print(f"loaded state: ns={ns}, mnmax={mnmax}", flush=True)

    # ---- self-check 1: state LCFS vs the input boundary -----------------
    print("self-check 1/2: state LCFS vs input rbc/zbs boundary ...", flush=True)
    th_ax = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    zt_ax = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    b = eval_state(fams, ns, ns - 1, th_ax, zt_ax)
    maxsc = max(np.sqrt((ns - 1) / (ns - 1)), np.sqrt(1.0 / (ns - 1)))
    R_phys = b["re"] + maxsc * b["ro"]
    Z_phys = b["ze"] + maxsc * b["zo"]
    TH_c, ZT_c = np.meshgrid(th_ax, zt_ax, indexing="ij")
    Rb, Zb = boundary_from_input(args.input, TH_c, ZT_c)
    err = max(np.max(np.abs(R_phys - Rb)), np.max(np.abs(Z_phys - Zb)))
    print(f"  max err = {err:.3e}", flush=True)
    assert err < 1e-8, "state LCFS does not match the input boundary"

    # ---- surfaces: edge + two interior half-grid surfaces ----------------
    print("surfaces: solving chi' + half-grid geometry (jh = 97, 65, 32) ...",
          flush=True)
    th = np.linspace(0.0, 2.0 * np.pi, NTHETA, endpoint=False)
    zt = np.linspace(0.0, 2.0 * np.pi, NZETA, endpoint=False)
    surf = []
    for jh in (ns - 2, 65, 32):
        chip = solve_chip(fams, ns, jh)
        a = [eval_state(fams, ns, j, th, zt) for j in (jh, jh + 1)]
        h = half_grid(a, ns, jh, chip)
        surf.append(h)
        print(f"  jh={jh:3d}: chip={chip:+.4e}, |B| "
              f"{h['bmag'].min():.3f}-{h['bmag'].max():.3f} T", flush=True)
    edge = surf[0]
    # W7-X standard config: B ~ 2.5 T on axis, boundary |B| modulated by the
    # mirror field (roughly 1.9-2.9 T at this aspect ratio).
    assert 1.5 < edge["bmag"].min() and edge["bmag"].max() < 3.5, \
        "edge |B| outside the physical W7-X range"

    # ---- fine render grid (analytic reconstruction, same formulas) -------
    print("fine render grid: evaluating geometry + |B| on 240x120 ...", flush=True)
    nth_r, nzt_r = 240, 120
    th_r = np.linspace(0.0, 2.0 * np.pi, nth_r, endpoint=False)
    zt_r = np.linspace(0.0, 2.0 * np.pi, nzt_r, endpoint=False)
    surf_f = []
    for jh in (ns - 2, 65, 32):
        chip = solve_chip(fams, ns, jh)
        a = [eval_state(fams, ns, j, th_r, zt_r) for j in (jh, jh + 1)]
        h = half_grid(a, ns, jh, chip)
        surf_f.append(h)
    b_all = np.concatenate([s["bmag"].ravel() for s in surf_f])
    print(f"  plotted-surface |B| range {b_all.min():.3f} - {b_all.max():.3f} T",
          flush=True)

    # ---- field lines on the edge surface (opt-in) ------------------------
    lines = []
    if args.field_lines:
        print("field lines: tracing d(theta)/d(zeta) = B^theta/B^zeta (RK4) ...",
              flush=True)
        try:
            lines = field_lines(edge, th, zt,
                                seeds=[0.0, 2.0 * np.pi / 3, 4.0 * np.pi / 3])
            print(f"  {len(lines)} lines traced", flush=True)
        except Exception as exc:  # decorative; the figure survives without them
            lines = []
            print(f"  field lines skipped: {exc}", flush=True)

    # ---- converged magnetic axis (m=0 content of the innermost surface) --
    axx, axy, axz = converged_axis(fams, ns)
    print(f"magnetic axis: R {axx.min():.3f}-{axx.max():.3f}, "
          f"Z {axz.min():.3f}-{axz.max():.3f}", flush=True)

    # ---- rendering --------------------------------------------------------
    # High-contrast viridis (reversed: low |B| = bright yellow, high |B| =
    # deep purple) mapped over the actual data range; the inner surfaces sit
    # at the low-|B| end and stay bright. Shading uses the geometric surface
    # normals (surface_intensity), never the |B| heightfield.
    cmap = plt.get_cmap("viridis_r")
    norm = mcolors.Normalize(vmin=b_all.min(), vmax=b_all.max())
    light_dir = mcolors.LightSource(azdeg=330, altdeg=55).direction
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    base = os.path.splitext(args.out)[0]

    # One file per view.
    views = [("perspective", "perspective view", dict(elev=24.0, azim=-58.0)),
             ("top", "top view", dict(elev=90.0, azim=-90.0))]
    for suffix, label, vw in views:
        print(f"rendering {label} ...", flush=True)
        fig = plt.figure(figsize=(8.6, 7.8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        for k, s in enumerate(surf_f[::-1]):  # inner surfaces first
            X, Y, Z = to_cartesian(s["r12"], s["z12"])
            B = periodic_close(np.concatenate([s["bmag"]] * NFP, axis=1))
            lam = surface_intensity(X, Y, Z, light_dir)
            hsv = mcolors.rgb_to_hsv(cmap(norm(B))[:, :, :3])
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * lam, 0.0, 1.0)
            face = np.concatenate(
                [mcolors.hsv_to_rgb(hsv), np.ones(hsv.shape[:2] + (1,))], axis=2)
            is_edge = k == len(surf_f) - 1
            ax.plot_surface(X, Y, Z, facecolors=face,
                            rstride=1, cstride=2 if is_edge else 4,
                            linewidth=0, antialiased=False,
                            alpha=0.5 if is_edge else 1.0, shade=False)
        for (x, y, z) in lines:
            ax.plot(x, y, z, color="#0b0b0b", linewidth=1.1, alpha=0.85, zorder=50)
        ax.plot(axx, axy, axz, color="#0b0b0b", linewidth=1.4, zorder=60)
        lim = 7.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.8, 0.8)   # true W7-X is flat: z extent ~0.65 vs R ~6.8
        ax.set_box_aspect((1, 1, 0.15))
        ax.set_axis_off()
        ax.view_init(**vw)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02, aspect=44)
        cbar.set_label("|B| (T)", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        fig.suptitle(
            f"W7-X standard configuration — cuMES converged equilibrium, {label}\n"
            "|B| on nested half-grid surfaces "
            f"(boundary from {os.path.basename(args.input)}; magnetic axis in black)",
            fontsize=12.5, y=0.99)
        out_png = f"{base}_{suffix}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
