#!/usr/bin/env python3
"""Render 3D figures of the converged W7-X equilibrium (cuMES state).

Reads any solver state container (docs/output-formats.md): legacy binary
v0, versioned binary (v1), checkpoint (CUMECKP1), NetCDF, or HDF5.
Reconstructs real-space geometry with the solver's exact conventions
(parity-split e/o arrays, odd-m scalxc regularization, staggered half-grid
metric — src/geometry_impl.cuh), solves the ncurr=1 current constraint for
chi' per plotted surface (ncurr1FinalizeKernel), and renders the full
5-period torus colored by |B| with light-source shading.

One PNG file per view: <out>_perspective.png, <out>_top.png,
<out>_combined.png (both views side by side), and <out>_slices.png (the
top view with a column of three RZ poloidal cross-sections: bean,
transition, triangle). The magnetic axis is the converged axis extracted
from the state (m=0 content of the innermost surface, seed_state.hpp
convention: nfp-fold modulation).

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
from matplotlib.collections import LineCollection
from PIL import Image as PILImage
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
    """Load the converged state from any solver output container
    (docs/output-formats.md): versioned binary (v1), checkpoint (CUMECKP1),
    NetCDF, or HDF5. Returns ns, mnmax and the six mode-major families
    (index = mode * ns + surface)."""
    fam_names = ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")
    with open(path, "rb") as f:
        head = f.read(16)
    if head.startswith(b"CUMES001") or head.startswith(b"CUMECKP1"):
        # Versioned binary: magic(8), version(4) [, precision(4) for the
        # checkpoint], ns(4), mnmax(4), then the six families.
        with open(path, "rb") as f:
            f.seek(8)
            version = struct.unpack("<i", f.read(4))[0]
            if head.startswith(b"CUMECKP1"):
                struct.unpack("<i", f.read(4))[0]  # precision (always double)
            ns, mnmax = struct.unpack("<ii", f.read(8))
            n = ns * mnmax
            fams = {name: np.frombuffer(f.read(8 * n), dtype="<f8")
                    for name in fam_names}
        return ns, mnmax, fams
    if head.startswith(b"CDF"):
        # NetCDF (v1): the six families are 2-D [surface, mode] datasets.
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as nc:
            ns = nc.dimensions["ns"]
            mnmax = nc.dimensions["mnmax"]
            fams = {name: np.asarray(nc.variables[name][:], dtype="<f8").T.ravel()
                    for name in fam_names}
        return ns, mnmax, fams
    if head.startswith(b"\x89HDF"):
        # HDF5 (v1): same [surface, mode] dataset layout.
        import h5py
        with h5py.File(path, "r") as f5:
            fams = {}
            ns = mnmax = None
            for name in fam_names:
                dset = np.asarray(f5[name][:], dtype="<f8")  # [surface, mode]
                ns, mnmax = dset.shape
                fams[name] = dset.T.ravel()
        return ns, mnmax, fams
    raise SystemExit(f"error: {path} is not a cumes state container "
                     f"(expected CUMES001/CUMECKP1/NetCDF/HDF5)")


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


def trim_white(path, pad_px=12, title_gap_px=8, center="title"):
    """Post-process the saved PNG: crop to the non-white content, collapse
    the white strip between the title and the figure (3-D projections
    reserve margin for rotation, so bbox_inches='tight' cannot remove it),
    and pad the narrower side so the requested anchor ends up horizontally
    centered. center='title' centers on the topmost ink band (the title);
    center='body' centers on the median x of the body ink below it, which
    is robust against an asymmetric colorbar and centers the torus itself
    (with an axes title, both coincide)."""
    img = PILImage.open(path).convert("RGB")
    a = np.asarray(img)
    nonwhite = ~(a > 250).all(axis=2)
    ys, xs = np.where(nonwhite)
    if len(ys) == 0:
        return
    rows = nonwhite.any(axis=1)
    row_ys = np.where(rows)[0]
    gap0 = np.where(~rows[row_ys[0]:])[0]
    t1 = row_ys[0] + (gap0[0] if len(gap0) else a.shape[0] - row_ys[0])
    band = nonwhite[row_ys[0]:t1]
    band_xs = np.where(band)[1]
    title_cx = 0.5 * (band_xs.min() + band_xs.max()) if len(band_xs) else \
        0.5 * (xs.min() + xs.max())
    below = np.where(rows[t1:])[0]
    body_top = t1 + (below[0] if len(below) else 0)
    # Anchor = the horizontal midpoint of the body ink extent, colorbar
    # included, so the whole figure block (torus + colorbar) is centered
    # and both side margins balance.
    cut = int(0.92 * a.shape[1])  # separates the title text from the colorbar
    body_xs = xs[ys >= body_top]
    anchor = 0.5 * (body_xs.min() + body_xs.max()) \
        if center == "body" and len(body_xs) else title_cx
    # The 3-D box is not centered in the axes window, so the title (placed
    # at the window center by matplotlib) must be shifted onto the torus
    # center in post. The shift moves the title text only; the colorbar
    # label inside the same band stays put.
    title_shift = 0
    tx0 = tx1 = 0
    if center == "body":
        band_mask = band_xs < cut
        if band_mask.any():
            tx0, tx1 = band_xs[band_mask].min(), band_xs[band_mask].max()
            title_shift = int(round(anchor - 0.5 * (tx0 + tx1)))
    # Title piece and body piece; merge when they already touch. The body
    # piece ends at its own ink bottom (the pre-crop image may carry the
    # axes window's white margin below the projection).
    pieces = [(max(row_ys[0] - pad_px, 0), t1 + pad_px),
              (max(body_top - pad_px, 0), min(ys.max() + pad_px, a.shape[0]))]
    merged = []
    for lo, hi in pieces:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    x0 = max(xs.min() - pad_px, 0)
    x1 = min(xs.max() + pad_px, a.shape[1] - 1)
    total_h = sum(hi - lo for lo, hi in merged) + title_gap_px * (len(merged) - 1)
    canvas = PILImage.new("RGB", (x1 + 1 - x0, total_h), (255, 255, 255))
    y_cursor = 0
    for i, (lo, hi) in enumerate(merged):
        piece = img.crop((x0, lo, x1 + 1, hi))
        if i == 0 and title_shift:
            white = PILImage.new("RGB", (tx1 - tx0 + 1, piece.height),
                                 (255, 255, 255))
            piece.paste(white, (tx0 - x0, 0))
            text = img.crop((tx0, lo, tx1 + 1, hi))
            piece.paste(text, (tx0 - x0 + title_shift, 0))
        canvas.paste(piece, (0, y_cursor))
        y_cursor += piece.height + (title_gap_px if i < len(merged) - 1 else 0)
    w, h = canvas.size
    # Anchor center inside the canvas: c = anchor - x0. Padding the canvas
    # with L left / R right (L + c = (w + L + R - 1)/2) gives
    # L = w - 1 - 2c for L >= 0, else R = 2c - (w - 1).
    c = anchor - x0
    shift = int(round((w - 1) - 2.0 * c))
    out = PILImage.new("RGB", (w + abs(shift), h), (255, 255, 255))
    out.paste(canvas, (max(shift, 0), 0))
    out.save(path)


def compose_combined(path, pad_px=12, title_gap_px=8, panel_gap_px=22,
                     cbar_gap_px=14):
    """Post-process the combined PNG: crop each panel's ink and the
    colorbar separately and recompose them side by side with fixed gaps,
    then center the title over the composite. The 3-D boxes do not fill
    their axes windows (and sit right of the window center), so no layout
    setting can remove the inter-panel white space."""
    img = PILImage.open(path).convert("RGB")
    a = np.asarray(img)
    nonwhite = ~(a > 250).all(axis=2)
    ys, xs = np.where(nonwhite)
    if len(ys) == 0:
        return
    rows = nonwhite.any(axis=1)
    row_ys = np.where(rows)[0]
    gap0 = np.where(~rows[row_ys[0]:])[0]
    t1 = row_ys[0] + (gap0[0] if len(gap0) else a.shape[0] - row_ys[0])
    band = nonwhite[row_ys[0]:t1]
    band_xs = np.where(band)[1]
    cut = int(0.92 * a.shape[1])
    tx = band_xs[band_xs < cut]
    below = np.where(rows[t1:])[0]
    body_top = t1 + (below[0] if len(below) else 0)
    body_bottom = ys.max()
    # column runs over the body only
    col = nonwhite[body_top:body_bottom + 1].any(axis=0)
    runs = []
    start = None
    for i, v in enumerate(col):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(col) - 1])
    groups = []
    for r in runs:
        if groups and r[0] - groups[-1][1] < 40:
            groups[-1][1] = r[1]
        else:
            groups.append(r)
    if len(groups) < 2:
        trim_white(path, center="body")
        return
    cbar = groups[-1]
    pieces = []
    for g in groups[:-1]:
        sub = nonwhite[body_top:body_bottom + 1, g[0]:g[1] + 1]
        if not sub.any():
            continue
        gy, _ = np.where(sub)
        lo = max(body_top + gy.min() - pad_px, body_top)
        hi = min(body_top + gy.max() + pad_px, body_bottom)
        pieces.append(img.crop((g[0], lo, g[1] + 1, hi + 1)))
    csub = nonwhite[body_top:body_bottom + 1, cbar[0]:cbar[1] + 1]
    cy, _ = np.where(csub)
    cla = max(body_top + cy.min() - pad_px, body_top)
    clb = min(body_top + cy.max() + pad_px, body_bottom)
    cbar_piece = img.crop((cbar[0], cla, cbar[1] + 1, clb + 1))
    body_w = sum(p.width for p in pieces) + panel_gap_px * (len(pieces) - 1) \
        + cbar_gap_px + cbar_piece.width
    body_h = max([p.height for p in pieces] + [cbar_piece.height])
    body = PILImage.new("RGB", (body_w, body_h), (255, 255, 255))
    x = 0
    for p in pieces:
        body.paste(p, (x, (body_h - p.height) // 2))
        x += p.width + panel_gap_px
    x -= panel_gap_px
    body.paste(cbar_piece, (x + cbar_gap_px, (body_h - cbar_piece.height) // 2))
    # title strip: white canvas + the title text centered over the composite
    title_h = t1 + pad_px - max(row_ys[0] - pad_px, 0)
    out = PILImage.new("RGB", (body_w, title_h + title_gap_px + body_h),
                       (255, 255, 255))
    if len(tx):
        text = img.crop((tx.min(), max(row_ys[0] - pad_px, 0),
                         tx.max() + 1, t1 + pad_px))
        out.paste(text, ((body_w - text.width) // 2, 0))
    out.paste(body, (0, title_h + title_gap_px))
    # outer margin
    final = PILImage.new("RGB", (out.width + 2 * pad_px, out.height + 2 * pad_px),
                         (255, 255, 255))
    final.paste(out, (pad_px, pad_px))
    final.save(path)


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

    # One file per view, plus a combined two-panel figure.
    def draw_scene(ax, vw):
        """Draw surfaces + axis curve into one 3-D axis."""
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
        # Tight limits hug the plasma (R 4.66-6.22, |Z| <= 0.90) so the
        # torus fills the frame instead of floating in white space.
        lim = 6.55
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.98, 0.98)
        ax.set_box_aspect((1, 1, 0.15))
        ax.set_axis_off()
        ax.view_init(**vw)

    views = [("perspective", "perspective view", dict(elev=24.0, azim=-58.0)),
             ("top", "top view", dict(elev=90.0, azim=-90.0))]
    for suffix, label, vw in views:
        print(f"rendering {label} ...", flush=True)
        # constrained_layout: 3-D projections reserve wide margins for
        # rotation; constrained_layout + bbox_inches='tight' mitigates them
        # (the rest is handled by trim_white).
        fig = plt.figure(figsize=(7.4, 5.6), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        draw_scene(ax, vw)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, aspect=20,
                            pad=0.01)
        cbar.set_label("|B| (T)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        # The title is re-centered onto the torus by trim_white (the 3-D
        # box is not centered in the axes window, so matplotlib-side
        # placement cannot do it).
        fig.suptitle(
            f"W7-X standard configuration — cuMES equilibrium ({label})",
            fontsize=11, y=1.01)
        out_png = f"{base}_{suffix}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        trim_white(out_png, center="body")
        print(f"saved {out_png}", flush=True)

    # ---- combined two-panel figure (one shared colorbar) -----------------
    print("rendering combined view ...", flush=True)
    # rect leaves a clean strip for the suptitle so trim_white can separate
    # it from the panel titles without ambiguity.
    fig = plt.figure(figsize=(12.6, 4.7), dpi=100,
                     layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.93))
    axes = []
    for i, (suffix, label, vw) in enumerate(views):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        axes.append(ax)
        draw_scene(ax, vw)
        ax.set_title(label, fontsize=10, pad=-4)
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.6, aspect=20, pad=0.01)
    cbar.set_label("|B| (T)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle("W7-X standard configuration — cuMES converged equilibrium",
                 fontsize=11, y=1.01)
    out_png = f"{base}_combined.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    compose_combined(out_png)
    print(f"saved {out_png}", flush=True)

    # ---- top view + poloidal cross-section slices -------------------------
    # One column of three RZ-plane contours at fixed toroidal angles (the
    # bean, the intermediate transition, and the triangle cross-sections),
    # to the left of the top-view 3D plot.
    print("rendering top view + RZ slices ...", flush=True)
    zeta_cuts = [(0.0, "bean  (φ = 0°)"),
                 (0.5 * np.pi, "transition  (φ = 18°)"),
                 (np.pi, "triangle  (φ = 36°)")]
    # rect reserves a clean strip for the suptitle so trim_white can
    # separate it from the slice column without ambiguity.
    fig = plt.figure(figsize=(7.6, 7.2), dpi=100, layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.93))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.7], height_ratios=[1, 1, 1])
    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax3d = fig.add_subplot(gs[:, 1], projection="3d")
    draw_scene(ax3d, dict(elev=90.0, azim=-90.0))
    r_all = np.concatenate([s["r12"].ravel() for s in surf_f])
    z_all = np.concatenate([s["z12"].ravel() for s in surf_f])
    Rmin, Rmax = r_all.min(), r_all.max()
    Zmax = max(abs(z_all.min()), abs(z_all.max()))
    for i, (zc, name) in enumerate(zeta_cuts):
        idx = int(round(zc / (2.0 * np.pi) * nzt_r)) % nzt_r
        ax = axs[i]
        for k, s in enumerate(surf_f):
            R = np.append(s["r12"][:, idx], s["r12"][0, idx])
            Z = np.append(s["z12"][:, idx], s["z12"][0, idx])
            B = np.append(s["bmag"][:, idx], s["bmag"][0, idx])
            pts = np.stack([R, Z], axis=1)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm,
                                linewidths=2.6 if k == 0 else 1.9)
            lc.set_array(B[:-1])
            ax.add_collection(lc)
        Rax = sum(fams["rmncc"][nn * ns + 1] * np.cos(nn * zc)
                  for nn in range(NTOR + 1))
        Zax = sum(fams["zmncs"][nn * ns + 1] * np.sin(nn * zc)
                  for nn in range(NTOR + 1))
        ax.plot([Rax], [Zax], "o", color="#0b0b0b", markersize=3.0)
        ax.set_xlim(Rmin - 0.05, Rmax + 0.05)
        ax.set_ylim(-1.05 * Zmax, 1.05 * Zmax)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=9, pad=2)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("Z (m)", fontsize=9)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("R (m)", fontsize=9)
    cbar = fig.colorbar(mappable, ax=ax3d, shrink=0.6, aspect=20, pad=0.01)
    cbar.set_label("|B| (T)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle("W7-X standard configuration — cuMES converged equilibrium "
                 "(top view + poloidal cross-sections)", fontsize=11, y=1.01)
    out_png = f"{base}_slices.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
