#!/usr/bin/env python3
"""Compare a cuMES converged state against a vmecpp .out.h5 (wout) run.

Verification tool for the vmecpp-agreement gate: both codes solved the same
input on the same radial/angular grid, and this script compares the six
spectral families mode by mode after translating the VMEC signed-n
convention into cuMES's folded product basis:

  cuMES rmncc(m,n)  = wout rmnc(m,n) + rmnc(m,-n)   (m>0; n=0: single row)
  cuMES rmnss(m,n)  = wout rmnc(m,n) - rmnc(m,-n)   (m>0)
  cuMES zmnsc(m,n)  = wout zmns(m,n) + zmns(m,-n)   (m>0; n=0: single row)
  cuMES zmncs(m,n)  = -wout zmns(m,n) + zmns(m,-n)  (m>0)
  cuMES lmnsc/lmncs = the same folding of wout lmns_full (read the FULL-grid
                      lmns_full, not the half-grid lmns)
  m = 0 (the axis displacement modes): the sin(mθ)/cos(mθ) factor kills the
  Z/λ *mnsc families, so those modes live in cuMES's *mncs families with a
  sign flip: cuMES zmncs(0,n) = -wout zmns(0,n), lmncs(0,n) = -wout lmns(0,n).

The axis row (j = 0) is skipped by default: cuMES constant-extrapolates it
(state-file representation only, known issue #1) while vmecpp keeps it zero.

Usage: compare_wout.py <cumes_state.bin> <vmecpp_output.h5> [--with-axis]

Prints the max absolute and relative difference per family; exits 1 when a
family's max absolute difference exceeds --tol (default 1e-9).
"""
import argparse
import struct
import sys

import h5py
import numpy as np

FAMS = ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")


def load_cumes(path):
    header = struct.Struct("<8siii")
    with open(path, "rb") as f:
        magic, version, ns, mnmax = header.unpack(f.read(header.size))
        if magic != b"CUMES001":
            raise SystemExit(f"error: {path} is not a cuMES state container")
        if not (1 <= version <= 4) or ns < 1 or mnmax < 1:
            raise SystemExit(f"error: {path}: bad version/dimensions")
        n = ns * mnmax
        fams = {}
        for name in FAMS:
            fams[name] = np.frombuffer(f.read(8 * n), dtype="<f8").reshape(
                mnmax, ns)
    return ns, mnmax, fams


def load_vmecpp(path):
    with h5py.File(path, "r") as f:
        w = f["wout"]
        ntor = int(w["ntor"][()])
        mpol = int(w["mpol"][()])
        rmnc = np.asarray(w["rmnc"])
        zmns = np.asarray(w["zmns"])
        lmn = np.asarray(w["lmns_full"])
    ns = rmnc.shape[1]

    # wout mode order: m=0 -> n=0..ntor, then m=1..mpol-1 -> n=-ntor..ntor.
    def wmode(m, n):
        if m == 0:
            return n
        return (ntor + 1) + (m - 1) * (2 * ntor + 1) + (n + ntor)

    fams = {name: np.zeros((mpol * (ntor + 1), ns)) for name in FAMS}
    for m in range(mpol):
        for n in range(ntor + 1):
            mode = m * (ntor + 1) + n
            if m == 0:
                fams["rmncc"][mode] = rmnc[wmode(m, n)]
                fams["zmncs"][mode] = -zmns[wmode(m, n)]
                fams["lmncs"][mode] = -lmn[wmode(m, n)]
                continue
            fams["rmncc"][mode] = rmnc[wmode(m, n)] + (
                rmnc[wmode(m, -n)] if n > 0 else 0.0)
            fams["zmnsc"][mode] = zmns[wmode(m, n)] + (
                zmns[wmode(m, -n)] if n > 0 else 0.0)
            fams["lmnsc"][mode] = lmn[wmode(m, n)] + (
                lmn[wmode(m, -n)] if n > 0 else 0.0)
            # The sin(0·ζ) factor kills the odd-ζ families at n = 0 (both
            # codes store 0 there); the signed-n difference applies for n>0.
            fams["rmnss"][mode] = rmnc[wmode(m, n)] - rmnc[
                wmode(m, -n)] if n > 0 else 0.0
            fams["zmncs"][mode] = -zmns[wmode(m, n)] + zmns[
                wmode(m, -n)] if n > 0 else 0.0
            fams["lmncs"][mode] = -lmn[wmode(m, n)] + lmn[
                wmode(m, -n)] if n > 0 else 0.0
    return ns, fams


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cumes_state", help="cuMES schema-v1 binary state")
    ap.add_argument("vmecpp_h5", help="vmecpp .out.h5 (wout)")
    ap.add_argument("--with-axis", action="store_true",
                    help="include the axis row (j=0) in the comparison")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="max allowed absolute difference per family (exit gate)")
    args = ap.parse_args()

    c_ns, c_mn, cumes = load_cumes(args.cumes_state)
    v_ns, vmecpp = load_vmecpp(args.vmecpp_h5)
    if c_ns != v_ns or c_mn != vmecpp["rmncc"].shape[0]:
        raise SystemExit(
            f"error: grid mismatch: cuMES ns={c_ns} mnmax={c_mn} vs "
            f"vmecpp ns={v_ns} mnmax={vmecpp['rmncc'].shape[0]}")
    skip = 0 if args.with_axis else 1
    worst = 0.0
    print(f"ns={c_ns} mnmax={c_mn}"
          f"{' (axis row included)' if args.with_axis else ' (axis row j=0 skipped)'}")
    for name in FAMS:
        a = cumes[name][:, skip:]
        b = vmecpp[name][:, skip:]
        d = np.abs(a - b)
        scale = np.maximum(np.abs(b), 1e-30)
        rel = d / scale
        i = np.unravel_index(np.argmax(d), d.shape)
        worst = max(worst, float(d.max()))
        print(f"{name:6s} max|d| = {d.max():.3e} at (mode {i[0]}, "
              f"j {i[1] + skip})   max rel = {rel.max():.3e}   "
              f"[vmecpp |x| range {np.abs(b).min():.3e} .. "
              f"{np.abs(b).max():.3e}]")
    if worst > args.tol:
        print(f"FAIL: worst family max|d| {worst:.3e} exceeds --tol {args.tol:g}")
        sys.exit(1)
    print(f"OK: worst family max|d| {worst:.3e} within --tol {args.tol:g}")


if __name__ == "__main__":
    main()
