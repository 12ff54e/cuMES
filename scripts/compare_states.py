#!/usr/bin/env python3
"""Compare two cuMES converged-state binaries (schema-v1 binary format).

Format: magic "CUMES001" (8) + int32 version + 2 ints (ns, mnmax), then 6
families of ns*mnmax doubles each, in the order rmncc, zmnsc, lmnsc, rmnss,
zmncs, lmncs (mode-major [m*ns + j]); the provenance trailer after the state
is ignored.

Usage: compare_states.py <state_a.bin> <state_b.bin>
Prints the max abs/rel difference per family and the per-mode max, skipping
the axis row (j=0), which is extrapolated and may differ between runs.
"""
import struct
import sys

FAMS = ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")
MAGIC = b"CUMES001"
HEADER = struct.Struct("<8siii")  # magic(8) version ns mnmax = 20 bytes


def load(path):
    with open(path, "rb") as f:
        head = f.read(HEADER.size)
        if len(head) != HEADER.size:
            raise SystemExit(f"error: {path} is not a v1 cumes state container")
        magic, version, ns, mnmax = HEADER.unpack(head)
        if magic != MAGIC or not (1 <= version <= 6) or ns < 1 or mnmax < 1:
            raise SystemExit(f"error: {path} is not a v1 cumes state container")
        n = ns * mnmax
        fams = {}
        for name in FAMS:
            fams[name] = struct.unpack(f"<{n}d", f.read(8 * n))
    return ns, mnmax, fams


def main():
    a_ns, a_mn, a = load(sys.argv[1])
    b_ns, b_mn, b = load(sys.argv[2])
    assert (a_ns, a_mn) == (b_ns, b_mn), f"size mismatch: {(a_ns, a_mn)} vs {(b_ns, b_mn)}"
    ns, mnmax = a_ns, a_mn
    print(f"ns={ns} mnmax={mnmax} (comparing interior j=1..{ns-1})")
    worst = 0.0
    for name in FAMS:
        amax = 0.0
        for m in range(mnmax):
            for j in range(1, ns):  # skip the extrapolated axis row
                i = m * ns + j
                d = abs(a[name][i] - b[name][i])
                scale = max(1.0, abs(b[name][i]))
                amax = max(amax, d / scale)
        worst = max(worst, amax)
        print(f"  {name:6s}: max rel diff (interior) = {amax:.3e}")
    print(f"worst over all families: {worst:.3e}")
    # Two runs stopped at the same residual floor (FSQR ~ 1e-13) can end at
    # state differences of ~1e-9 relative (the direct backend itself matches
    # vmecpp's wout at <=1.5e-9) — anything worse than 1e-8 indicates a
    # real transform discrepancy.
    ok = worst < 1e-8
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
