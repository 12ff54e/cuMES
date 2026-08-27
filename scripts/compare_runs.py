#!/usr/bin/env python3
"""Old/new comparison of two cuMES runs (no vmecpp reference needed).

Given two run logs and two converged-state binaries, verifies that a new run
reproduces the baseline run:

  1. Converged state (the schema-v1 binary container, on-disk versions 1-8,
     or the legacy raw baseline dump): six spectral families
     (rmncc/zmnsc/lmnsc/rmnss/zmncs/lmncs), axis row (j=0) skipped, PASS at
     <= --tol relative (default 1e-8). Mirrors scripts/compare_states.py.
  2. Per-iteration residual rows parsed from the solver log
     ("%5d | %11.3e %11.3e %11.3e | %8.2e"): fsqr/fsqz/fsql/delt compared at
     <= --log-tol relative (default 5e-3). The printed precision is 3-4
     significant digits (%.3e / %.2e), so the log tolerance must be looser
     than the state tolerance; a real trajectory divergence compounds far
     beyond it within a few iterations.
  3. Restart markers: BAD JACOBIAN / BAD PROGRESS / CONVERGENCE PROBLEM
     (RESETTING DELT) must occur at the same iterations in the same order.
  4. Convergence: both runs must converge, at iterations differing by at most
     --max-iter-delta (default 10), with matching final FSQR/FSQZ/FSQL
     (within --log-tol).

Usage:
  compare_runs.py <log_a> <state_a> <log_b> <state_b> [--tol 1e-8]
                  [--log-tol 5e-3] [--max-iter-delta 10]
  compare_runs.py --no-state <log_a> <log_b> [options]

Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import re
import struct
import sys

FAMS = ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")

# Per-iteration row: "    0 | 1.616e-01   1.616e-01   4.775e-02 | 9.00e-01 ..."
ITER_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)"
)
# Restart / progress markers
BADJ_RE = re.compile(r"BAD JACOBIAN \(iter2=(\d+)\) delt=([0-9.eE+-]+)")
BADP_RE = re.compile(r"BAD PROGRESS \(iter2=(\d+)\) delt=([0-9.eE+-]+)")
RESET_RE = re.compile(r"CONVERGENCE PROBLEM: RESETTING DELT to ([0-9.eE+-]+) \(ijacob=(\d+)\)")
CONV_RE = re.compile(r"CONVERGED at iter (\d+)")


def load_state(path):
    """Read the schema-v1 binary state payload: six mode-major families.

    Accepts on-disk container versions 1-8 (magic "CUMES001" + version +
    ns + mnmax header; the provenance trailer is ignored), plus the legacy
    pre-container raw dump the frozen baselines were captured in (int32 ns +
    int32 mnmax + the same six families, no magic, exactly sized).
    """
    fail = f"error: {path} is not a cumes state container"
    with open(path, "rb") as f:
        data = f.read()

    def families(offset, ns, mnmax):
        fams = {}
        pos = offset
        for name in FAMS:
            fams[name] = struct.unpack_from(f"<{ns * mnmax}d", data, pos)
            pos += 8 * ns * mnmax
        return fams

    header = struct.Struct("<8siii")  # magic(8) version ns mnmax = 20 bytes
    if len(data) >= header.size:
        magic, version, ns, mnmax = header.unpack_from(data)
        if magic == b"CUMES001" and 1 <= version <= 8 and ns >= 1 and mnmax >= 1:
            try:
                return ns, mnmax, families(header.size, ns, mnmax)
            except struct.error:
                raise SystemExit(fail)

    # Legacy raw baseline dump (pre-container): ns, mnmax, six families.
    if len(data) >= 8:
        ns, mnmax = struct.unpack_from("<ii", data)
        if ns >= 1 and mnmax >= 1 and len(data) == 8 + 6 * ns * mnmax * 8:
            return ns, mnmax, families(8, ns, mnmax)

    raise SystemExit(fail)


def parse_log(path):
    """Return (rows, restarts, converged_iter, status, iterations, summary).

    rows: dict iter -> (fsqr, fsqz, fsql, delt)
    restarts: list of (kind, iter2, delt); kind in {BADJ, BADP, RESET}
    summary: dict of "FSQR"/"FSQZ"/"FSQL" final values, or None
    """
    rows = {}
    restarts = []
    converged_iter = None
    status = None
    iterations = None
    summary = {}
    with open(path, "r") as f:
        for line in f:
            m = ITER_RE.match(line)
            if m:
                it = int(m.group(1))
                fr, fz, fl, dt = (float(m.group(i)) for i in (2, 3, 4, 5))
                rows[it] = (fr, fz, fl, dt)
                continue
            m = BADJ_RE.search(line)
            if m:
                restarts.append(("BADJ", int(m.group(1)), float(m.group(2))))
                continue
            m = BADP_RE.search(line)
            if m:
                restarts.append(("BADP", int(m.group(1)), float(m.group(2))))
                continue
            m = RESET_RE.search(line)
            if m:
                restarts.append(("RESET", None, float(m.group(1))))
                continue
            m = CONV_RE.search(line)
            if m:
                converged_iter = int(m.group(1))
                continue
            if line.startswith("  Status:"):
                status = line.split(":", 1)[1].strip()
            elif line.startswith("  Iterations:"):
                iterations = int(line.split(":", 1)[1].strip())
            elif line.startswith("  FSQR:"):
                summary["FSQR"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("  FSQZ:"):
                summary["FSQZ"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("  FSQL:"):
                summary["FSQL"] = float(line.split(":", 1)[1].strip())
    return rows, restarts, converged_iter, status, iterations, summary


def rel(a, b):
    """Difference scaled like compare_states.py: |a-b|/max(1, |b|).

    Effectively absolute for |b| < 1 (so tiny coefficients don't blow up the
    metric), relative for |b| > 1.
    """
    return abs(a - b) / max(1.0, abs(b))


def compare(rows_a, rows_b, tol):
    """Per-row residual comparison over the common iteration range."""
    iters = sorted(set(rows_a) & set(rows_b))
    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    worst = {k: 0.0 for k in ("fsqr", "fsqz", "fsql", "delt")}
    worst_at = {k: None for k in worst}
    for it in iters:
        a, b = rows_a[it], rows_b[it]
        for idx, k in enumerate(worst):
            d = rel(a[idx], b[idx])
            if d > worst[k]:
                worst[k], worst_at[k] = d, it
    ok = all(w < tol for w in worst.values()) and not only_a and not only_b
    return ok, worst, worst_at, len(iters), only_a, only_b


def compare_state(sa, sb, tol):
    (a_ns, a_mn, a), (b_ns, b_mn, b) = sa, sb
    if (a_ns, a_mn) != (b_ns, b_mn):
        return False, f"size mismatch: {(a_ns, a_mn)} vs {(b_ns, b_mn)}", {}
    ns, mnmax = a_ns, a_mn
    worst = {}
    for name in FAMS:
        amax = 0.0
        for m in range(mnmax):
            for j in range(1, ns):  # skip the extrapolated axis row
                i = m * ns + j
                amax = max(amax, rel(a[name][i], b[name][i]))
        worst[name] = amax
    ok = all(v < tol for v in worst.values())
    return ok, f"ns={ns} mnmax={mnmax} (interior j=1..{ns-1})", worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_a")
    ap.add_argument("state_a", nargs="?")
    ap.add_argument("log_b")
    ap.add_argument("state_b", nargs="?")
    ap.add_argument("--no-state", action="store_true", help="log-only comparison")
    ap.add_argument("--tol", type=float, default=1e-8, help="state tolerance (default 1e-8)")
    ap.add_argument("--log-tol", type=float, default=5e-3,
                    help="printed-row tolerance (default 5e-3; print precision is ~1e-4)")
    ap.add_argument("--max-iter-delta", type=int, default=10,
                    help="max |converged iters| difference (default 10)")
    args = ap.parse_args()

    if not args.no_state and (not args.state_a or not args.state_b):
        print("error: need state files, or pass --no-state for log-only comparison")
        return 2

    fails = []

    # ---- Per-iteration residuals + markers ----
    rows_a, rst_a, conv_a, stat_a, niter_a, summ_a = parse_log(args.log_a)
    rows_b, rst_b, conv_b, stat_b, niter_b, summ_b = parse_log(args.log_b)
    ok, worst, worst_at, n_rows, only_a, only_b = compare(rows_a, rows_b, args.log_tol)
    print(f"residual rows: {n_rows} iterations compared (worst rel diff per column)")
    for k in ("fsqr", "fsqz", "fsql", "delt"):
        print(f"  {k:4s}: {worst[k]:.3e} @ iter {worst_at[k]}" + ("  <-- EXCEEDS TOL" if worst[k] >= args.log_tol else ""))
    if only_a:
        print(f"  rows only in A: {only_a[:10]}{' ...' if len(only_a) > 10 else ''}")
    if only_b:
        print(f"  rows only in B: {only_b[:10]}{' ...' if len(only_b) > 10 else ''}")
    if not ok:
        fails.append("per-iteration residuals")

    # ---- Restart sequence ----
    seq_a = [(k, i) for k, i, _ in rst_a]
    seq_b = [(k, i) for k, i, _ in rst_b]
    print(f"restarts: A={len(rst_a)} events, B={len(rst_b)} events")
    if seq_a != seq_b:
        fails.append("restart sequence")
        print(f"  A: {seq_a}")
        print(f"  B: {seq_b}")
    else:
        print(f"  identical: {seq_a}")
        for (ka, ia, da), (kb, ib, db) in zip(rst_a, rst_b):
            if ka == "RESET":
                if rel(da, db) >= args.log_tol:
                    print(f"  RESET delt drift: {da:.3e} vs {db:.3e}")
                    fails.append("reset delt")
            elif rel(da, db) >= args.log_tol:
                print(f"  {ka}@iter{ia} delt drift: {da:.3e} vs {db:.3e}")
                fails.append("restart delt")

    # ---- Convergence ----
    print(f"converged: A={'iter ' + str(conv_a) if conv_a else 'NO'}  "
          f"B={'iter ' + str(conv_b) if conv_b else 'NO'}  (status {stat_a} / {stat_b})")
    if not conv_a or not conv_b:
        fails.append("one or both runs did not converge")
    else:
        d = abs(conv_a - conv_b)
        print(f"  converged-iter delta: {d} (window {args.max_iter_delta})")
        if d > args.max_iter_delta:
            fails.append("converged iteration count")
    for k in ("FSQR", "FSQZ", "FSQL"):
        if k in summ_a and k in summ_b:
            d = rel(summ_a[k], summ_b[k])
            print(f"  final {k}: A={summ_a[k]:.3e} B={summ_b[k]:.3e} rel={d:.2e}")
            if d >= args.log_tol:
                fails.append(f"final {k}")

    # ---- State ----
    if not args.no_state:
        sa, sb = load_state(args.state_a), load_state(args.state_b)
        ok_s, msg, worst_s = compare_state(sa, sb, args.tol)
        print(f"state: {msg}")
        for name in FAMS:
            print(f"  {name:6s}: max rel diff (interior) = {worst_s[name]:.3e}"
                  + ("  <-- EXCEEDS TOL" if worst_s[name] >= args.tol else ""))
        if not ok_s:
            fails.append("state")

    if fails:
        print(f"FAIL: {', '.join(fails)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
