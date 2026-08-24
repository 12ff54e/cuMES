#!/usr/bin/env python3
"""Byte-exact comparison of two cuMES run trees (Class A bitwise gate).

The overhaul blueprint (docs/cuda-overhaul-blueprint.md, section 10) requires a
Class A "bitwise equivalence" gate for build/library splits and scheduling
changes: identical inputs, toolchain, flags, GPU, and stage caps must produce
identical bytes. This script is that gate.

The baseline is NOT committed as binary data — it is regenerated on demand by
checking out the baseline revision and running scripts/capture_baseline.sh (see
that script). This tool compares two run trees produced that way:

  baseline_dir = capture output from the baseline revision build
  run_dir      = capture output from the candidate (post-change) build

Each tree holds, per <precision>/<config>/:

  cumes_state.bin              # final state (schema-v1 container: magic +
                               # version + ns + mnmax + 6 x ns*mnmax doubles
                               # + provenance trailer)
  per_iter_residuals_cumes.bin # 15-col double trajectory record (all stages)
  init_*.bin                   # deterministic initial-state snapshots
  dump_manifest.sha256         # SHA-256 of the FULL dump set (if generated)
  PROVENANCE.txt               # revision/build/knobs that produced the tree

Comparison modes:

  --essentials (default)
      Byte-compares the cumes_state.bin STATE PAYLOAD (the container's
      provenance trailer embeds the git revision, so full-file bytes differ
      across revisions by design), per_iter_residuals_cumes.bin and the
      init_* set. When the baseline carries a dump_manifest.sha256, the
      run's full dump/cuMES/* set is verified against those checksums. Fast;
      the per-iteration record already spans every pass of every multigrid
      stage, so a byte-identical trajectory + final state is a strong
      equivalence claim.

  --full
      Byte-compares EVERY file in dump/cuMES/* (sorted) between the two
      trees, including the large postinverse..preconditioned / fspec /
      state / vel component
      snapshots. Use when both trees kept the full dump set.

Both modes require the runs to use the SAME environment knobs as each other
(CUMES_DUMP=1 and the same CUMES_DUMP_ITER etc.), because the dump machinery
couples to the trajectory (solver.cu adds a precon refresh at iter2==kDumpIter).
Comparing runs recorded with different knobs is meaningless.

Usage:
  compare_bitwise.py <baseline_dir> <run_dir> [--full] [--verbose]

  <baseline_dir>  e.g. .verify-scratch/baseline/double/solovev
  <run_dir>       the matching capture from the candidate build, or a
                  directory holding cumes_state.bin + the run's dump tree.

Exit status 0 = byte-identical; 1 = differences; 2 = usage/missing-file error.
"""
import argparse
import hashlib
import os
import struct
import sys

# Family order in the binary state file (matches the v1 container payload).
STATE_FAMS = ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")

# Essential artifacts that must exist in the baseline and run directories.
ESSENTIALS = ("cumes_state.bin", "per_iter_residuals_cumes.bin")

# Schema-v1 binary header: magic(8) version(4) ns(4) mnmax(4) = 20 bytes.
STATE_HEADER = struct.Struct("<8siii")
STATE_MAGIC = b"CUMES001"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dump_paths(tree_root):
    """Sorted (relative-path) dump files under <root>/dump/cuMES, if any."""
    dump = os.path.join(tree_root, "dump", "cuMES")
    if not os.path.isdir(dump):
        return []
    paths = []
    for name in sorted(os.listdir(dump)):
        full = os.path.join(dump, name)
        if os.path.isfile(full):
            paths.append(os.path.join("dump", "cuMES", name))
    return paths


def _read_state_payload(path):
    """Return (ns, mnmax, payload_bytes) of the v1 container, or None.

    The payload is the 20-byte header (magic + version + ns + mnmax) plus the
    six mode-major family byte strings. The provenance trailer after the
    state is NOT part of the payload (it embeds the git revision, so it
    differs across revisions by design).
    """
    with open(path, "rb") as f:
        head = f.read(STATE_HEADER.size)
        if len(head) != STATE_HEADER.size:
            return None
        magic, version, ns, mnmax = STATE_HEADER.unpack(head)
        # The state payload layout is identical across versions 1..4 (the
        # provenance trailer changes); accept them all and compare payloads.
        if magic != STATE_MAGIC or not (1 <= version <= 4) or ns < 1 or mnmax < 1:
            return None
        n = ns * mnmax
        payload = f.read(6 * n * 8)
        if len(payload) != 6 * n * 8:
            return None
    return ns, mnmax, payload


def _parse_manifest(path):
    """Parse a dump_manifest.sha256 file: '<sha256>  <relative-path>' per line."""
    entries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                digest, rel = line.split("  ", 1)
            except ValueError:
                digest, rel = line.split(" ", 1)
            entries[rel] = digest
    return entries


def _report(label, status, detail=""):
    flag = "OK " if status else "DIFF"
    print(f"  [{flag}] {label}" + (f"  ({detail})" if detail else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline_dir", help="baseline run-tree directory")
    ap.add_argument("run_dir", help="candidate run-tree directory")
    ap.add_argument("--full", action="store_true",
                    help="byte-compare every dump/cuMES file, not just essentials")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    b = args.baseline_dir
    r = args.run_dir
    if not os.path.isdir(b):
        print(f"error: baseline dir not found: {b}", file=sys.stderr)
        return 2
    if not os.path.isdir(r):
        print(f"error: run dir not found: {r}", file=sys.stderr)
        return 2

    failures = 0

    # ---- 1. final state (payload byte + semantic) --------------------------
    bs, rs = os.path.join(b, "cumes_state.bin"), os.path.join(r, "cumes_state.bin")
    if not os.path.isfile(bs) or not os.path.isfile(rs):
        print("error: cumes_state.bin missing from baseline or run", file=sys.stderr)
        return 2
    pb = _read_state_payload(bs)
    pr = _read_state_payload(rs)
    if pb is None or pr is None:
        _report("cumes_state.bin", False, "not a valid v1 container")
        failures += 1
    elif pb[2] != pr[2]:
        _report("cumes_state.bin", False, "state payload byte mismatch")
        failures += 1
        if args.verbose and (pb[0], pb[1]) == (pr[0], pr[1]):
            ns, mnmax = pb[0], pb[1]
            for c, name in enumerate(STATE_FAMS):
                off = c * ns * mnmax * 8
                fa = struct.unpack_from(f"<{ns*mnmax}d", pb[2], off)
                fb = struct.unpack_from(f"<{ns*mnmax}d", pr[2], off)
                amax = max(abs(x - y) for x, y in zip(fa, fb))
                print(f"      {name:6s}: max abs diff = {amax:.3e}")
    else:
        _report("cumes_state.bin", True)

    # ---- 2. per-iteration trajectory record --------------------------------
    for f in ("per_iter_residuals_cumes.bin",):
        bf, rf = os.path.join(b, f), os.path.join(r, f)
        if not os.path.isfile(bf) or not os.path.isfile(rf):
            print(f"error: {f} missing from baseline or run", file=sys.stderr)
            return 2
        if _sha256(bf) != _sha256(rf):
            _report(f, False, "byte mismatch")
            failures += 1
        else:
            _report(f, True)

    # ---- 3. init_* initial-state snapshots ---------------------------------
    b_steps = sorted(n for n in os.listdir(b) if n.startswith("init_"))
    r_steps = sorted(n for n in os.listdir(r) if n.startswith("init_"))
    if b_steps != r_steps:
        _report("init set", False,
                f"name sets differ: baseline {b_steps} vs run {r_steps}")
        failures += 1
    else:
        for name in b_steps:
            if _sha256(os.path.join(b, name)) != _sha256(os.path.join(r, name)):
                _report(name, False, "byte mismatch")
                failures += 1
            elif args.verbose:
                _report(name, True)
        if not failures:
            _report(f"init set ({len(b_steps)} files)", True)

    # ---- 4. full dump-set manifest verification ----------------------------
    b_manifest = os.path.join(b, "dump_manifest.sha256")
    if os.path.isfile(b_manifest):
        expected = _parse_manifest(b_manifest)
        # Manifest entries are relative to dump/cuMES/; run paths are
        # relative to the tree root.
        expected_abs = {os.path.join("dump", "cuMES", rel): digest
                        for rel, digest in expected.items()}
        run_dumps = set(_dump_paths(r))
        missing = [rel for rel in expected_abs if rel not in run_dumps]
        if missing:
            _report("dump manifest", False,
                    f"{len(missing)} expected dump files missing from run: "
                    f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
            failures += 1
        else:
            mismatched = []
            for rel, digest in sorted(expected_abs.items()):
                actual = _sha256(os.path.join(r, rel))
                if actual != digest:
                    mismatched.append(rel)
            if mismatched:
                _report("dump manifest", False,
                        f"{len(mismatched)}/{len(expected_abs)} files differ from "
                        f"baseline checksums: {mismatched[:5]}{' ...' if len(mismatched) > 5 else ''}")
                failures += 1
            else:
                _report(f"dump manifest ({len(expected_abs)} files)", True)
    elif args.full:
        print("  (no baseline dump_manifest.sha256; --full needs both trees to "
              "keep the full dump set)", file=sys.stderr)

    # ---- 5. --full: byte-compare every dump file between the two trees -----
    if args.full:
        b_dumps = set(_dump_paths(b))
        r_dumps = set(_dump_paths(r))
        if b_dumps != r_dumps:
            only_b = sorted(b_dumps - r_dumps)
            only_r = sorted(r_dumps - b_dumps)
            _report("full dump set", False,
                    f"set differs (only baseline: {only_b[:3]}...; only run: "
                    f"{only_r[:3]}...)")
            failures += 1
        else:
            for rel in b_dumps:
                if _sha256(os.path.join(b, rel)) != _sha256(os.path.join(r, rel)):
                    _report(rel, False, "byte mismatch")
                    failures += 1
            if not failures:
                _report(f"full dump set ({len(b_dumps)} files)", True)

    print()
    if failures:
        print(f"FAIL: {failures} difference(s) found")
        return 1
    print("PASS: byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
