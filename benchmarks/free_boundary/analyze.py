#!/usr/bin/env python3
"""Check free-boundary native outputs and summarize retained paired runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axisymmetric_newton"))
from validate import (FAMILIES, FULL_FIELDS, HALF_FIELDS, read_container,
                      read_native_report, same_bits, source_hash)


def write_json(path, value):
    """Replace a JSON document atomically; callers control overwrite policy."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     prefix=path.name + ".", suffix=".tmp") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def configured_stages(config):
    arrays = [config[name] for name in ("ns_array", "niter_array", "ftol_array")]
    if not arrays[0] or len({len(values) for values in arrays}) != 1:
        raise ValueError("require equally sized, nonempty explicit stage arrays")
    if config.get("n_grids", len(arrays[0])) != len(arrays[0]):
        raise ValueError("n_grids does not match the configured stage arrays")
    return [dict(ns=ns, max_iter=cap, ftol=tol)
            for ns, cap, tol in zip(*arrays)]


def numerical_fingerprint(native):
    """Hash every coefficient/field byte and the numerical RunReport."""
    def array_record(values):
        return {"shape": list(values.shape),
                "sha256": hashlib.sha256(values.tobytes()).hexdigest()}

    stages = [dict(stage, residual_bits=struct.pack("<3d", *stage["residual"]).hex())
              for stage in native["stages"]]
    return {
        "shape": {key: native[key] for key in
                  ("ns", "mnmax", "ntheta", "nzeta", "precision")},
        "state": {name: array_record(values)
                  for name, values in zip(FAMILIES, native["state"])},
        "half_fields": {name: array_record(values)
                        for name, values in native["half_fields"].items()},
        "full_fields": {name: array_record(values)
                        for name, values in native["full_fields"].items()},
        "report": {"status": native["status"],
                   "total_iterations": native["total_iterations"], "stages": stages},
        "normalized_input_sha256": canonical_hash(native["params"]),
    }


def inspect_output(input_path, output_path, checkpoint_path):
    raw = Path(input_path).read_bytes()
    config = json.loads(raw)
    targets = configured_stages(config)
    native = read_container(output_path)
    if native["checkpoint"]:
        raise ValueError("native result is a checkpoint")
    # Keep the shared report reader as the public JSON representation.
    report = read_native_report(output_path)
    stages, params = native["stages"], native["params"]
    checks = {
        "free_boundary": config.get("lfreeb") is True and params["lfreeb"],
        "input_hash_matches": native["provenance"].get("input_hash") == source_hash(raw),
        "stage_schedule_preserved": params["stages"] == targets,
        "all_stages_converged": native["status"] == 0 and len(stages) == len(targets)
        and all(stage["ns"] == target["ns"] and stage["converged"]
                and 1 <= stage["iterations"] <= target["max_iter"]
                and all(math.isfinite(x) and 0 <= x < target["ftol"]
                        for x in stage["residual"])
                for stage, target in zip(stages, targets)),
        "iteration_total_consistent": native["total_iterations"] == sum(
            stage["iterations"] for stage in stages),
        "restart_indices_valid": all(
            stage["restarts"] == sorted(stage["restarts"])
            and all(1 <= index <= stage["iterations"] for index in stage["restarts"])
            for stage in stages),
        "shape_consistent": native["ns"] == targets[-1]["ns"]
        and native["mnmax"] == params["mpol"] * (params["ntor"] + 1)
        and native["ntheta"] == params["ntheta"]
        and native["nzeta"] == params["nzeta"],
        "state_finite": bool(np.all(np.isfinite(native["state"]))),
        "fields_complete_finite": set(native["half_fields"]) == set(HALF_FIELDS)
        and set(native["full_fields"]) == set(FULL_FIELDS)
        and all(np.all(np.isfinite(values)) for group in ("half_fields", "full_fields")
                for values in native[group].values()),
    }
    fields = native["half_fields"]
    checks["positive_oriented_jacobian"] = bool(
        "sqrtg" in fields and np.all(-fields["sqrtg"] > 0))
    if set(fields) == set(HALF_FIELDS):
        b_squared = sum(fields[a] * fields[b] for a, b in
                        zip(("bsups", "bsupu", "bsupv"), ("bsubs", "bsubu", "bsubv")))
        checks["nonnegative_magnetic_energy"] = bool(
            np.all(np.isfinite(b_squared)) and np.all(b_squared >= 0))
    else:
        checks["nonnegative_magnetic_energy"] = False
    checkpoint = read_container(checkpoint_path)
    checks["checkpoint_matches_native"] = (
        checkpoint["checkpoint"] and checkpoint["precision"] == native["precision"]
        and checkpoint["params"] == params and same_bits(checkpoint["state"], native["state"]))
    return {"native": report, "checks": {k: bool(v) for k, v in checks.items()},
            "numerical_fingerprint": numerical_fingerprint(native),
            "output_valid": all(checks.values())}


def compare_pair(baseline, candidate):
    before = baseline.get("numerical_fingerprint")
    after = candidate.get("numerical_fingerprint")
    if before is None or after is None:
        return {"exact": False, "reason": "one or both native outputs unavailable"}
    checks = {key: before[key] == after[key] for key in before}
    return {"exact": all(checks.values()), "checks": checks}


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    median = statistics.median(values)
    return {"median": median, "p95": percentile(values, .95),
            "mad": statistics.median(abs(value - median) for value in values),
            "min": min(values), "max": max(values)}


def paired_reduction(values):
    result = {"values_percent": values, "median_percent": statistics.median(values)}
    if len(values) >= 5:
        rng = random.Random(20260909)
        bootstrap = [statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(20000)]
        result["bootstrap_ci95_percent"] = [percentile(bootstrap, .025),
                                             percentile(bootstrap, .975)]
    return result


def summarize(rows):
    result = {}
    for case in sorted({row["case"] for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        measured = [row for row in case_rows if not row["warmup"]]
        variants = {}
        for variant in ("baseline", "candidate"):
            selected = [row for row in measured if row["variant"] == variant]
            valid = [row for row in selected if row.get("successful", False)]
            variants[variant] = {
                "runs": len(selected), "successful": len(valid),
                "failures": [row["record"] for row in selected if not row.get("successful")],
                "warmup_failures": [row["record"] for row in case_rows
                                    if row["variant"] == variant and row["warmup"]
                                    and not row.get("successful")],
            }
            if valid:
                variants[variant].update(
                    device_ms=distribution([row["device_ms"] for row in valid]),
                    wall_seconds=distribution([row["wall_seconds"] for row in valid]),
                    repeat_numerical_exact=len({canonical_hash(row["numerical_fingerprint"])
                                               for row in valid}) == 1,
                    stages=valid[0]["native"]["stages"])
        pairs = []
        for warmup, repetition in sorted({(row["warmup"], row["repetition"])
                                          for row in case_rows}):
            pair = {row["variant"]: row for row in case_rows
                    if (row["warmup"], row["repetition"]) == (warmup, repetition)}
            complete = set(pair) == {"baseline", "candidate"}
            entry = {"warmup": warmup, "repetition": repetition, "complete": complete}
            if complete:
                entry.update(compare_pair(pair["baseline"], pair["candidate"]))
                entry["successful"] = all(row.get("successful", False) for row in pair.values())
                if entry["successful"]:
                    for metric in ("device_ms", "wall_seconds"):
                        entry[metric + "_reduction_percent"] = 100 * (
                            1 - pair["candidate"][metric] / pair["baseline"][metric])
            pairs.append(entry)
        eligible = bool(measured) and all(
            pair["complete"] and pair.get("successful") and pair.get("exact") for pair in pairs)
        entry = {"variants": variants, "pairs": pairs, "timing_claim_eligible": eligible,
                 "all_paired_numerical_exact": bool(pairs) and all(pair.get("exact") for pair in pairs)}
        if eligible:
            for metric in ("device_ms", "wall_seconds"):
                entry[metric + "_reduction"] = paired_reduction([
                    pair[metric + "_reduction_percent"] for pair in pairs if not pair["warmup"]])
        result[case] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out", type=Path,
                        help="default: print JSON; existing files require --replace")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = summarize(json.loads(args.samples.read_text()))
    if args.out:
        if args.out.exists() and not args.replace:
            parser.error("output exists; use --replace to regenerate this derived summary")
        write_json(args.out, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
