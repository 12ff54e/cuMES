#!/usr/bin/env python3
"""Compare one fixed Newton policy on a predefined axisymmetric input matrix."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time


POLICY = {
    "CUMES_NEWTON_PERIOD": "100",
    "CUMES_NEWTON_START": "100",
    "CUMES_NEWTON_STEPS": "32",
    "CUMES_NEWTON_BASIS": "32",
    "CUMES_NEWTON_FORWARD": "1",
    "CUMES_NEWTON_EPSILON": "1e-6",
    "CUMES_NEWTON_RATIO": "0.95",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(log, marker):
    decoder = json.JSONDecoder()
    result, errors = [], []
    for suffix in log.split(marker)[1:]:
        try:
            record, _ = decoder.raw_decode(suffix.lstrip())
            # Lowercase C printf nan/inf are invalid JSON and must not silently
            # become successful convergence or timing records.
            json.dumps(record, allow_nan=False)
            result.append(record)
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    return result, errors


def telemetry(gpu):
    command = [
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,"
        "power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def configured_convergence(stages, config):
    expected = list(zip(config["ns_array"], config["niter_array"],
                        config["ftol_array"]))
    return len(stages) == len(expected) and all(
        stage["ns"] == ns and stage["converged"]
        and 1 <= stage["iterations"] <= cap
        and len(stage["residual"]) == 3
        and all(math.isfinite(r) and 0 <= r < tolerance
                for r in stage["residual"])
        for stage, (ns, cap, tolerance) in zip(stages, expected)
    )


def run_case(args, case, input_path, variant, repetition, warmup, binary_hash):
    from validate import read_native_report

    config = json.loads(input_path.read_text())
    prefix = args.out / (
        f"{case['id']}-{'warm' if warmup else 'pair'}{repetition}-{variant}"
    )
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith("CUMES_")}
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment.update(POLICY if variant == "newton"
                       else {"CUMES_NEWTON_PERIOD": "0"})
    controls = {k: v for k, v in environment.items()
                if k.startswith("CUMES_") or k == "CUDA_VISIBLE_DEVICES"}
    identity = {
        "case": case["id"], "variant": variant, "repetition": repetition,
        "warmup": warmup, "input_sha256": digest(input_path),
        "binary_sha256": binary_hash, "environment": controls,
        "cpu": args.cpu, "timeout_seconds": args.timeout,
    }
    record_path = prefix.with_suffix(".json")
    if args.resume and record_path.exists():
        previous = json.loads(record_path.read_text())
        if all(previous.get(k) == v for k, v in identity.items()):
            return previous
        raise ValueError(f"cached run identity differs: {record_path}")
    if record_path.exists():
        raise ValueError(f"refusing to overwrite {record_path}; use --resume")

    command = [str(args.exe), str(input_path), "--output",
               str(prefix.with_suffix(".bin")), "--checkpoint",
               str(prefix.with_suffix(".ckpt"))]
    before = telemetry(args.gpu)
    start = time.perf_counter()
    timed_out = False
    with prefix.with_suffix(".log").open("w") as log:
        try:
            process = subprocess.run(command, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT,
                                     timeout=args.timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    elapsed = time.perf_counter() - start
    after = telemetry(args.gpu)
    log = prefix.with_suffix(".log").read_text(errors="replace")
    stages, stage_errors = records(log, "EXPERIMENT_STAGE ")
    trials, trial_errors = records(log, "EXPERIMENT_NEWTON ")
    native, native_error = None, None
    if prefix.with_suffix(".bin").exists():
        try:
            native = read_native_report(prefix.with_suffix(".bin"))
        except (ValueError, OSError, AssertionError) as error:
            native_error = str(error)
    complete = returncode == 0 and configured_convergence(stages, config)
    consistent = native is not None and len(native["stages"]) == len(stages)
    if consistent:
        consistent = all(
            all(a[key] == b[key] for key in
                ("ns", "iterations", "converged", "residual"))
            for a, b in zip(native["stages"], stages)
        )
    row = dict(identity, command=command, input=str(input_path),
               output=str(prefix.with_suffix(".bin")),
               checkpoint=str(prefix.with_suffix(".ckpt")),
               returncode=returncode, timed_out=timed_out,
               wall_seconds=elapsed, telemetry_before=before,
               telemetry_after=after, configured_stages=[
                   {"ns": ns, "cap": cap, "ftol": tol}
                   for ns, cap, tol in zip(config["ns_array"],
                                           config["niter_array"],
                                           config["ftol_array"])],
               stages=stages, trials=trials, native=native,
               stage_parse_errors=stage_errors, trial_parse_errors=trial_errors,
               native_error=native_error, report_consistent=consistent,
               full_configured_convergence=complete and consistent,
               device_ms=sum(x["device_ms"] for x in stages),
               extra_evaluations=sum(x["extra_evaluations"] for x in stages))
    write_json(record_path, row)
    print(case["id"], variant, repetition, "warm" if warmup else "measured",
          returncode, row["full_configured_convergence"],
          [x["iterations"] for x in stages], f"{row['device_ms']:.3f} ms",
          flush=True)
    return row


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows):
    result = {}
    for case in sorted({row["case"] for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        measured = [row for row in case_rows if not row["warmup"]]
        by_variant = {}
        for variant in ("baseline", "newton"):
            values = [row for row in measured if row["variant"] == variant]
            complete = [row for row in values
                        if row["full_configured_convergence"]]
            entry = {"runs": len(values), "converged": len(complete),
                     "failed_runs": [row["output"] for row in values
                                     if not row["full_configured_convergence"]],
                     "warmup_failures": [row["output"] for row in case_rows
                                         if row["variant"] == variant
                                         and row["warmup"]
                                         and not row["full_configured_convergence"]]}
            if complete:
                times = [row["device_ms"] for row in complete]
                median = statistics.median(times)
                trajectories = {
                    json.dumps([
                        {k: stage[k] for k in ("ns", "iterations", "converged",
                                              "residual", "restarts")}
                        for stage in row["native"]["stages"]], sort_keys=True)
                    for row in complete
                }
                entry.update(
                    median_device_ms=median, p95_device_ms=percentile(times, .95),
                    mad_device_ms=statistics.median(abs(t - median) for t in times),
                    median_wall_seconds=statistics.median(
                        row["wall_seconds"] for row in complete),
                    repeat_trajectory_exact=len(trajectories) == 1,
                    state_sha256=sorted({row["native"]["state_sha256"]
                                         for row in complete}),
                    iterations=[x["iterations"] for x in complete[0]["stages"]],
                    extra_evaluations=complete[0]["extra_evaluations"],
                )
            by_variant[variant] = entry
        pairs = []
        for repetition in sorted({row["repetition"] for row in measured}):
            pair = {row["variant"]: row for row in measured
                    if row["repetition"] == repetition}
            if len(pair) == 2 and all(row["full_configured_convergence"]
                                     for row in pair.values()):
                pairs.append(100 * (1 - pair["newton"]["device_ms"]
                                   / pair["baseline"]["device_ms"]))
        entry = {"variants": by_variant, "paired_gain_percent": pairs}
        # A candidate with any failure gets no overall timing gain claim.
        if (pairs and len(pairs) * 2 == len(measured)
                and all(row["full_configured_convergence"] for row in case_rows)):
            entry["median_gain_percent"] = statistics.median(pairs)
            if len(pairs) >= 5:
                rng = random.Random(20260908)
                bootstrap = [statistics.median(rng.choices(pairs, k=len(pairs)))
                             for _ in range(20000)]
                entry["bootstrap_ci95_percent"] = [
                    percentile(bootstrap, .025), percentile(bootstrap, .975)]
        result[case] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "paired"), default="screen")
    parser.add_argument("--pairs", type=int, default=5,
                        help="measured pairs in paired mode (default: 5)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.exe, args.manifest, args.out = (p.resolve() for p in
                                       (args.exe, args.manifest, args.out))
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be finite and positive")
    if not 5 <= args.pairs <= 50:
        parser.error("--pairs must be between 5 and 50")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    selected = set(args.cases or (case["id"] for case in manifest["cases"]))
    known = {case["id"] for case in manifest["cases"]}
    if selected - known:
        parser.error(f"unknown cases: {sorted(selected - known)}")
    binary_hash = digest(args.exe)
    write_json(args.out / "protocol.json", {
        "manifest": str(args.manifest), "manifest_sha256": digest(args.manifest),
        "binary_sha256": binary_hash, "phase": args.phase, "policy": POLICY,
        "cases": [case["id"] for case in manifest["cases"]
                  if case["id"] in selected],
        "warmups_per_variant": 2 if args.phase == "paired" else 0,
        "pairs": args.pairs if args.phase == "paired" else 1,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "metric": "sum of complete per-stage instrumented solver CUDA intervals",
    })
    rows = []
    for case in manifest["cases"]:
        if case["id"] not in selected:
            continue
        input_path = (args.manifest.parent / case["input"]).resolve()
        if digest(input_path) != case["sha256"]:
            raise ValueError(f"input differs from declared matrix: {input_path}")
        config = json.loads(input_path.read_text())
        if config.get("ntor", 0) != 0 or config.get("lfreeb", False):
            raise ValueError(f"requires fixed-boundary axisymmetry: {input_path}")
        repetitions = range(-2, args.pairs) if args.phase == "paired" else range(1)
        for repetition in repetitions:
            order = ("baseline", "newton") if repetition % 2 == 0 else (
                "newton", "baseline")
            for variant in order:
                rows.append(run_case(args, case, input_path, variant, repetition,
                                     repetition < 0, binary_hash))
                write_json(args.out / "samples.json", rows)
        write_json(args.out / "summary.json", summarize(rows))


if __name__ == "__main__":
    main()
