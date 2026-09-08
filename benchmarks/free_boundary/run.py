#!/usr/bin/env python3
"""Run paired complete free-boundary solves with native numerical checks."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from analyze import configured_stages, digest, inspect_output, summarize, write_json


def telemetry(gpu):
    command = ["nvidia-smi", "-i", str(gpu),
               "--query-gpu=timestamp,uuid,name,driver_version,pstate,clocks.sm,"
               "clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "error": str(error)}


def load_cases(path):
    if path.is_dir():
        cases = [{"id": item.stem, "input": str(item), "sha256": digest(item)}
                 for item in sorted(path.glob("*.json"))]
    else:
        manifest = json.loads(path.read_text())
        cases = [dict(case, input=str((path.parent / case["input"]).resolve()))
                 for case in manifest["cases"]]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("require at least one input and unique case identifiers")
    result = []
    for case in cases:
        if re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) is None:
            raise ValueError(f"unsafe case identifier: {case['id']}")
        input_path = Path(case["input"])
        actual_hash = digest(input_path)
        if actual_hash != case["sha256"]:
            raise ValueError(f"input differs from manifest: {input_path}")
        config = json.loads(input_path.read_text())
        if config.get("lfreeb") is not True:
            raise ValueError(f"input must select free boundary: {input_path}")
        dependencies = {}
        for name in ("mgrid_file", "coils_file", "makegrid_parameters_file"):
            if config.get(name):
                dependency = (input_path.parent / config[name]).resolve()
                dependencies[name] = {"path": str(dependency), "sha256": digest(dependency)}
        result.append({"id": case["id"], "input": str(input_path),
                       "sha256": actual_hash, "dependencies": dependencies,
                       "stages": configured_stages(config)})
    return result


def run_case(args, case, variant, repetition, warmup, binaries):
    prefix = args.out / f"{case['id']}-{'warm' if warmup else 'pair'}{repetition:02d}-{variant}"
    paths = {name: prefix.with_suffix(suffix) for name, suffix in
             (("record", ".json"), ("log", ".log"), ("output", ".bin"), ("checkpoint", ".ckpt"))}
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CUMES_")}
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if variant == "candidate" and args.candidate_direct:
        environment["CUMES_DISABLE_CUDA_GRAPHS"] = "1"
    controls = {key: value for key, value in environment.items()
                if key.startswith("CUMES_") or key == "CUDA_VISIBLE_DEVICES"}
    command = [binaries[variant]["path"], case["input"], "--output", str(paths["output"]),
               "--checkpoint", str(paths["checkpoint"])]
    if args.cpu is not None:
        command = ["taskset", "-c", str(args.cpu)] + command
    identity = {"case": case["id"], "variant": variant, "repetition": repetition,
                "warmup": warmup, "input_sha256": case["sha256"],
                "dependencies": case["dependencies"], "binary": binaries[variant],
                "environment": controls, "cpu": args.cpu, "timeout_seconds": args.timeout,
                "command": command, "cwd": str(Path(case["input"]).parent)}
    if paths["record"].exists():
        previous = json.loads(paths["record"].read_text())
        if not args.resume or previous["identity"] != identity:
            raise ValueError(f"existing run requires --resume with identical inputs: {paths['record']}")
        for name, expected in previous.get("artifact_sha256", {}).items():
            if not paths[name].exists() or digest(paths[name]) != expected:
                raise ValueError(f"cached artifact changed: {paths[name]}")
        return previous
    if any(path.exists() for path in paths.values()):
        raise ValueError(f"orphaned artifacts would be overwritten: {prefix}")
    row = {"identity": identity, "case": case["id"], "variant": variant,
           "repetition": repetition, "warmup": warmup, "input": case["input"],
           **{name: str(path) for name, path in paths.items()},
           "configured_stages": case["stages"], "state": "running", "successful": False}
    # A killed driver leaves its command/identity and raw files reviewable.
    # Resume retains that interrupted attempt; use a new directory to retry it.
    write_json(paths["record"], row)
    before = telemetry(args.gpu)
    started = time.perf_counter()
    returncode, failure = None, None
    with paths["log"].open("x") as log:
        try:
            # Popen.wait(timeout=...) polls with sleeps up to 50 ms. A watchdog
            # retains the cap while plain wait() measures process completion
            # without that quantization, which would swamp small speed gains.
            expired = threading.Event()
            with subprocess.Popen(command, cwd=identity["cwd"], env=environment,
                                  stdout=log, stderr=subprocess.STDOUT) as process:
                def stop_process():
                    expired.set()
                    process.kill()

                watchdog = threading.Timer(args.timeout, stop_process)
                watchdog.daemon = True
                watchdog.start()
                try:
                    returncode = process.wait()
                finally:
                    watchdog.cancel()
                    watchdog.join()
                if expired.is_set():
                    failure = f"process exceeded {args.timeout} seconds"
        except OSError as error:
            failure = str(error)
    wall_seconds = time.perf_counter() - started
    after = telemetry(args.gpu)
    log_text = paths["log"].read_text(errors="replace")
    timings = re.findall(r"^total device time:\s*(\S+)\s+ms\s*$", log_text, re.MULTILINE)
    device_ms, timing_error = None, None
    try:
        if len(timings) != 1:
            raise ValueError(f"expected one total device time record, found {len(timings)}")
        device_ms = float(timings[0])
        if not math.isfinite(device_ms) or device_ms <= 0:
            raise ValueError("device time must be finite and positive")
    except ValueError as error:
        device_ms, timing_error = None, str(error)
    inspection, native_error = {}, None
    try:
        inspection = inspect_output(case["input"], paths["output"], paths["checkpoint"])
    except (OSError, ValueError, AssertionError, KeyError) as error:
        native_error = str(error)
    row.update(inspection, state="finished", returncode=returncode, failure=failure,
               wall_seconds=wall_seconds, device_ms=device_ms, timing_error=timing_error,
               telemetry_before=before, telemetry_after=after, native_error=native_error,
               successful=returncode == 0 and failure is None and device_ms is not None and inspection.get("output_valid", False),
               artifact_sha256={name: digest(path) for name, path in paths.items()
                                if name != "record" and path.exists()})
    write_json(paths["record"], row)
    print(case["id"], variant, "warm" if warmup else "pair", repetition,
          "PASS" if row["successful"] else "FAIL", device_ms, "ms", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="directory of input JSON files or manifest with cases [{id,input,sha256}]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--candidate-direct", action="store_true",
                        help="disable candidate CUDA graphs to isolate transfer changes")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or args.warmups < 0 or args.gpu < 0 or (args.cpu is not None and args.cpu < 0):
        parser.error("pairs must be positive; warmups, GPU and CPU must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    args.out, args.inputs = args.out.resolve(), args.inputs.resolve()
    binaries = {variant: {"path": str(path.resolve()), "sha256": digest(path)}
                for variant, path in (("baseline", args.baseline), ("candidate", args.candidate))}
    cases = load_cases(args.inputs)
    if args.cases:
        unknown = set(args.cases) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.cases]
    protocol = {"schema": "cumes-free-boundary-paired-v1", "binaries": binaries,
                "inputs": str(args.inputs), "cases": cases,
                "manifest_sha256": digest(args.inputs) if args.inputs.is_file() else None,
                "harness_sha256": {
                    "run.py": digest(Path(__file__)),
                    "analyze.py": digest(Path(__file__).with_name("analyze.py")),
                    "native_parser": digest(Path(__file__).resolve().parents[1]
                                            / "axisymmetric_newton" / "validate.py")},
                "gpu": args.gpu, "cpu": args.cpu, "pairs": args.pairs,
                "warmups": args.warmups, "timeout_seconds": args.timeout,
                "candidate_direct": args.candidate_direct,
                "order": "warmups then measured pairs per case; alternate AB/BA from AB",
                "environment": "remove inherited CUMES_*; set CUDA_VISIBLE_DEVICES",
                "device_metric": "CLI total device time, rounded to 0.001 ms; sum of stage solver CUDA-event intervals including host vacuum gaps, excluding outer stage construction, interstage transfer, publication and output",
                "wall_metric": "subprocess elapsed time including startup, setup and output; telemetry and validation excluded",
                "bootstrap": "median paired reductions, 20000 resamples, seed 20260909; 95% CI for at least five pairs"}
    if args.out.exists() and any(args.out.iterdir()):
        existing = args.out / "protocol.json"
        if not args.resume or not existing.exists() or json.loads(existing.read_text()) != protocol:
            parser.error("nonempty output directory requires --resume with identical protocol")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "protocol.json", protocol)
    rows = []
    for case in cases:
        for warmup, count in ((True, args.warmups), (False, args.pairs)):
            for repetition in range(count):
                order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                for variant in order:
                    rows.append(run_case(args, case, variant, repetition, warmup, binaries))
                    write_json(args.out / "samples.json", rows)
        write_json(args.out / "summary.json", summarize(rows))


if __name__ == "__main__":
    main()
