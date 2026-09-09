#!/usr/bin/env python3
"""Replay completed matrix checkpoints at their original final tolerance."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run import digest, records, write_json
from validate import read_container, replay_state_comparison


def state_checks(source, replay):
    result = replay_state_comparison(source, replay, ntor=0)
    return dict(state_preserved=result["passed"],
                state_bit_exact=result["bit_exact"],
                numeric_state_exact=result["numeric_exact"],
                dependent_axis_zero_sign_changes=len(
                    result["dependent_axis_zero_sign_changes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    args.exe, args.samples, args.manifest, args.out = (p.resolve() for p in
        (args.exe, args.samples, args.manifest, args.out))
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "results.json").exists():
        parser.error("replay output already exists; use a fresh directory")
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith("CUMES_")}
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUMES_NEWTON_PERIOD="0")
    manifest = json.loads(args.manifest.read_text())
    samples = json.loads(args.samples.read_text())
    binary_hash = digest(args.exe)
    results = []
    for case in manifest["cases"]:
        input_path = (args.manifest.parent / case["input"]).resolve()
        if digest(input_path) != case["sha256"]:
            raise ValueError(f"input differs from declared matrix: {input_path}")
        config = json.loads(input_path.read_text())
        if config.get("ntor", 0) != 0 or config.get("lfreeb", False):
            raise ValueError(f"requires fixed-boundary axisymmetry: {input_path}")
        for key in ("ns_array", "niter_array", "ftol_array"):
            config[key] = config[key][-1:]
        replay_input = args.out / (case["id"] + "-final.json")
        write_json(replay_input, config)
        for variant in ("baseline", "newton"):
            matches = [row for row in samples if row["case"] == case["id"]
                       and row["variant"] == variant and not row["warmup"]
                       and row["full_configured_convergence"]]
            if not matches:
                results.append(dict(case=case["id"], variant=variant,
                                    passed=False, reason="no converged checkpoint"))
                continue
            sample = min(matches, key=lambda row: row["repetition"])
            # Native records retain absolute remote paths; the associated
            # files can also be read from a copied sample directory locally.
            checkpoint = args.samples.parent / Path(sample["checkpoint"]).name
            source_output = args.samples.parent / Path(sample["output"]).name
            prefix = args.out / f"{case['id']}-{variant}"
            command = [str(args.exe), str(replay_input), "--restart",
                       str(checkpoint), "--output", str(prefix.with_suffix(".bin"))]
            start = time.perf_counter()
            with prefix.with_suffix(".log").open("w") as log:
                try:
                    process = subprocess.run(command, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    returncode = None
            stages, errors = records(prefix.with_suffix(".log").read_text(),
                                     "EXPERIMENT_STAGE ")
            row = dict(case=case["id"], variant=variant, command=command,
                checkpoint=str(checkpoint), original_output=str(source_output),
                original_input_sha256=case["sha256"], replay_input=str(replay_input),
                binary_sha256=binary_hash, returncode=returncode,
                wall_seconds=time.perf_counter() - start, stages=stages,
                parse_errors=errors, passed=False,
                note="final-grid shape only for checkpoint verification; not a speed result")
            if returncode == 0 and prefix.with_suffix(".bin").exists():
                source, replay = read_container(source_output), read_container(
                    prefix.with_suffix(".bin"))
                final = replay["stages"]
                params = dict(source["params"], stages=replay["params"]["stages"])
                fixed_point = (len(final) == 1 and final[0]["ns"] == source["ns"]
                    and final[0]["iterations"] == 1 and final[0]["converged"]
                    and not final[0]["restarts"]
                    and all(0 <= r < config["ftol_array"][0]
                            for r in final[0]["residual"]))
                state = state_checks(source["state"], replay["state"])
                row.update(state)
                row.update(passed=bool(fixed_point and params == replay["params"]
                                      and state["state_preserved"]),
                    residual_exact=final[-1]["residual"] == source["stages"][-1]["residual"],
                    native_stages=final)
            results.append(row)
            write_json(prefix.with_suffix(".json"), row)
            write_json(args.out / "results.json", results)
            print(case["id"], variant, row["passed"],
                  [stage["iterations"] for stage in stages], flush=True)
    write_json(args.out / "results.json", results)
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
