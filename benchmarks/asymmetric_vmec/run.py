#!/usr/bin/env python3
"""Run the three asymmetric fixtures serially with cuMES or Fortran VMEC."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    executable = parser.add_mutually_exclusive_group(required=True)
    executable.add_argument("--cumes", type=Path)
    executable.add_argument("--vmec", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", action="append", choices=("qa", "heliotron", "qh"))
    parser.add_argument("--float-tolerance", type=float)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.float_tolerance is not None and (
            args.vmec or not 1e-6 <= args.float_tolerance < 1):
        parser.error("--float-tolerance requires cuMES and a value in [1e-6, 1)")
    manifest = json.loads((HERE / "manifest.json").read_text())
    binary = (args.cumes or args.vmec).resolve(strict=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("CUMES_")}
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for case in manifest["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        directory = args.out.resolve() / case["id"]
        directory.mkdir(parents=True, exist_ok=False)
        source = HERE / case["input" if args.cumes else "namelist"]
        expected = case["input_sha256" if args.cumes else "namelist_sha256"]
        if digest(source) != expected:
            raise ValueError(f"fixture differs from manifest: {source}")
        label = "cumes" if args.cumes else "vmec"
        record_path = directory / (label + "-run.json")
        if args.vmec:
            input_path = directory / ("input." + case["id"])
            shutil.copyfile(source, input_path)
            command = [str(binary), input_path.name]
        else:
            config = json.loads(source.read_text())
            if args.float_tolerance is not None:
                config["ftol_array"] = [args.float_tolerance] * len(config["ns_array"])
            input_path = directory / "cumes-input.json"
            write_json(input_path, config)
            command = [str(binary), str(input_path), "--output", "cumes.h5",
                       "--checkpoint", "cumes.ckpt"]
        record = dict(command=command, cwd=str(directory),
                      executable_sha256=digest(binary), input_sha256=digest(input_path))
        write_json(record_path, record)
        with (directory / (label + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
        record["returncode"] = result.returncode
        write_json(record_path, record)
        result.check_returncode()
        if args.cumes:
            config["ns_array"] = config["ns_array"][-1:]
            config["ftol_array"] = config["ftol_array"][-1:]
            config["niter_array"] = [100]
            replay = directory / "replay-input.json"
            write_json(replay, config)
            replay_command = [str(binary), str(replay), "--restart", "cumes.ckpt",
                              "--output", "replay.h5"]
            record["replay_command"] = replay_command
            record["replay_input_sha256"] = digest(replay)
            write_json(record_path, record)
            with (directory / "replay.log").open("w") as log:
                result = subprocess.run(replay_command, cwd=directory, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout, check=False)
            record["replay_returncode"] = result.returncode
            write_json(record_path, record)
            result.check_returncode()
        print(f"{case['id']}: {label} completed; {directory}", flush=True)


if __name__ == "__main__":
    main()
