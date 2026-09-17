#!/usr/bin/env python3
"""Run every selected free-boundary case, retaining failures and actual residuals."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.io import netcdf_file


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    binary = parser.add_mutually_exclusive_group(required=True)
    binary.add_argument("--cumes", type=Path)
    binary.add_argument("--vmec", type=Path)
    parser.add_argument("--case", action="append", choices=("diiid", "ncsx", "cth"))
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    executable = (args.cumes or args.vmec).resolve(strict=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CUMES_")}
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    failed = False
    for case in args.case or ("diiid", "ncsx", "cth"):
        directory = args.out.resolve() / case
        directory.mkdir(parents=True, exist_ok=False)
        source = args.inputs.resolve() / case
        name = "input.json" if args.cumes else "input." + case
        input_path = directory / name
        shutil.copyfile(
            source / ("cumes-input.json" if args.cumes else "vmec/" + name), input_path
        )
        command = [str(executable), name]
        if args.cumes:
            command += ["--output", "output.h5", "--checkpoint", "output.ckpt"]
        record = dict(
            command=command,
            executable_sha256=digest(executable),
            input_sha256=digest(input_path),
        )
        start = time.monotonic()
        with (directory / "run.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=directory,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                record["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
        record["wall_seconds"] = time.monotonic() - start
        record["converged"] = False
        if args.vmec and (directory / ("wout_" + case + ".nc")).exists():
            with netcdf_file(directory / ("wout_" + case + ".nc"), mmap=False) as f:
                value = lambda k: float(f.variables[k].data)
                record.update(
                    ier_flag=int(value("ier_flag")),
                    ftolv=value("ftolv"),
                    residuals=[value(k) for k in ("fsqr", "fsqz", "fsql")],
                )
            record["converged"] = (
                record.get("returncode") == 0
                and record["ier_flag"] == 0
                and all(
                    np.isfinite(x) and 0 <= x <= record["ftolv"]
                    for x in record["residuals"]
                )
            )
        elif args.cumes and (directory / "output.h5").exists():
            from compare import inspect_native

            checks, _, _, _, iterations = inspect_native(directory / "output.h5")
            record.update(checks=checks, iterations=iterations)
            record["converged"] = record.get("returncode") == 0 and all(checks.values())
            if record["converged"]:
                config = json.loads(input_path.read_text())
                for key in ("ns_array", "ftol_array"):
                    config[key] = config[key][-1:]
                config["niter_array"] = [4000]
                (directory / "replay-input.json").write_text(
                    json.dumps(config, indent=2) + "\n"
                )
                replay_command = [
                    str(executable),
                    "replay-input.json",
                    "--restart",
                    "output.ckpt",
                    "--output",
                    "replay.h5",
                ]
                with (directory / "replay.log").open("w") as log:
                    try:
                        replay = subprocess.run(
                            replay_command,
                            cwd=directory,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                        )
                        record["replay_returncode"] = replay.returncode
                        if (directory / "replay.h5").exists():
                            record["replay_checks"] = inspect_native(
                                directory / "replay.h5"
                            )[0]
                    except subprocess.TimeoutExpired:
                        record["replay_timed_out"] = True
                record["replay_converged"] = record.get(
                    "replay_returncode"
                ) == 0 and all(record.get("replay_checks", {"missing": False}).values())
        failed |= not record["converged"] or not record.get("replay_converged", True)
        (directory / "run.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        print(case, "converged" if record["converged"] else "NOT CONVERGED", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
