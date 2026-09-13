#!/usr/bin/env python3
"""Prepare/run QH gauge, stopping-tolerance and resolution experiments."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
CASES = {
    "default-cold": {},
    "default-cold-step05": {"delt": 0.5},
    "normalized-cold": {},
    "later-cold": {},
    "later-cold-step05": {"delt": 0.5},
    "fixed-cold": {},
    "fixed-cold-step05": {"delt": 0.5},
    **{
        f"default-tol{power}": {"ftol_array": [1e-14, 10.0**-power]}
        for power in (10, 12, 14)
    },
    "default-radial101": {
        "ns_array": [25, 51, 101],
        "ftol_array": [1e-12, 1e-14, 1e-16],
    },
    "default-radial201": {
        "ns_array": [25, 51, 101, 201],
        "ftol_array": [1e-12, 1e-13, 1e-14, 1e-16],
    },
    "default-quadrature": {"ntheta": 40, "nzeta": 80},
    "default-angular": {"mpol": 10, "ntor": 15, "ntheta": 40, "nzeta": 80},
    "default-angular12": {"mpol": 12, "ntor": 15, "ntheta": 48, "nzeta": 96},
    "default-combined101": {
        "mpol": 10,
        "ntor": 15,
        "ntheta": 40,
        "nzeta": 80,
        "ns_array": [25, 51, 101],
        "ftol_array": [1e-12, 1e-14, 1e-16],
    },
    "fixed-warm-radial201": {"ns_array": [51, 101, 201], "ftol_array": [1e-16] * 3},
}
THRESHOLDS = dict(default=1e-6, normalized=4e-6, later=1e-10, fixed=1e100)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES, required=True)
    binaries = parser.add_mutually_exclusive_group()
    binaries.add_argument("--cumes", type=Path)
    binaries.add_argument("--vmec", type=Path)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if "fixed-warm-radial201" in args.case and not args.restart:
        parser.error("the warm study requires --restart with the converged ns=51 state")
    base = json.loads((HERE / "inputs/qh.json").read_text())
    base.update(tcon0=0.25, ftol_array=[1e-14, 1e-16])
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CUMES_")}
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    binary = args.cumes or args.vmec
    if binary:
        binary = binary.resolve(strict=True)
    failed = False
    for case in args.case:
        config = copy.deepcopy(base)
        config.update(CASES[case])
        config["niter_array"] = [80000] * len(config["ns_array"])
        policy = case.split("-", 1)[0]
        if args.vmec and (policy != "default" or "step05" in case):
            parser.error("Fortran runs use the standard policy and DELT")
        label = (
            case
            if not args.vmec
            else "vmec-"
            + ("base" if case == "default-cold" else case.removeprefix("default-"))
        )
        directory = args.out.resolve() / label
        directory.mkdir(parents=True, exist_ok=False)
        input_path = directory / ("input.qh" if args.vmec else "input.json")
        if args.vmec:
            import f90nml

            namelist = f90nml.read(HERE / "sources/input.qh")
            for key in ("ns_array", "ftol_array", "mpol", "ntor", "ntheta", "nzeta"):
                if key in config:
                    namelist["indata"][key] = config[key]
            namelist["indata"]["niter_array"] = [100000] * len(config["ns_array"])
            namelist["indata"]["niter"] = 100000
            namelist.write(input_path)
        else:
            write_json(input_path, config)
        record = dict(
            case=case,
            expected_gauge_threshold=THRESHOLDS[policy],
            input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        )
        if binary:
            command = [str(binary), str(input_path)]
            if args.cumes:
                command += ["--output", "output.h5", "--checkpoint", "output.ckpt"]
                if args.restart:
                    command += ["--restart", str(args.restart.resolve(strict=True))]
            record.update(
                command=command,
                executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
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
                    failed |= result.returncode != 0
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
                    failed = True
            record["wall_seconds"] = time.monotonic() - start
        write_json(directory / "run.json", record)
        print(label, record.get("returncode", "prepared/timeout"), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
