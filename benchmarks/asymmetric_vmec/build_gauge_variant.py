#!/usr/bin/env python3
"""Build an isolated CUDA-double m=1 threshold experiment from a Ninja build."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1e100:
        parser.error("threshold must be in (0, 1e100]")
    repo = Path(__file__).resolve().parents[2]
    build = args.build.resolve(strict=True)
    cache = (build / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo}" not in cache:
        parser.error("build belongs to a different source tree")
    if "CUMES_USE_FLOAT:BOOL=ON" in cache:
        parser.error("the gauge study requires CUDA double")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    header = out / "include/cumes/solver/control_policy.hpp"
    header.parent.mkdir(parents=True)
    text = (repo / "include/cumes/solver/control_policy.hpp").read_text()
    text, count = re.subn(
        r"(M1_GAUGE_RESIDUAL_THRESHOLD\s*=\s*)[^;]+",
        rf"\g<1>{args.threshold:.17g}",
        text,
    )
    if count != 1:
        raise ValueError("expected one gauge threshold")
    header.write_text(text)
    commands = subprocess.check_output(
        ["ninja", "-t", "commands", "cumes"], cwd=build, text=True
    ).splitlines()
    compile_command = shlex.split(
        next(
            line
            for line in commands
            if " -c " in line and "/src/solver_double.cu" in line
        )
    )
    compile_command.insert(1, "-I" + str(out / "include"))
    for flag, value in (
        ("-o", out / "solver.o"),
        ("-MT", out / "solver.o"),
        ("-MF", out / "solver.o.d"),
    ):
        compile_command[compile_command.index(flag) + 1] = str(value)
    # CMake wraps its link command in ': && ... && :'. Execute only the
    # actual argv, without a shell or modifying any production objects.
    link_command = shlex.split(
        next(line for line in reversed(commands) if " -o cumes " in line)
    )
    if link_command[:2] == [":", "&&"]:
        link_command = link_command[2:]
    if link_command[-2:] == ["&&", ":"]:
        link_command = link_command[:-2]
    link_command[link_command.index("-o") + 1] = str(out / "cumes")
    first_library = next(
        i for i, item in enumerate(link_command) if item.endswith(".a")
    )
    link_command.insert(first_library, str(out / "solver.o"))
    for command in (compile_command, link_command):
        subprocess.run(command, cwd=build, check=True)
    record = dict(
        threshold=args.threshold,
        compile=compile_command,
        link=link_command,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        executable_sha256=hashlib.sha256((out / "cumes").read_bytes()).hexdigest(),
    )
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
