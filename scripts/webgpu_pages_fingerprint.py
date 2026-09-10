"""Fingerprint the source inputs and commands of a completed Ninja target."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def fingerprint(source, build, target, package_inputs):
    source, build = Path(source).resolve(), Path(build).resolve()

    def ninja(tool, *args):
        return subprocess.check_output(
            ["ninja", "-C", str(build), "-t", tool, *args], text=True
        )

    def absolute(path):
        return (build / path).resolve()

    # `inputs` supplies the configured graph; compiler-discovered headers live
    # separately in Ninja's deps log. Limit that log to this target's closure.
    inputs = {
        absolute(shlex.split(line)[0])
        for line in ninja("inputs", target, "build.ninja").splitlines()
        if line
    }
    used, compiler_records = False, 0
    for line in ninja("deps").splitlines():
        record = re.fullmatch(r"(.+): #deps \d+, deps mtime \d+ \((\w+)\)", line)
        if record:
            used = absolute(record[1]) in inputs
            if used:
                if record[2] != "VALID":
                    raise RuntimeError("Build the target before fingerprinting: " + record[1])
                compiler_records += 1
        elif used and line.startswith("    "):
            inputs.add(absolute(line[4:]))
    if not compiler_records:
        raise RuntimeError("No compiler dependencies found; build the target first")
    inputs.update(Path(path).resolve() for path in json.loads(Path(package_inputs).read_text()))

    files = {}
    for path in inputs:
        # Generated binaries embed Git provenance. Their source inputs and
        # build commands determine relevance, without making every commit a
        # deployment. Pinned toolchain changes are covered by the workflow.
        if not path.is_relative_to(source) or path.is_relative_to(build):
            continue
        files[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise RuntimeError("No source dependencies found")

    commands = []
    for line in ninja("commands", target).splitlines():
        command = []
        for token in shlex.split(line):
            if token.startswith(("-DCUMES_GIT_REVISION=", "-DCUMES_GIT_DIRTY=")):
                continue
            command.append(token.replace(str(build), "<build>").replace(str(source), "<source>"))
        commands.append(command)
    record = {"version": 1, "files": files, "commands": commands}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return digest, record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("build")
    parser.add_argument("target")
    parser.add_argument("package_inputs")
    parser.add_argument("output")
    args = parser.parse_args()
    digest, record = fingerprint(args.source, args.build, args.target, args.package_inputs)
    Path(args.output).write_text(digest + "\n")
    Path(args.output + ".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Pages fingerprint: {digest} ({len(record['files'])} source inputs)")
