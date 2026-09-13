#!/usr/bin/env python3
"""Fetch hash-pinned original-VMEC free-boundary fixtures and prepare both decks."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

import f90nml
import numpy as np

HERE = Path(__file__).resolve().parent


def fetch(asset, work):
    target = work / asset["path"]
    if target.exists():
        data = target.read_bytes()
    else:
        with urlopen(asset["url"], timeout=90) as response:
            data = response.read()
        if "archive_sha256" in asset:
            if hashlib.sha256(data).hexdigest() != asset["archive_sha256"]:
                raise ValueError("archive checksum mismatch: " + asset["url"])
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                matches = [
                    m
                    for m in archive.getmembers()
                    if m.isfile() and Path(m.name).name == asset["member"]
                ]
                if len(matches) != 1 or matches[0].size != asset["bytes"]:
                    raise ValueError("expected one matching mgrid archive member")
                data = archive.extractfile(matches[0]).read()
    if (
        len(data) != asset["bytes"]
        or hashlib.sha256(data).hexdigest() != asset["sha256"]
    ):
        raise ValueError("fixture checksum mismatch: " + str(target))
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def prepare(entry, work, strict=False):
    case = entry["id"]
    source = entry["namelist"]
    directory = work / case
    namelist = f90nml.reads((directory / source).read_text().split("&END", 1)[0])
    nml = namelist["indata"]
    direct = (
        "lasym nfp mpol ntor ntheta nzeta ns_array ftol_array "
        "niter_array delt tcon0 phiedge lfreeb pmass_type am pres_scale "
        "spres_ped ncurr pcurr_type ac curtor bloat piota_type ai aphi "
        "nvacskip extcur"
    ).split()
    config = {key: nml[key] for key in direct if key in nml}
    for key in (
        "ns_array",
        "ftol_array",
        "niter_array",
        "am",
        "ac",
        "ai",
        "aphi",
        "extcur",
    ):
        if key in config and not isinstance(config[key], list):
            config[key] = [config[key]]
    config["mgrid_file"] = str((directory / nml["mgrid_file"]).resolve())
    config["gamma"] = nml.get("gamma", 0.0)
    ntor = nml["ntor"]
    for sources, target in [
        (("raxis_cc", "raxis"), "raxis_c"),
        (("raxis_cs",), "raxis_s"),
        (("zaxis_cc",), "zaxis_c"),
        (("zaxis_cs", "zaxis"), "zaxis_s"),
    ]:
        values = [0.0] * (ntor + 1)
        for key in sources:
            if key not in nml:
                continue
            start = nml.start_index.get(key, [0])[0] or 0
            for i, value in enumerate(np.atleast_1d(nml[key])):
                if value is not None and 0 <= start + i <= ntor:
                    values[start + i] = float(value)
            break
        config[target] = values
    for key in ("rbc", "zbs", "rbs", "zbc"):
        config[key] = []
        if key not in nml:
            continue
        n0, m0 = nml.start_index[key]
        for im, row in enumerate(nml[key]):
            for jn, value in enumerate(row):
                if (
                    value is not None
                    and value != 0
                    and m0 + im < nml["mpol"]
                    and abs(n0 + jn) <= ntor
                ):
                    config[key].append(dict(m=m0 + im, n=n0 + jn, value=value))
    if case == "diiid":
        # ZAXIS is the legacy sine family. Its n=0 entry is ignored by VMEC.
        # The published coarse stages intentionally exhaust their caps.
        # cuMES requires each configured stage to converge; use a shared
        # converging schedule in the comparison and retain the original deck.
        config["ftol_array"] = [1e-8, 1e-10, 1e-11, 1e-12]
        config["niter_array"] = [10000, 10000, 10000, 20000]
    if case == "cth":
        # The bundled synthetic mgrid has one group. VMEC ignores extcur(2).
        config["extcur"] = config["extcur"][:1]
        config["niter_array"] = [20000]
    if strict and case == "ncsx":
        config["ftol_array"][-1] = 1e-12
    config["tcon0"] = min(abs(nml.get("tcon0", 1.0)), 1.0) / (
        4 if nml.get("lasym", False) else 1
    )
    if not nml.get("lasym", False):
        for key in ("raxis_s", "zaxis_c", "rbs", "zbc"):
            config.pop(key, None)
    (directory / "cumes-input.json").write_text(json.dumps(config, indent=2) + "\n")
    # Original Fortran physics with the same radial schedule as cuMES.
    nml["mgrid_file"] = config["mgrid_file"]
    for key in ("ftol_array", "niter_array", "extcur"):
        nml[key] = config[key]
    nml["niter"] = max(config["niter_array"])
    ref = directory / "vmec"
    ref.mkdir(exist_ok=True)
    f90nml.Namelist({"indata": nml}).write(ref / ("input." + case), force=True)
    print(case, {k: config[k] for k in ["ns_array", "ftol_array", "tcon0", "extcur"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=("diiid", "ncsx", "cth"))
    parser.add_argument("--ncsx-strict", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    for entry in manifest["cases"]:
        if args.case and entry["id"] not in args.case:
            continue
        for asset in entry["assets"]:
            fetch(asset, args.out.resolve())
        prepare(entry, args.out.resolve(), args.ncsx_strict)


if __name__ == "__main__":
    main()
