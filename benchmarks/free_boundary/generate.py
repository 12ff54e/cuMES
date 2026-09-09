#!/usr/bin/env python3
"""Regenerate pinned free-boundary fixtures or materialize their field paths."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VMEC_COMMIT = "335ef66441d82980331d0062ab8d0398eff50818"
CUMES_COMMIT = "bcdd3dab5149dc0c5ec8608574f59f2d6932b398"
SOURCE_SPECS = {
    "solovev_mgrid": ("solovev_free_bdy.json", "cuMES", CUMES_COMMIT,
                       "inputs/free_bdy/solovev_free_bdy.json"),
    "solovev_embedded": ("solovev_free_bdy_embedded.json", "cuMES", CUMES_COMMIT,
                          "inputs/free_bdy/solovev_free_bdy_embedded.json"),
    "cth_multigrid": ("cth_like_free_bdy_multigrid.json", "VMEC++", VMEC_COMMIT,
                       "src/vmecpp/cpp/vmecpp/test_data/cth_like_free_bdy_multigrid.json"),
    "w7x_original": ("w7x_free_bdy_vac.json", "VMEC++", VMEC_COMMIT,
                     "examples/data/w7x_free_bdy_vac.json"),
}
DATA = {
    "mgrid_solovev.nc": {
        "bytes": 25215108,
        "sha256": "f17260704b9f6ca96c0f7202cc99dbfe8bdb625d1103d4597fd8ca5197b96fb1",
        "provenance": "VMEC++ v0.7.0 Git LFS object; src/vmecpp/cpp/vmecpp/test_data/mgrid_solovev.nc",
    },
    "mgrid_cth_like.nc": {
        "bytes": 35255952,
        "sha256": "41bb2cf388acbdee97d55bdcacd1ec373a0804f4a0cc2990081716326a95bfaa",
        "provenance": "VMEC++ v0.7.0 Git LFS object; src/vmecpp/cpp/vmecpp/test_data/mgrid_cth_like.nc",
    },
    "mgrid_w7x.nc": {
        "bytes": 88550256,
        "sha256": "ad612cc77434345d8d3fcc3fd9236fd48cbc5cd855040efc1bc69e737d1f94f8",
        "provenance": "Existing cuMES figure_data/w7x_free_boundary/mgrid_w7x.nc, dated 2026-08-31; generating binary revision unavailable, exact table pinned here.",
        "generating_parameters": "sources/makegrid_parameters_w7x.json",
        "generating_parameters_sha256": "dcad32ec1d0d79bc446011feefaeabfa59111d6281b9fe68558a62c8657b5d1a",
        "generating_coils_sha256": "a1810c5f12e7ac37114b1593f5df0ef559ca7f6b82a592645d90851f28efcc3d",
    },
    "coils.solovev": {
        "bytes": 985,
        "sha256": "50f11371c1528722fd9c967f072c0ab95ce79211ef745a263a785f7d72529823",
        "provenance": "VMEC++ v0.7.0 coils.solovev, vendored in vacuum-field edbbb2803203d9590af9d5444b1527078a0dc30e tests/data.",
        "tracked_copy": "sources/coils.solovev",
    },
}
CASES = [
    ("solovev_mgrid", "Solovev with precomputed field table", "solovev_mgrid",
     "mgrid_file", "mgrid_solovev.nc"),
    ("solovev_embedded", "Solovev with embedded MAKEGRID parameters", "solovev_embedded",
     "coils_file", "coils.solovev"),
    ("cth_multigrid", "CTH-like finite-pressure/current multigrid", "cth_multigrid",
     "mgrid_file", "mgrid_cth_like.nc"),
    ("w7x_original", "W7-X original negative-flux input", "w7x_original",
     "mgrid_file", "mgrid_w7x.nc"),
    ("w7x_positive_flux", "W7-X separately labeled positive-flux adaptation", "w7x_original",
     "mgrid_file", "mgrid_w7x.nc"),
]


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def generate():
    sources, originals, outputs, cases = {}, {}, {}, []
    for name, (filename, repository, commit, original_path) in SOURCE_SPECS.items():
        path = f"sources/{filename}"
        raw = (ROOT / path).read_bytes()
        originals[name] = json.loads(raw)
        sources[name] = {"file": path, "repository": repository, "commit": commit,
                         "original_path": original_path, "sha256": digest(raw)}
        if repository == "VMEC++":
            sources[name].update(tag="v0.7.0", license="sources/VMECpp-LICENSE.txt",
                                 repository_url="https://github.com/proximafusion/vmecpp")
    for case_id, title, source, dependency_key, dependency in CASES:
        original = originals[source]
        value = copy.deepcopy(original)
        value[dependency_key] = dependency
        value.pop("free_boundary_method", None)
        if case_id == "w7x_positive_flux":
            value["phiedge"] = 1.74
        changes = [{"key": key, "before": original.get(key), "after": value.get(key)}
                   for key in sorted(set(original) | set(value))
                   if original.get(key) != value.get(key)]
        for key in ("ns_array", "ftol_array", "niter_array", "delt", "nvacskip"):
            assert original[key] == value[key]
        allowed_changes = {dependency_key, "free_boundary_method"}
        if case_id == "w7x_positive_flux":
            allowed_changes.add("phiedge")
        assert all(change["key"] in allowed_changes for change in changes)
        assert value["lfreeb"] and not value["lasym"]
        path = f"inputs/{case_id}.json"
        raw = encode(value)
        outputs[path] = raw
        cases.append({
            "id": case_id, "title": title, "input": path, "sha256": digest(raw),
            "source": source, "changes": changes,
            "dependencies": {dependency_key: dependency},
            "expected": {key: value[key] for key in
                         ("lfreeb", "lasym", "nfp", "mpol", "ntor", "ntheta", "nzeta",
                          "ns_array", "ftol_array", "niter_array", "delt", "nvacskip", "phiedge")},
            "relationship": (
                "Same Solovev physics as the other Solovev field-source variant; setup coverage, not an independent geometry."
                if case_id.startswith("solovev") else
                "Supplement selected separately before its own runs; only phiedge sign differs from w7x_original. Preserve both outcomes."
                if case_id == "w7x_positive_flux" else "Original upstream physical workload."),
        })
    manifest = {
        "schema": "cumes-free-boundary-matrix-v1", "declared_date": "2026-09-09",
        "selection": "Four primary cases selected by source/data inspection before timing. Positive-flux W7-X separately declared before its own runs to reproduce the historical cuMES convention; retain the original negative-flux case and its failures. Fixture files were committed after initial runs because artifact-copy approval delayed generation; run protocols pin the actual immutable inputs.",
        "precision": "verify-double", "sources": sources, "data": DATA, "cases": cases,
        "policy": {
            "stages": "Preserve every source stage, cap and tolerance; W7-X's ns=51 is the source's native single stage.",
            "physics": "Preserve every physical value except the explicitly separate w7x_positive_flux sign adaptation.",
            "unsupported_selector": "Remove free_boundary_method=only_coils from W7-X. It is recognized but ignored by cuMES, which implements coil-only external fields.",
            "paths": "Portable templates contain data filenames. Materialize only declared path keys against a chosen data root before running; record resulting input hashes.",
            "failures": "Keep nonconvergence, invalid geometry, timeouts and regressions; do not alter inputs in response to outcomes.",
        },
    }
    outputs["manifest.json"] = encode(manifest)
    return outputs, manifest


def verify_data(path, spec):
    raw = path.read_bytes()
    if len(raw) != spec["bytes"] or digest(raw) != spec["sha256"]:
        raise ValueError(f"field dependency differs from pinned data: {path}")


def materialize(outputs, manifest, out, data_root):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite an existing materialized dataset: {out}")
    resolved = {}
    for name, spec in DATA.items():
        path = data_root / name
        if not path.exists() and spec.get("tracked_copy"):
            path = ROOT / spec["tracked_copy"]
        path = path.resolve()
        verify_data(path, spec)
        resolved[name] = str(path)
    result = copy.deepcopy(manifest)
    result["template_manifest_sha256"] = digest(outputs["manifest.json"])
    result["materialized_data_root"] = str(data_root.resolve())
    materialized = {}
    for case in result["cases"]:
        template_path = case["input"]
        value = json.loads(outputs[template_path])
        edits = []
        for key, name in case["dependencies"].items():
            edits.append({"key": key, "before": value[key], "after": resolved[name]})
            value[key] = resolved[name]
        raw = encode(value)
        case["template_sha256"] = case["sha256"]
        case["sha256"] = digest(raw)
        case["materialization_changes"] = edits
        materialized[template_path] = raw
    materialized["manifest.json"] = encode(result)
    for path, raw in materialized.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    print(f"Materialized {len(result['cases'])} cases: {out / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--materialize", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if bool(args.materialize) != bool(args.data_root) or (args.check and args.materialize):
        parser.error("use --check alone, or --materialize DIR with --data-root DIR")
    outputs, manifest = generate()
    if args.materialize:
        materialize(outputs, manifest, args.materialize, args.data_root)
    elif args.check:
        for path, raw in outputs.items():
            if (ROOT / path).read_bytes() != raw:
                raise ValueError(f"generated file is stale: {path}")
        print(f"Verified {len(manifest['cases'])} fixtures and their manifest")
    else:
        for path, raw in outputs.items():
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        print(f"Generated {len(manifest['cases'])} fixtures and their manifest")


if __name__ == "__main__":
    main()
