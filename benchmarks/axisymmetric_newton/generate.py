#!/usr/bin/env python3
"""Regenerate the predeclared axisymmetric matrix without running a solver."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMON = {
    "ns_array": [5, 11, 55],
    "niter_array": [1000, 2000, 2000],
    "ftol_array": [1e-16, 1e-16, 1e-16],
}
SOURCES = {
    "solovev": {
        "file": "sources/solovev.json",
        "repository": "cuMES",
        "commit": "e36063dfff895a560cf2185830ebe2a6939873b9",
        "original_path": "inputs/solovev.json",
        "relationship": "Existing cuMES qualification case, not independent data.",
    },
    "vmecpp_circular": {
        "file": "sources/vmecpp_circular_tokamak.json",
        "repository": "https://github.com/proximafusion/vmecpp",
        "tag": "v0.7.0",
        "commit": "335ef66441d82980331d0062ab8d0398eff50818",
        "original_path": "src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json",
        "license": "sources/VMECpp-LICENSE.txt",
        "relationship": "Upstream circular-tokamak fixture with distinct geometry, flux and iota profile.",
    },
    "vmecpp_analytical": {
        "file": "sources/vmecpp_solovev_analytical.json",
        "repository": "https://github.com/proximafusion/vmecpp",
        "tag": "v0.7.0",
        "commit": "335ef66441d82980331d0062ab8d0398eff50818",
        "original_path": "src/vmecpp/cpp/vmecpp/test_data/solovev_analytical.json",
        "license": "sources/VMECpp-LICENSE.txt",
        "relationship": "Separate upstream analytical Solovev fixture; shares the Solovev family, but exercises prescribed current and substantial pressure.",
    },
}


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def generate():
    sources = copy.deepcopy(SOURCES)
    data = {}
    for key, source in sources.items():
        raw = (ROOT / source["file"]).read_bytes()
        source["sha256"] = sha256(raw)
        data[key] = json.loads(raw)
    outputs = {}
    cases = []

    def add(case_id, title, categories, changes=None, source="solovev", notes=None):
        original = data[source]
        value = copy.deepcopy(original)
        value.update(copy.deepcopy(COMMON))
        value.update(copy.deepcopy(changes or {}))
        edits = [
            {"key": key, "before": original.get(key), "after": value.get(key)}
            for key in sorted(set(original) | set(value))
            if original.get(key) != value.get(key)
        ]
        assert value.get("ntor", 0) == 0
        assert not value.get("lasym", False) and not value.get("lfreeb", False)
        assert value["niter_array"] == COMMON["niter_array"]
        assert value["ftol_array"] == COMMON["ftol_array"]
        assert len(value["ns_array"]) == 3
        assert all(entry["n"] == 0 for family in ["rbc", "zbs"] for entry in value[family])
        input_path = f"inputs/{case_id}.json"
        raw = encode(value)
        outputs[input_path] = raw
        mpol = value["mpol"]
        ntheta = max(value.get("ntheta", 0), 2 * mpol + 6)
        ntheta = 2 * (ntheta // 2)
        cases.append({
            "id": case_id,
            "title": title,
            "input": input_path,
            "categories": categories,
            "source": source,
            "provenance": "reference" if case_id == "00_solovev_reference" else (
                "generated_solovev_variation" if source == "solovev" else "adapted_upstream_fixture"
            ),
            "changes": edits,
            "notes": notes or [],
            "sha256": sha256(raw),
            "expected": {
                "ns_array": value["ns_array"],
                "niter_array": value["niter_array"],
                "ftol_array": value["ftol_array"],
                "mpol": mpol,
                "ntor": 0,
                "ntheta": ntheta,
                "nzeta": max(value.get("nzeta", 0), 1),
                "ncurr": value["ncurr"],
                "fixed_boundary": True,
            },
        })

    def coefficient(family, m, number):
        values = copy.deepcopy(data["solovev"][family])
        entry = next(item for item in values if item["n"] == 0 and item["m"] == m)
        entry["value"] = number
        return values

    add("00_solovev_reference", "Existing Solovev reference", ["reference"])
    add("01_solovev_circular", "Generated circular boundary", ["boundary"], {
        "rbc": [{"n": 0, "m": 0, "value": 4.0}, {"n": 0, "m": 1, "value": 1.0}],
        "zbs": [{"n": 0, "m": 1, "value": 1.0}],
    })
    add("02_solovev_elongation_low", "Reduced vertical extent", ["boundary"],
        {"zbs": coefficient("zbs", 1, 1.2)})
    add("03_solovev_elongation_high", "Increased vertical extent", ["boundary"],
        {"zbs": coefficient("zbs", 1, 2.0)})
    add("04_solovev_triangularity_stronger", "Stronger negative R m=2 shaping", ["boundary"],
        {"rbc": coefficient("rbc", 2, -0.18)})
    add("05_solovev_triangularity_reversed", "Reversed R m=2 shaping", ["boundary"],
        {"rbc": coefficient("rbc", 2, 0.068)})
    add("06_solovev_pressure_half", "Half pressure amplitude", ["pressure_amplitude"],
        {"am": [0.0625, -0.0625]})
    add("07_solovev_pressure_double", "Double pressure amplitude", ["pressure_amplitude"],
        {"am": [0.25, -0.25]})
    add("08_solovev_pressure_quadratic", "Pressure proportional to (1-s)^2", ["pressure_profile"],
        {"am": [0.125, -0.25, 0.125]})
    add("09_solovev_pressure_flatcore", "Pressure proportional to 1-s^2", ["pressure_profile"],
        {"am": [0.125, 0.0, -0.125]})
    add("10_solovev_mpol4", "Lower poloidal resolution", ["poloidal_resolution"], {"mpol": 4})
    add("11_solovev_mpol10", "Higher poloidal resolution", ["poloidal_resolution"], {"mpol": 10})
    add("12_solovev_radial99", "Higher final radial resolution", ["radial_resolution"],
        {"ns_array": [5, 11, 99]}, notes=["Only this case changes the common radial schedule; caps and tolerances remain unchanged."])
    add("13_solovev_ntheta36", "Denser angular quadrature at fixed spectral resolution", ["angular_resolution"],
        {"ntheta": 36})
    add("14_vmecpp_circular", "Upstream circular tokamak with sheared iota and zero pressure", ["upstream_fixture", "boundary", "iota_profile", "zero_pressure"],
        source="vmecpp_circular", notes=["Upstream one-stage ns=17/tol=1e-20/cap=3000 controls are deliberately replaced by the common three-stage schedule; physical inputs are retained."])
    analytical = data["vmecpp_analytical"]
    boundary = {}
    for family, offset in [("rbc", 0), ("zbs", 1)]:
        boundary[family] = [dict(entry, value=entry["value"] * (-1) ** (entry["m"] + offset))
                            for entry in analytical[family]]
    boundary["raxis_c"] = [4.0]
    add("15_vmecpp_analytical_ncurr1", "Upstream analytical Solovev with prescribed current", ["upstream_fixture", "current_closure", "pressure_amplitude", "poloidal_resolution"],
        boundary, source="vmecpp_analytical", notes=[
            "Poloidal coordinate reflection theta_old=pi-theta_new gives R_m_new=(-1)^m R_m_old and Z_m_new=(-1)^(m+1) Z_m_old. It preserves the geometric boundary point set and supplies cuMES's negative-Jacobian orientation.",
            "The upstream zero R-axis seed is replaced by R=4, inside the boundary. This is an initialization adaptation, not an independently converged axis.",
            "Upstream ns=31/tol=1e-16/cap=2000 controls are deliberately replaced by the common three-stage schedule. Current, pressure, flux, mpol13 and delt1.0 are retained.",
            "This is an adapted upstream Solovev-family fixture, not an unmodified independent equilibrium or a copy of the cuMES ncurr0 input.",
        ])
    assert len(cases) == 16
    manifest = {
        "schema": "cumes-axisymmetric-newton-matrix-v1",
        "declared_date": "2026-09-08",
        "selection": "All sixteen cases fixed by input/code inspection before solver timing. Keep failures, caps and regressions; do not tune or filter cases after outcomes.",
        "precision": "verify-double",
        "newton_policy": {
            "difference": "forward", "krylov_vectors": 32, "basis_vectors": 32,
            "period": 100, "start": 100, "epsilon": 1e-6, "accept_merit_ratio": 0.95,
        },
        "common_stage_controls": COMMON,
        "sources": sources,
        "cases": cases,
    }
    outputs["manifest.json"] = encode(manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated bytes without writing.")
    args = parser.parse_args()
    mismatch = []
    for name, raw in generate().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_bytes() != raw:
                mismatch.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    if mismatch:
        raise SystemExit("Generated content differs: " + ", ".join(mismatch))
    print("Checked" if args.check else "Generated", "16 predeclared inputs and manifest")


if __name__ == "__main__":
    main()
