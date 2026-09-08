#!/usr/bin/env python3
"""Scientific checks for axisymmetric Newton experiments; no solver calls.

Only current native-v8/checkpoint-v6 containers are accepted. Convergence is
judged from the native RunReport, never the progress table. VMEC++ is an
independent diagnostic and does not replace cuMES's discrete residual gates.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


FAMILIES = ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")
HALF_FIELDS = ("sqrtg", "bsups", "bsupu", "bsupv", "bsubs", "bsubu", "bsubv")
FULL_FIELDS = ("jsups", "jsupu", "jsupv", "jsubs", "jsubu", "jsubv")


class Reader:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.offset = 0

    def take(self, size):
        if size < 0 or size > len(self.data) - self.offset:
            raise ValueError(f"{self.path}: truncated or invalid payload")
        value = self.data[self.offset:self.offset + size]
        self.offset += size
        return value

    def read(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def count(self, limit=1 << 20):
        value, = self.read("i")
        if not 0 <= value <= limit:
            raise ValueError(f"{self.path}: invalid length {value}")
        return value

    def string(self):
        return self.take(self.count(1 << 24)).decode("utf-8")

    def array(self, count, dtype="<f8"):
        dtype = np.dtype(dtype)
        return np.frombuffer(self.take(count * dtype.itemsize), dtype=dtype)

    def vector(self, dtype="<f8"):
        return self.array(self.count(), dtype).tolist()

    def finish(self):
        if self.offset != len(self.data):
            raise ValueError(f"{self.path}: unexpected trailing bytes")


def input_record(reader):
    result = dict(zip(("mpol", "ntor", "nfp", "ntheta", "nzeta", "ncurr"),
                      reader.read("6i")))
    result.update(zip(("delt", "phiedge", "pres_scale", "adiabatic_index",
                       "spres_ped", "bloat", "curtor", "tcon0"), reader.read("8d")))
    for name in ("schema", "pmass_type", "piota_type", "pcurr_type"):
        result[name] = reader.string()
    for name in ("am", "ac", "ai", "aphi", "raxis_c", "zaxis_s"):
        result[name] = reader.vector()
    result["stages"] = []
    for _ in range(reader.count()):
        ns, cap, tolerance = reader.read("iid")
        result["stages"].append({"ns": ns, "max_iter": cap, "ftol": tolerance})
    for name in ("rbc", "zbs"):
        m, n, value = reader.vector("<i4"), reader.vector("<i4"), reader.vector()
        if not len(m) == len(n) == len(value):
            raise ValueError(f"{reader.path}: inconsistent boundary lengths")
        result[name] = [dict(m=mm, n=nn, value=v) for mm, nn, v in zip(m, n, value)]
    for name in ("rbcc", "rbss", "zbsc", "zbcs"):
        result[name] = reader.vector()
    lfreeb, result["nvacskip"] = reader.read("2i")
    result["lfreeb"] = bool(lfreeb)
    result["mgrid_file"] = reader.string()
    result["extcur"] = reader.vector()
    result["coils_file"] = reader.string()
    result["makegrid_parameters_file"] = reader.string()
    present, = reader.read("i")
    result["makegrid_parameters"] = None
    if present:
        normalized, symmetric, nfp = reader.read("3i")
        r_min, r_max = reader.read("2d")
        nr, = reader.read("i")
        z_min, z_max = reader.read("2d")
        nz, nphi = reader.read("2i")
        result["makegrid_parameters"] = dict(
            normalize_by_currents=bool(normalized),
            assume_stellarator_symmetry=bool(symmetric), number_of_field_periods=nfp,
            r_grid_minimum=r_min, r_grid_maximum=r_max, number_of_r_grid_points=nr,
            z_grid_minimum=z_min, z_grid_maximum=z_max, number_of_z_grid_points=nz,
            number_of_phi_grid_points=nphi)
    return result


def read_container(path):
    reader = Reader(path)
    magic = reader.take(8)
    version, = reader.read("i")
    checkpoint = magic == b"CUMECKP1"
    if (magic, version) not in ((b"CUMES001", 8), (b"CUMECKP1", 6)):
        raise ValueError(f"{path}: require native v8 or checkpoint v6")
    precision = reader.read("i")[0] if checkpoint else None
    ns, mnmax = reader.read("2i")
    if ns < 2 or mnmax < 1:
        raise ValueError(f"{path}: invalid state dimensions")
    state_bytes = reader.take(6 * ns * mnmax * 8)
    state = np.frombuffer(state_bytes, dtype="<f8").reshape(6, mnmax, ns)
    result = dict(path=str(Path(path).resolve()), version=version, ns=ns,
                  mnmax=mnmax, checkpoint=checkpoint, state=state,
                  sha256=hashlib.sha256(reader.data).hexdigest(),
                  state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                  half_fields={}, full_fields={}, stages=[], provenance={})
    if not checkpoint:
        ntheta, nzeta = reader.read("2i")
        if ntheta < 0 or nzeta < 0 or bool(ntheta) != bool(nzeta):
            raise ValueError(f"{path}: invalid field dimensions")
        result.update(ntheta=ntheta, nzeta=nzeta)
        for names, key, surfaces in ((HALF_FIELDS, "half_fields", ns - 1),
                                     (FULL_FIELDS, "full_fields", ns)):
            if ntheta:
                for name in names:
                    result[key][name] = reader.array(surfaces * nzeta * ntheta).reshape(
                        surfaces, nzeta, ntheta)
        precision, status, iterations, stage_count = reader.read("4i")
        if not 0 <= stage_count <= 1 << 20:
            raise ValueError(f"{path}: invalid stage count")
        result.update(status=status, total_iterations=iterations)
        provenance = result["provenance"]
        provenance["revision"] = reader.string()
        provenance["dirty"] = bool(reader.read("B")[0])
        for name in ("build_type", "precision_policy", "compile_flags", "input_path",
                     "input_hash", "gpu_name", "driver", "runtime", "toolkit"):
            provenance[name] = reader.string()
        for _ in range(stage_count):
            radial_size, count, converged = reader.read("iiB")
            residual = list(reader.read("3d"))
            restarts = reader.vector("<i4")
            if converged not in (0, 1):
                raise ValueError(f"{path}: invalid convergence flag")
            result["stages"].append(dict(ns=radial_size, iterations=count,
                                         converged=bool(converged), residual=residual,
                                         restarts=restarts))
    result["precision"] = precision
    result["params"] = input_record(reader)
    reader.finish()
    return result


def read_native_report(path):
    """JSON-safe native RunReport and provenance; no field/state arrays."""
    result = read_container(path)
    if result["checkpoint"]:
        raise ValueError(f"{path}: checkpoint has no RunReport")
    report = {key: value for key, value in result.items()
              if key not in ("state", "half_fields", "full_fields")}
    json.dumps(report, allow_nan=False)
    return report


def source_hash(data):
    # Match src/main.cu exactly, including its historical nonstandard seed.
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def same_bits(a, b):
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def metrics(candidate, baseline):
    if candidate.shape != baseline.shape:
        raise ValueError(f"comparison shape mismatch: {candidate.shape}, {baseline.shape}")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(baseline)):
        return {"finite": False, "max_abs": None, "relative_l2": None}
    difference = candidate - baseline
    error_norm, base_norm = float(np.linalg.norm(difference)), float(np.linalg.norm(baseline))
    return dict(finite=True, max_abs=float(np.max(np.abs(difference), initial=0)),
                absolute_l2=error_norm, baseline_l2=base_norm,
                relative_l2=error_norm / base_norm if base_norm else
                (0.0 if error_norm == 0 else None))


def output_checks(input_path, native, checkpoint_path=None, replay_path=None):
    raw = Path(input_path).read_bytes()
    config = json.loads(raw)
    params, state = native["params"], native["state"]
    checks = {}
    checks["double_native_output"] = not native["checkpoint"] and native["precision"] == 0
    checks["axisymmetric_fixed_boundary"] = (params["ntor"] == 0 and params["nzeta"] == 1
                                                 and not params["lfreeb"])
    checks["input_bytes_match_provenance"] = native["provenance"].get("input_hash") == source_hash(raw)
    if not (len(config["ns_array"]) == len(config["niter_array"]) == len(config["ftol_array"])):
        raise ValueError(f"{input_path}: inconsistent stage array lengths")
    configured = [dict(ns=ns, max_iter=cap, ftol=tol) for ns, cap, tol in
                  zip(config["ns_array"], config["niter_array"], config["ftol_array"])]
    checks["configured_schedule_preserved"] = params["stages"] == configured
    checks["resolved_shape_consistent"] = (native["mnmax"] == params["mpol"] * (params["ntor"] + 1)
                                              and native["ns"] == configured[-1]["ns"]
                                              and native["ntheta"] == params["ntheta"]
                                              and native["nzeta"] == params["nzeta"])
    checks["all_stages_converged"] = (native["status"] == 0 and
        len(native["stages"]) == len(configured) and all(
            stage["ns"] == target["ns"] and stage["converged"] and
            1 <= stage["iterations"] <= target["max_iter"] and
            all(math.isfinite(x) and 0 <= x < target["ftol"] for x in stage["residual"])
            for stage, target in zip(native["stages"], configured)))
    checks["total_iterations_consistent"] = native["total_iterations"] == sum(
        stage["iterations"] for stage in native["stages"])
    checks["restart_indices_valid"] = all(
        all(1 <= i <= stage["iterations"] for i in stage["restarts"]) and
        stage["restarts"] == sorted(stage["restarts"]) for stage in native["stages"])
    checks["state_finite"] = bool(np.all(np.isfinite(state)))
    checks["scientific_fields_complete_finite"] = (
        len(native["half_fields"]) == 7 and len(native["full_fields"]) == 6 and
        all(np.all(np.isfinite(a)) for group in ("half_fields", "full_fields")
            for a in native[group].values()))
    boundary = np.stack([np.asarray(params[name]) for name in ("rbcc", "zbsc", "rbss", "zbcs")])
    checks["fixed_boundary_exact"] = same_bits(state[[0, 1, 3, 4], :, -1], boundary)
    if params["ntor"] == 0:
        checks["null_parities_zero"] = bool(np.all(state[3:] == 0) and
                                                  np.all(state[1:3, 0] == 0))
        checks["dependent_m1_axis_exact"] = params["mpol"] < 2 or same_bits(state[:, 1, 0], state[:, 1, 1])
    if "bsups" in native["half_fields"]:
        checks["radial_magnetic_field_zero"] = bool(np.all(native["half_fields"]["bsups"] == 0))
    optional = {"checkpoint": None, "replay": None}
    if checkpoint_path is not None:
        checkpoint = read_container(checkpoint_path)
        valid = (checkpoint["checkpoint"] and checkpoint["precision"] == 0 and
                 same_bits(state, checkpoint["state"]) and params == checkpoint["params"])
        checks["native_checkpoint_exact"] = valid
        optional["checkpoint"] = dict(path=checkpoint["path"], sha256=checkpoint["sha256"], passed=valid)
    if replay_path is not None:
        replay = read_container(replay_path)
        stages = replay["stages"]
        same_physics = all(value == replay["params"].get(key)
                           for key, value in params.items() if key != "stages")
        valid = (not replay["checkpoint"] and replay["status"] == 0 and
                 replay["precision"] == 0 and len(stages) == 1 and
                 replay["total_iterations"] == 1 and same_physics and
                 replay["params"]["stages"] == params["stages"][-1:] and
                 stages[0]["ns"] == native["ns"] and stages[0]["iterations"] == 1 and
                 stages[0]["converged"] and not stages[0]["restarts"] and
                 all(math.isfinite(x) and 0 <= x < params["stages"][-1]["ftol"]
                     for x in stages[0]["residual"]) and
                 same_bits(state, replay["state"]))
        checks["replay_iteration1_original_tolerance_state_exact"] = valid
        optional["replay"] = dict(path=replay["path"], stages=stages, passed=valid,
            # A fresh checkpoint epoch can recompute norm factors differently;
            # its own original-tolerance convergence is the required check.
            residual_triple_exact=bool(stages and
                stages[0]["residual"] == native["stages"][-1]["residual"]),
            original_residual=native["stages"][-1]["residual"])
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(passed=all(checks.values()), checks=checks,
                failed_checks=[key for key, value in checks.items() if not value],
                optional_checks=optional, report={key: value for key, value in native.items()
                    if key not in ("state", "half_fields", "full_fields")})


def compare_states(candidate, baseline):
    return {name: metrics(candidate[i, :, 1:], baseline[i, :, 1:])
            for i, name in enumerate(FAMILIES)}


def compare_surfaces(candidate, baseline, ntheta=256):
    """Physical Fourier synthesis away from the dependent axis row.

    State coefficients already include odd-m radial scaling. Reapplying sqrt(s)
    or the orthonormal basis factor would distort this geometry comparison.
    Surfaces are compared at equal native theta, so lambda gauge differences
    can move the labels even when the underlying geometric surfaces agree.
    """
    mpol = candidate.shape[1]
    theta = 2 * math.pi * np.arange(max(ntheta, 8 * mpol)) / max(ntheta, 8 * mpol)
    phase = np.arange(mpol)[:, None] * theta[None, :]
    cosine, sine = np.cos(phase), np.sin(phase)
    r_candidate, r_baseline = candidate[0, :, 1:].T @ cosine, baseline[0, :, 1:].T @ cosine
    z_candidate, z_baseline = candidate[1, :, 1:].T @ sine, baseline[1, :, 1:].T @ sine
    displacement = np.hypot(r_candidate - r_baseline, z_candidate - z_baseline)
    return dict(theta_points=len(theta), axis_excluded=True,
                comparison_coordinates="equal native poloidal angle; gauge dependent",
                r=metrics(r_candidate, r_baseline), z=metrics(z_candidate, z_baseline),
                displacement_max=float(displacement.max()),
                displacement_rms=float(np.sqrt(np.mean(displacement ** 2))))


def physical_diagnostics(native):
    fields = native["half_fields"]
    jacobian = np.abs(fields["sqrtg"])
    b_squared = sum(fields[contra] * fields[covar] for contra, covar in
                    zip(("bsups", "bsupu", "bsupv"), ("bsubs", "bsubu", "bsubv")))
    weight = np.sum(jacobian, axis=(1, 2))
    if not np.all(weight > 0):
        raise ValueError(f"{native['path']}: degenerate flux-surface volume")
    surface_mean = np.sum(b_squared * jacobian, axis=(1, 2)) / weight
    volume = float(4 * math.pi ** 2 * np.mean(jacobian))
    return dict(b_squared=b_squared, flux_surface_mean_b_squared=surface_mean,
                volume=volume, min_b_squared=float(np.min(b_squared)),
                min_oriented_jacobian=float(np.min(-fields["sqrtg"])),
                max_abs_jacobian=float(np.max(jacobian)))


def read_vmecpp(path):
    """Fold signed n, then undo wout's lambda factor using stored flux profiles."""
    import h5py

    with h5py.File(path) as file:
        wout, internal = file["wout"], file["vmec_internal_results"]
        ns, mpol, ntor, nfp = (int(wout[key][()]) for key in ("ns", "mpol", "ntor", "nfp"))
        xm, xn = wout["xm"][()], wout["xn"][()]
        mode_index = {(int(m), int(round(n / nfp))): i for i, (m, n) in enumerate(zip(xm, xn))}
        spectra = [np.asarray(wout[name][()]) for name in ("rmnc", "zmns", "lmns_full")]
        if any(a.shape != (len(xm), ns) for a in spectra):
            raise ValueError(f"{path}: unexpected VMEC++ spectral layout")
        state = np.zeros((6, mpol * (ntor + 1), ns))
        for m in range(mpol):
            for n in range(ntor + 1):
                mode = m * (ntor + 1) + n
                rp, zp, lp = (a[mode_index[m, n]] for a in spectra)
                if m == 0:
                    state[0, mode], state[4, mode], state[5, mode] = rp, -zp, -lp
                else:
                    rn, zn, ln = (a[mode_index[m, -n]] if n else 0.0 for a in spectra)
                    state[0, mode], state[1, mode], state[2, mode] = rp + rn, zp + zn, lp + ln
                    if n:
                        state[3, mode], state[4, mode], state[5, mode] = rp - rn, -zp + zn, -lp + ln
        phip_f, phip_h = internal["phipF"][()], internal["phipH"][()]
        if phip_f.shape != (ns,) or phip_h.shape != (ns - 1,):
            raise ValueError(f"{path}: unexpected VMEC++ flux layout")
        # Sequential binary64 sum matches the defining lambda normalization;
        # do not assume phipF is constant or replace it by phiedge/(2*pi).
        sum_squared = 0.0
        for value in phip_h:
            sum_squared += float(value) * float(value)
        lamscale = math.sqrt(sum_squared / (ns - 1))
        if not math.isfinite(lamscale) or lamscale <= 0:
            raise ValueError(f"{path}: invalid lambda normalization")
        factor = phip_f / lamscale
        state[[2, 5]] *= factor[None, None, :]
        internal_checks = {}
        for name, component in (("lmnsc", 2), ("lmncs", 5)):
            values = internal[name][()]
            if not values.size:
                continue
            values = values.reshape(ns, ntor + 1, mpol).transpose(2, 1, 0).copy()
            for m in range(mpol):
                for n in range(ntor + 1):
                    values[m, n] *= (math.sqrt(2) if m else 1) * (math.sqrt(2) if n else 1)
            internal_checks[name] = metrics(state[component, :, 1:],
                values.reshape(mpol * (ntor + 1), ns)[:, 1:])
            # This is a format-conversion check, not a solver-agreement gate.
            tolerance = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
            internal_checks[name]["roundoff_check_passed"] = bool(
                internal_checks[name]["finite"] and internal_checks[name]["max_abs"] <= tolerance)
            if not internal_checks[name]["roundoff_check_passed"]:
                raise ValueError(f"{path}: wout/internal {name} normalization disagrees")
        metadata = dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        ns=ns, mpol=mpol, ntor=ntor, nfp=nfp, lamscale=lamscale,
                        phipF_range=[float(phip_f.min()), float(phip_f.max())],
                        lambda_conversion="folded(lmns_full) * phipF / lamscale",
                        internal_lambda_checks=internal_checks)
        indata = file["indata"]
        physics = {}
        for key in ("ncurr", "nfp", "mpol", "ntor", "lfreeb", "phiedge", "pres_scale",
                    "gamma", "spres_ped", "bloat", "curtor", "tcon0", "pmass_type",
                    "piota_type", "pcurr_type", "am", "ac", "ai", "aphi"):
            value = np.asarray(indata[key][()]).tolist()
            physics["adiabatic_index" if key == "gamma" else key] = (
                value.decode() if isinstance(value, bytes) else value)
        if ntor == 0:
            physics["rbcc"] = np.asarray(indata["rbc"][()]).reshape(mpol).tolist()
            physics["zbsc"] = np.asarray(indata["zbs"][()]).reshape(mpol).tolist()
        metadata["input_physics"] = physics
        for name in ("ier_flag", "fsqr", "fsqz", "fsql"):
            if name in wout:
                metadata[name] = np.asarray(wout[name][()]).item()
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{path}: nonfinite VMEC++ state")
    return state, metadata


def validate_pair(input_path, baseline_path, candidate_path, *, baseline_checkpoint=None,
                  candidate_checkpoint=None, baseline_replay=None, candidate_replay=None,
                  vmecpp_path=None):
    baseline, candidate = read_container(baseline_path), read_container(candidate_path)
    result = dict(schema="cumes-axisymmetric-newton-validation-v1",
        baseline=output_checks(input_path, baseline, baseline_checkpoint, baseline_replay),
        candidate=output_checks(input_path, candidate, candidate_checkpoint, candidate_replay))
    result["same_input_and_shape"] = (baseline["params"] == candidate["params"] and
        baseline["state"].shape == candidate["state"].shape)
    result["passed"] = (result["baseline"]["passed"] and result["candidate"]["passed"]
                         and result["same_input_and_shape"])
    if result["same_input_and_shape"]:
        result["coefficient_differences_axis_excluded"] = compare_states(candidate["state"], baseline["state"])
        if baseline["params"]["ntor"] == 0:
            result["surface_differences"] = compare_surfaces(candidate["state"], baseline["state"])
        result["axis_r00_difference"] = float(candidate["state"][0, 0, 0] - baseline["state"][0, 0, 0])
        result["field_differences_native_coordinates"] = {
            name: metrics(candidate[group][name], values)
            for group in ("half_fields", "full_fields") for name, values in baseline[group].items()
            if name in candidate[group]}
        if (result["baseline"]["checks"]["scientific_fields_complete_finite"] and
                result["candidate"]["checks"]["scientific_fields_complete_finite"]):
            a, b = physical_diagnostics(candidate), physical_diagnostics(baseline)
            result["physical_diagnostics"] = dict(
                volume_baseline=b["volume"], volume_candidate=a["volume"],
                relative_volume_change=(a["volume"] - b["volume"]) / b["volume"],
                baseline_geometry={key: b[key] for key in
                    ("min_b_squared", "min_oriented_jacobian", "max_abs_jacobian")},
                candidate_geometry={key: a[key] for key in
                    ("min_b_squared", "min_oriented_jacobian", "max_abs_jacobian")},
                b_squared_native=metrics(a["b_squared"], b["b_squared"]),
                flux_surface_mean_b_squared=metrics(a["flux_surface_mean_b_squared"], b["flux_surface_mean_b_squared"]))
    if vmecpp_path is not None:
        vmec_state, metadata = read_vmecpp(vmecpp_path)
        if any(value != baseline["params"].get(key)
               for key, value in metadata["input_physics"].items()):
            raise ValueError(f"{vmecpp_path}: VMEC++ and cuMES physical inputs differ")
        result["vmecpp_diagnostic"] = dict(metadata=metadata,
            baseline=compare_states(baseline["state"], vmec_state),
            candidate=compare_states(candidate["state"], vmec_state),
            convergence_oracle=False)
        if metadata["ntor"] == 0:
            result["vmecpp_diagnostic"]["baseline_surfaces"] = compare_surfaces(baseline["state"], vmec_state)
            result["vmecpp_diagnostic"]["candidate_surfaces"] = compare_surfaces(candidate["state"], vmec_state)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-checkpoint", "candidate-checkpoint", "baseline-replay", "candidate-replay", "vmecpp"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        result = validate_pair(args.input, args.baseline, args.candidate,
            baseline_checkpoint=args.baseline_checkpoint, candidate_checkpoint=args.candidate_checkpoint,
            baseline_replay=args.baseline_replay, candidate_replay=args.candidate_replay, vmecpp_path=args.vmecpp)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(passed=result["passed"], output=str(args.output))))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(2, f"validation: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
