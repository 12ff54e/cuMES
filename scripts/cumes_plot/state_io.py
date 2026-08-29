"""Readers for native cuMES state containers."""

import os
import struct

import numpy as np


FAM_NAMES = ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")
NO_PARAMS_ERROR = ("container predates the embedded-input record; "
                   "re-run the solver to regenerate it")


def _read_str(f):
    n = struct.unpack("<i", f.read(4))[0]
    if not 0 <= n < 1 << 24:
        raise SystemExit("error: corrupt string length in container")
    return f.read(n).decode()


def _read_vec(f, fmt, cap=1 << 20):
    n = struct.unpack("<i", f.read(4))[0]
    if not 0 <= n <= cap:
        raise SystemExit("error: corrupt vector count in container")
    if n == 0:
        return []
    return struct.unpack(f"<{n}{fmt}", f.read(n * struct.calcsize(fmt)))


def _read_input_record(f, has_profile_types=False,
                       has_free_boundary_extension=False,
                       has_inline_makegrid_extension=False,
                       has_embedded_makegrid_extension=False):
    """The fixed-order embedded-input record (io_common.hpp write/readInput
    Params): 6 i32 + 8 f64 scalars, schema string, six f64 vectors, the
    input stages, the raw boundary, four folded vectors (skipped), and the
    optional free-boundary extension. The combined binary-v8/checkpoint-v6
    layout also carries profile types, inline-Makegrid paths, and optional
    embedded parameters."""
    mpol, ntor, nfp, ntheta, nzeta, ncurr = struct.unpack("<6i", f.read(24))
    (delt, phiedge, pres_scale, adiabatic_index, spres_ped, bloat, curtor,
     tcon0) = struct.unpack("<8d", f.read(64))
    schema = _read_str(f)
    pmass_type = "power_series"
    piota_type = "power_series"
    pcurr_type = "power_series"
    if has_profile_types:
        pmass_type = _read_str(f)
        piota_type = _read_str(f)
        pcurr_type = _read_str(f)
    am = list(_read_vec(f, "d"))
    ac = list(_read_vec(f, "d"))
    ai = list(_read_vec(f, "d"))
    aphi = list(_read_vec(f, "d"))
    raxis_c = list(_read_vec(f, "d"))
    zaxis_s = list(_read_vec(f, "d"))
    nstages = struct.unpack("<i", f.read(4))[0]
    if not 0 <= nstages <= 1 << 20:
        raise SystemExit("error: corrupt stage count in container")
    stages = []
    for _ in range(nstages):
        sn, smi, sft = struct.unpack("<iid", f.read(16))
        stages.append({"ns": sn, "max_iter": smi, "ftol": sft})
    rbc_m = list(_read_vec(f, "i"))
    rbc_n = list(_read_vec(f, "i"))
    rbc_v = list(_read_vec(f, "d"))
    zbs_m = list(_read_vec(f, "i"))
    zbs_n = list(_read_vec(f, "i"))
    zbs_v = list(_read_vec(f, "d"))
    if not (len(rbc_m) == len(rbc_n) == len(rbc_v) and
            len(zbs_m) == len(zbs_n) == len(zbs_v)):
        raise SystemExit("error: corrupt boundary vectors in container")
    for _ in range(4):  # folded rbcc/rbss/zbsc/zbcs
        _read_vec(f, "d")
    lfreeb, nvacskip, mgrid_file, extcur = False, 1, "", []
    coils_file, makegrid_parameters_file = "", ""
    if has_free_boundary_extension:
        lfreeb_i, nvacskip = struct.unpack("<2i", f.read(8))
        lfreeb = bool(lfreeb_i)
        mgrid_file = _read_str(f)
        extcur = list(_read_vec(f, "d"))
    if has_inline_makegrid_extension:
        coils_file = _read_str(f)
        makegrid_parameters_file = _read_str(f)
    makegrid_parameters = None
    if has_embedded_makegrid_extension:
        present = struct.unpack("<i", f.read(4))[0]
        if present:
            normalize, symmetry, nfp_mg = struct.unpack("<3i", f.read(12))
            r_min, r_max = struct.unpack("<2d", f.read(16))
            nr = struct.unpack("<i", f.read(4))[0]
            z_min, z_max = struct.unpack("<2d", f.read(16))
            nz, nphi = struct.unpack("<2i", f.read(8))
            makegrid_parameters = {
                "normalize_by_currents": bool(normalize),
                "assume_stellarator_symmetry": bool(symmetry),
                "number_of_field_periods": nfp_mg,
                "r_grid_minimum": r_min, "r_grid_maximum": r_max,
                "number_of_r_grid_points": nr,
                "z_grid_minimum": z_min, "z_grid_maximum": z_max,
                "number_of_z_grid_points": nz,
                "number_of_phi_grid_points": nphi,
            }

    return {
        "schema": schema, "mpol": mpol, "ntor": ntor, "nfp": nfp,
        "pmass_type": pmass_type, "piota_type": piota_type,
        "pcurr_type": pcurr_type,
        "ntheta": ntheta, "nzeta": nzeta, "ncurr": ncurr, "delt": delt,
        "phiedge": phiedge, "pres_scale": pres_scale,
        "adiabatic_index": adiabatic_index, "spres_ped": spres_ped,
        "bloat": bloat, "curtor": curtor, "tcon0": tcon0,
        "am": am, "ac": ac, "ai": ai, "aphi": aphi,
        "raxis_c": raxis_c, "zaxis_s": zaxis_s,
        "stages": stages,
        "rbc": list(zip(rbc_m, rbc_n, rbc_v)),
        "zbs": list(zip(zbs_m, zbs_n, zbs_v)),
        "lfreeb": lfreeb, "nvacskip": nvacskip,
        "mgrid_file": mgrid_file, "coils_file": coils_file,
        "makegrid_parameters_file": makegrid_parameters_file,
        "makegrid_parameters": makegrid_parameters,
        "extcur": extcur,
    }


def _no_params(path):
    raise SystemExit(f"error: {path}: " + NO_PARAMS_ERROR)


def load_state(path):
    """Load the converged state + the embedded structured input record from
    any solver output container (docs/output-formats.md): versioned binary
    (v8), checkpoint (v6), NetCDF, or HDF5. Returns (ns, mnmax, fams,
    params, name) — the six mode-major families (index = mode * ns +
    surface), the input record as a dict mirroring InputParams, and a
    display name (the recorded input path stem when available, else the
    container stem). Containers without the record are rejected; there is
    no input-JSON fallback."""
    with open(path, "rb") as f:
        head = f.read(16)
    name = os.path.splitext(os.path.basename(path))[0]
    if head.startswith(b"CUMES001"):
        # Versioned binary: magic(8), version(4), ns(4), mnmax(4), the six
        # families, then the provenance trailer (run outcome + provenance
        # strings + stage records), then the v3 input record. Version 8 adds a
        # real-space derived-field block between the families and trailer.
        with open(path, "rb") as f:
            f.seek(8)
            version = struct.unpack("<i", f.read(4))[0]
            if not 1 <= version <= 8:
                raise SystemExit(f"error: unsupported container version "
                                 f"{version} in {path}")
            ns, mnmax = struct.unpack("<ii", f.read(8))
            n = ns * mnmax
            fams = {fam: np.frombuffer(f.read(8 * n), dtype="<f8")
                    for fam in FAM_NAMES}
            if version >= 8:
                ntheta_fields, nzeta_fields = struct.unpack("<2i", f.read(8))
                if ntheta_fields < 0 or nzeta_fields < 0:
                    raise SystemExit("error: corrupt derived-field dimensions "
                                     f"in {path}")
                if bool(ntheta_fields) != bool(nzeta_fields):
                    raise SystemExit("error: incomplete derived-field dimensions "
                                     f"in {path}")
                if ntheta_fields:
                    points = ntheta_fields * nzeta_fields
                    field_values = (7 * (ns - 1) + 6 * ns) * points
                    if len(f.read(8 * field_values)) != 8 * field_values:
                        raise SystemExit("error: truncated derived-field block "
                                         f"in {path}")
            if version < 3:
                _no_params(path)
            nstages = struct.unpack("<4i", f.read(16))[3]
            # precision, status, total_iter, nstages (the header count is
            # the stage-record count; no second count follows the strings)
            if not 0 <= nstages <= 1 << 20:
                raise SystemExit("error: corrupt stage count in container")
            _read_str(f)  # revision
            f.read(1)     # dirty
            _read_str(f)  # build_type
            if version >= 2:
                _read_str(f)  # precision_policy
                _read_str(f)  # compile_flags
            else:
                _read_str(f)  # historical scalar_type slot
            source_path = _read_str(f)
            _read_str(f)  # source_hash
            for _ in range(4):  # gpu_name, driver, runtime, toolkit
                _read_str(f)
            for _ in range(nstages):
                f.read(8)   # ns, iterations
                f.read(1)   # converged
                f.read(24)  # fsqr, fsqz, fsql
                nrst = struct.unpack("<i", f.read(4))[0]
                f.read(4 * nrst)
            params = _read_input_record(
                f, version == 4 or version >= 7,
                version >= 5, version >= 5, version >= 6)
            params["_source_path"] = source_path
        if source_path:
            name = os.path.splitext(os.path.basename(source_path))[0]
        return ns, mnmax, fams, params, name
    if head.startswith(b"CUMECKP1"):
        # Checkpoint: magic(8), version field(4), precision(4), ns(4), mnmax(4),
        # the six families, then the v2 input record; v3 appends the
        # free-boundary fields, v4 the inline-Makegrid paths, and v5 the
        # optional embedded Makegrid parameters.
        with open(path, "rb") as f:
            f.seek(8)
            version = struct.unpack("<i", f.read(4))[0]
            if not 1 <= version <= 6:
                raise SystemExit(f"error: unsupported checkpoint version "
                                 f"{version} in {path}")
            f.read(4)  # precision (always double)
            ns, mnmax = struct.unpack("<ii", f.read(8))
            n = ns * mnmax
            fams = {fam: np.frombuffer(f.read(8 * n), dtype="<f8")
                    for fam in FAM_NAMES}
            if version < 2:
                _no_params(path)
            params = _read_input_record(
                f, version == 3 or version >= 6,
                version >= 4, version >= 4, version >= 5)
        return ns, mnmax, fams, params, name
    if head.startswith(b"CDF"):
        # NetCDF: the six families are 2-D [surface, mode] datasets; the
        # input record is the native scalar variables + 1-D arrays.
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as nc:
            ns = nc.dimensions["ns"]
            mnmax = nc.dimensions["mnmax"]
            fams = {fam: np.asarray(nc.variables[fam][:],
                                    dtype="<f8").T.ravel()
                    for fam in FAM_NAMES}
            if "mpol" not in nc.variables:
                _no_params(path)

            def scalar(var):
                v = nc.variables[var]
                return v.getValue() if hasattr(v, "getValue") else \
                    np.asarray(v[:]).item()

            def sattr(attr, default):
                v = getattr(nc, attr, default)
                return v.decode() if isinstance(v, bytes) else v

            def darray(var):
                return list(np.asarray(nc.variables[var][:], dtype="<f8")) \
                    if var in nc.variables else []

            def iarray(var):
                return list(np.asarray(nc.variables[var][:], dtype="<i4")) \
                    if var in nc.variables else []

            makegrid_parameters = None
            if "makegrid_parameters_present" in nc.variables and \
                    scalar("makegrid_parameters_present"):
                makegrid_parameters = {
                    "normalize_by_currents":
                        bool(scalar("makegrid_normalize_by_currents")),
                    "assume_stellarator_symmetry": bool(scalar(
                        "makegrid_assume_stellarator_symmetry")),
                    "number_of_field_periods": int(scalar(
                        "makegrid_number_of_field_periods")),
                    "r_grid_minimum": float(scalar(
                        "makegrid_r_grid_minimum")),
                    "r_grid_maximum": float(scalar(
                        "makegrid_r_grid_maximum")),
                    "number_of_r_grid_points": int(scalar(
                        "makegrid_number_of_r_grid_points")),
                    "z_grid_minimum": float(scalar(
                        "makegrid_z_grid_minimum")),
                    "z_grid_maximum": float(scalar(
                        "makegrid_z_grid_maximum")),
                    "number_of_z_grid_points": int(scalar(
                        "makegrid_number_of_z_grid_points")),
                    "number_of_phi_grid_points": int(scalar(
                        "makegrid_number_of_phi_grid_points")),
                }

            params = {
                "schema": sattr("schema", "cumes-config-v1"),
                "mpol": scalar("mpol"), "ntor": scalar("ntor"),
                "nfp": scalar("nfp"), "ntheta": scalar("ntheta"),
                "nzeta": scalar("nzeta"), "ncurr": scalar("ncurr"),
                "delt": scalar("delt"), "phiedge": scalar("phiedge"),
                "pres_scale": scalar("pres_scale"),
                "adiabatic_index": scalar("adiabatic_index"),
                "spres_ped": scalar("spres_ped"), "bloat": scalar("bloat"),
                "curtor": scalar("curtor"), "tcon0": scalar("tcon0"),
                "am": darray("am"), "ac": darray("ac"), "ai": darray("ai"),
                "aphi": darray("aphi"), "raxis_c": darray("raxis_c"),
                "zaxis_s": darray("zaxis_s"),
                "stages": [{"ns": a, "max_iter": b, "ftol": c}
                           for a, b, c in zip(iarray("stage_in_ns"),
                                              iarray("stage_max_iter"),
                                              darray("stage_ftol"))],
                "rbc": list(zip(iarray("rbc_m"), iarray("rbc_n"),
                                darray("rbc_value"))),
                "zbs": list(zip(iarray("zbs_m"), iarray("zbs_n"),
                                darray("zbs_value"))),
                "lfreeb": bool(scalar("lfreeb"))
                    if "lfreeb" in nc.variables else False,
                "nvacskip": int(scalar("nvacskip"))
                    if "nvacskip" in nc.variables else 1,
                "mgrid_file": sattr("mgrid_file", ""),
                "coils_file": sattr("coils_file", ""),
                "makegrid_parameters_file":
                    sattr("makegrid_parameters_file", ""),
                "makegrid_parameters": makegrid_parameters,
                "extcur": darray("extcur"),
            }
            sp = getattr(nc, "source_path", "")
            if isinstance(sp, bytes):
                sp = sp.decode()
            params["_source_path"] = sp
            if sp:
                name = os.path.splitext(os.path.basename(sp))[0]
        return ns, mnmax, fams, params, name
    if head.startswith(b"\x89HDF"):
        # HDF5: same [surface, mode] dataset layout; the input record is
        # native scalar attributes + 1-D datasets.
        import h5py
        with h5py.File(path, "r") as f5:
            fams = {}
            ns = mnmax = None
            for fam in FAM_NAMES:
                dset = np.asarray(f5[fam][:], dtype="<f8")  # [surface, mode]
                ns, mnmax = dset.shape
                fams[fam] = dset.T.ravel()
            if "mpol" not in f5.attrs:
                _no_params(path)

            def sattr(attr, default):
                v = f5.attrs.get(attr, default)
                return v.decode() if isinstance(v, bytes) else v

            def darray(var):
                return list(np.asarray(f5[var][:], dtype="<f8")) \
                    if var in f5 else []

            def iarray(var):
                return list(np.asarray(f5[var][:], dtype="<i4")) \
                    if var in f5 else []

            makegrid_parameters = None
            if int(f5.attrs.get("makegrid_parameters_present", 0)):
                makegrid_parameters = {
                    "normalize_by_currents": bool(int(f5.attrs[
                        "makegrid_normalize_by_currents"])),
                    "assume_stellarator_symmetry": bool(int(f5.attrs[
                        "makegrid_assume_stellarator_symmetry"])),
                    "number_of_field_periods": int(f5.attrs[
                        "makegrid_number_of_field_periods"]),
                    "r_grid_minimum": float(f5.attrs[
                        "makegrid_r_grid_minimum"]),
                    "r_grid_maximum": float(f5.attrs[
                        "makegrid_r_grid_maximum"]),
                    "number_of_r_grid_points": int(f5.attrs[
                        "makegrid_number_of_r_grid_points"]),
                    "z_grid_minimum": float(f5.attrs[
                        "makegrid_z_grid_minimum"]),
                    "z_grid_maximum": float(f5.attrs[
                        "makegrid_z_grid_maximum"]),
                    "number_of_z_grid_points": int(f5.attrs[
                        "makegrid_number_of_z_grid_points"]),
                    "number_of_phi_grid_points": int(f5.attrs[
                        "makegrid_number_of_phi_grid_points"]),
                }

            params = {
                "schema": sattr("schema", "cumes-config-v1"),
                "mpol": int(f5.attrs["mpol"]),
                "ntor": int(f5.attrs["ntor"]),
                "nfp": int(f5.attrs["nfp"]),
                "ntheta": int(f5.attrs["ntheta"]),
                "nzeta": int(f5.attrs["nzeta"]),
                "ncurr": int(f5.attrs["ncurr"]),
                "delt": float(f5.attrs["delt"]),
                "phiedge": float(f5.attrs["phiedge"]),
                "pres_scale": float(f5.attrs["pres_scale"]),
                "adiabatic_index": float(f5.attrs["adiabatic_index"]),
                "spres_ped": float(f5.attrs["spres_ped"]),
                "bloat": float(f5.attrs["bloat"]),
                "curtor": float(f5.attrs["curtor"]),
                "tcon0": float(f5.attrs["tcon0"]),
                "am": darray("am"), "ac": darray("ac"), "ai": darray("ai"),
                "aphi": darray("aphi"), "raxis_c": darray("raxis_c"),
                "zaxis_s": darray("zaxis_s"),
                "stages": [{"ns": a, "max_iter": b, "ftol": c}
                           for a, b, c in zip(iarray("stage_in_ns"),
                                              iarray("stage_max_iter"),
                                              darray("stage_ftol"))],
                "rbc": list(zip(iarray("rbc_m"), iarray("rbc_n"),
                                darray("rbc_value"))),
                "zbs": list(zip(iarray("zbs_m"), iarray("zbs_n"),
                                darray("zbs_value"))),
                "lfreeb": bool(int(f5.attrs.get("lfreeb", 0))),
                "nvacskip": int(f5.attrs.get("nvacskip", 1)),
                "mgrid_file": sattr("mgrid_file", ""),
                "coils_file": sattr("coils_file", ""),
                "makegrid_parameters_file":
                    sattr("makegrid_parameters_file", ""),
                "makegrid_parameters": makegrid_parameters,
                "extcur": darray("extcur"),
            }
            params["_source_path"] = sattr("source_path", "")
            if params["_source_path"]:
                name = os.path.splitext(
                    os.path.basename(params["_source_path"]))[0]
        return ns, mnmax, fams, params, name
    raise SystemExit(f"error: {path} is not a cumes state container "
                     f"(expected CUMES001/CUMECKP1/NetCDF/HDF5)")
