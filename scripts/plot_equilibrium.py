#!/usr/bin/env python3
"""Render 3D figures of a converged cuMES equilibrium.

Reads any solver state container (docs/output-formats.md): versioned
binary (v8), checkpoint (v6), NetCDF, or HDF5. Every container embeds the
structured normalized-input record, so the script needs no input JSON:
mpol/ntor/nfp, the resolved angular grid, phiedge, the am/ac/ai/aphi
profiles, and the raw initial boundary all come from the container.
Containers predating the embedded record are rejected.

Reconstructs real-space geometry with the solver's exact conventions
(parity-split e/o arrays, odd-m scalxc regularization, staggered half-grid
metric — src/kernels/geometry_impl.cuh) via batched 2-D FFT synthesis, evaluates
the radial profiles exactly as kernels/profiles_impl.cuh does (ncurr=0: chi' from
the prescribed iota; ncurr=1: the ncurr1FinalizeKernel constraint solve
with curr evaluated at the flux coordinate), and renders the full
nfp-period torus colored by |B| with light-source shading.

One PNG file per view: <out>_perspective.png, <out>_top.png,
<out>_combined.png (both views side by side), and <out>_slices.png (the
top view with six RZ poloidal cross-sections in two rows of three,
spanning one field period, each showing nested flux-surface contours).
The 3-D figures plot a single flux surface (the plasma boundary). The
magnetic axis is the converged axis extracted from the state (m=0 content
of the innermost surface, seed_state.hpp convention: nfp-fold
modulation).

Two self-checks run before rendering:
  1. for fixed boundary, the state LCFS (j = ns-1) must match the embedded
     rbc/zbs boundary; for free boundary, its displacement is reported;
  2. the edge |B| range is printed for a physical plausibility review.

For free-boundary states, ``--coils [PATH]`` adds the MAKEGRID filament
geometry to every 3-D view as bronze tubes.  PATH can be omitted when the
state embeds ``coils_file``; states made from a precomputed MGRID table need
an explicit coils-dot or ``cumes-coils-v1`` JSON path.  ``--coil-radius
METERS`` overrides the small data-scaled default tube radius.

Usage: plot_equilibrium.py [--state PATH] [--out PATH.png] [--field-lines]
                           [--coils [PATH]] [--coil-radius METERS]
"""

import argparse
import json
import multiprocessing
import os
import struct
import time

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image as PILImage
from scipy.interpolate import RegularGridInterpolator

SIGN_J = -1.0  # DeviceParams::kSignJacobian (device_params.hpp): the
               # solver's fixed coordinate-sign convention, not an input
MU0 = 4.0e-7 * np.pi
BRONZE = "#CD7F32"

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


def mode_tables(mnmax, ntor):
    m = np.arange(mnmax) // (ntor + 1)
    n = np.arange(mnmax) % (ntor + 1)
    return m, n


def pack_series(a, b, series, mm, nn, nth, nzt):
    """Pack one family-parity's modes into a complex 2-D spectrum C[p, q]
    over the full-period grid (theta_j = 2*pi*j/nth, zeta_k = 2*pi*k/nzt).
    series='R': a*cos(mθ)cos(nζ) + b*sin(mθ)sin(nζ) decomposes as
        1/4[(a-b)(e^{+i(mθ+nζ)} + e^{-i(mθ+nζ)}) + (a+b)(e^{+i(mθ-nζ)} + e^{-i(mθ-nζ)})]
    series='Z': a*sin(mθ)cos(nζ) + b*cos(mθ)sin(nζ) decomposes as
        1/4[(a+b)(-i e^{+i(mθ+nζ)} + i e^{-i(mθ+nζ)}) + (a-b)(-i e^{+i(mθ-nζ)} + i e^{-i(mθ-nζ)})]
    (λ shares the Z form). At m=0/n=0 the ±mirror entries collide, the sin
    terms cancel and the cos terms double — exactly the right result — so
    no special cases are needed."""
    C = np.zeros((nth, nzt), dtype=np.complex128)
    rows = np.concatenate([mm, -mm, mm, -mm]) % nth
    cols = np.concatenate([nn, -nn, -nn, nn]) % nzt
    if series == "R":
        vals = 0.25 * np.concatenate([a - b, a - b, a + b, a + b])
    else:
        vals = 0.25j * np.concatenate([-(a + b), a + b, -(a - b), a - b])
    np.add.at(C, (rows, cols), vals)
    return C


def eval_state(fams, ns, j, th, zt, ntor, nfp):
    """Stored parity arrays at full-grid surface j on a uniform full-period
    (th, zt) grid — the exact inverseDFT convention (kernels/fourier_impl.cuh),
    synthesized with a batched 2-D IFFT instead of a direct mode sum:

      R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
      Z = zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
      λ = lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ)

    even m -> e arrays (plain), odd m -> o arrays divided by
    maxsc = max(sqrt(s_j), sqrt(ds)) (vmecpp's scalxc decomposition).
    Derivatives are spectral differentiation of the packed spectrum:
    ×i·m for ∂/∂θ, ×i·n·nfp for ∂/∂ζ (the solver's toroidal derivative
    tables carry the nfp factor). The λ ζ-derivative slots store -dλ/dζ
    (signV = -1).
    """
    m, n = mode_tables(fams["rmncc"].shape[0] // ns, ntor)
    nth, nzt = th.size, zt.size
    assert th.ndim == 1 and zt.ndim == 1, "eval_state takes 1-D angle axes"
    assert np.allclose(th, 2.0 * np.pi * np.arange(nth) / nth) and \
        np.allclose(zt, 2.0 * np.pi * np.arange(nzt) / nzt), \
        "eval_state needs uniform full-period grids"
    par_odd = (m % 2 == 1)
    maxsc = max(np.sqrt(j / (ns - 1)), np.sqrt(1.0 / (ns - 1)))
    fac = np.where(par_odd, 1.0 / maxsc, 1.0)

    def fam(name):
        mode = m * (ntor + 1) + n     # folded mode index
        return fams[name][mode * ns + j]  # mode-major, surface j column

    rc, rs = fam("rmncc") * fac, fam("rmnss") * fac
    zs, zc = fam("zmnsc") * fac, fam("zmncs") * fac
    lsc, lcs = fam("lmnsc") * fac, fam("lmncs") * fac
    ev, od = ~par_odd, par_odd
    C = np.zeros((6, nth, nzt), dtype=np.complex128)
    C[0] = pack_series(rc[ev], rs[ev], "R", m[ev], n[ev], nth, nzt)
    C[1] = pack_series(rc[od], rs[od], "R", m[od], n[od], nth, nzt)
    C[2] = pack_series(zs[ev], zc[ev], "Z", m[ev], n[ev], nth, nzt)
    C[3] = pack_series(zs[od], zc[od], "Z", m[od], n[od], nth, nzt)
    C[4] = pack_series(lsc[ev], lcs[ev], "Z", m[ev], n[ev], nth, nzt)
    C[5] = pack_series(lsc[od], lcs[od], "Z", m[od], n[od], nth, nzt)
    # derivative spectra: ×i·m for ∂/∂θ, ×i·n·nfp for ∂/∂ζ; the λ
    # ζ-derivative slots (indices 16, 17) are negated (signV = -1).
    dth = 1j * (np.fft.fftfreq(nth) * nth)[:, None]
    dzt = 1j * (np.fft.fftfreq(nzt) * nzt * nfp)[None, :]
    S = np.concatenate([C, C * dth, C * dzt], axis=0)   # (18, nth, nzt)
    S[16:] *= -1.0
    G = np.fft.ifft2(S).real * (nth * nzt)
    keys = ("re", "ro", "ze", "zo", "le", "lo",
            "rue", "ruo", "zue", "zuo", "lue", "luo",
            "rve", "rvo", "zve", "zvo", "lve", "lvo")
    out = {"th": th, "zt": zt}
    for k, g in zip(keys, G):
        out[k] = g
    return out


def half_grid(arrays, ns, jh, chip, phip_avg, lamscale):
    """Mirror of baseGeometryKernel + magneticFieldKernel for half-grid
    surface jh (kernels/geometry_impl.cuh). arrays = the two adjacent full-grid
    surfaces (stored parity values); phip_avg = 0.5·(phipF(jh)+phipF(jh+1))
    and lamscale come from make_profiles. Returns the half-grid geometry
    (r12/z12), the covariant metric, lambda derivatives, and |B| (Tesla —
    the B^u·B_u products are invariant under the angular coordinate
    scaling, so the kernel units are physical)."""
    ds = 1.0 / (ns - 1)
    s_in = np.sqrt(jh / (ns - 1))
    s_out = np.sqrt((jh + 1) / (ns - 1))
    s_h = np.sqrt((jh + 0.5) / (ns - 1))
    r, rp = arrays[0], arrays[1]

    # ---- base geometry (stored parity values, exactly as the kernel) ----
    r12 = 0.5 * ((r["re"] + rp["re"]) + s_h * (r["ro"] + rp["ro"]))
    z12 = 0.5 * ((r["ze"] + rp["ze"]) + s_h * (r["zo"] + rp["zo"]))
    ru12 = 0.5 * ((r["rue"] + rp["rue"]) + s_h * (r["ruo"] + rp["ruo"]))
    zu12 = 0.5 * ((r["zue"] + rp["zue"]) + s_h * (r["zuo"] + rp["zuo"]))
    rs_ = ((rp["re"] - r["re"]) + s_h * (rp["ro"] - r["ro"])) / ds
    zs_ = ((rp["ze"] - r["ze"]) + s_h * (rp["zo"] - r["zo"])) / ds

    tau1 = ru12 * zs_ - rs_ * zu12
    tau2 = (rp["ruo"] * rp["zo"] + r["ruo"] * r["zo"]
            - rp["zuo"] * rp["ro"] - r["zuo"] * r["ro"]
            + (rp["rue"] * rp["zo"] + r["rue"] * r["zo"]
               - rp["zue"] * rp["ro"] - r["zue"] * r["ro"]) / s_h)
    tau = tau1 + 0.25 * tau2
    gsqrt = tau * r12

    sfi2, sfo2 = s_in * s_in, s_out * s_out
    guu = 0.5 * ((r["rue"] ** 2 + r["zue"] ** 2) + (rp["rue"] ** 2 + rp["zue"] ** 2)
                 + sfi2 * (r["ruo"] ** 2 + r["zuo"] ** 2)
                 + sfo2 * (rp["ruo"] ** 2 + rp["zuo"] ** 2)) \
        + s_h * ((r["rue"] * r["ruo"] + r["zue"] * r["zuo"])
                 + (rp["rue"] * rp["ruo"] + rp["zue"] * rp["zuo"]))
    # NOTE: the sH-weighted cross terms sit INSIDE the 0.5, matching vmecpp's
    # ComputeMetricElements exactly (metric_kernel.h) — moving them outside
    # doubles the cross-term weight (caught historically by the iter-1 dump).
    guv = 0.5 * ((r["rue"] * r["rve"] + r["zue"] * r["zve"])
                 + (rp["rue"] * rp["rve"] + rp["zue"] * rp["zve"])
                 + sfi2 * (r["ruo"] * r["rvo"] + r["zuo"] * r["zvo"])
                 + sfo2 * (rp["ruo"] * rp["rvo"] + rp["zuo"] * rp["zvo"])
                 + s_h * ((r["rue"] * rp["rvo"] + r["zue"] * rp["zvo"])
                          + (rp["rue"] * rp["rvo"] + rp["zue"] * rp["zvo"])
                          + (r["rve"] * r["ruo"] + r["zve"] * r["zuo"])
                          + (rp["rve"] * rp["ruo"] + rp["zve"] * rp["zuo"])))
    gvv = 0.5 * (r["re"] ** 2 + rp["re"] ** 2
                 + sfi2 * r["ro"] ** 2 + sfo2 * rp["ro"] ** 2) \
        + s_h * (r["re"] * r["ro"] + rp["re"] * rp["ro"]) \
        + 0.5 * ((r["rve"] ** 2 + r["zve"] ** 2) + (rp["rve"] ** 2 + rp["zve"] ** 2)
                 + sfi2 * (r["rvo"] ** 2 + r["zvo"] ** 2)
                 + sfo2 * (rp["rvo"] ** 2 + rp["zvo"] ** 2)) \
        + s_h * ((r["rve"] * r["rvo"] + r["zve"] * r["zvo"])
                 + (rp["rve"] * rp["rvo"] + rp["zve"] * rp["zvo"]))

    # ---- magnetic field (magneticFieldKernel) ----
    lu_h = 0.5 * ((r["lue"] + rp["lue"]) + s_h * (r["luo"] + rp["luo"]))
    lv_h = 0.5 * ((r["lve"] + rp["lve"]) + s_h * (r["lvo"] + rp["lvo"]))
    inv = np.zeros_like(gsqrt)
    np.divide(1.0, gsqrt, out=inv,
              where=np.isfinite(gsqrt) & (np.abs(gsqrt) > 1e-30))
    bsupv = (lamscale * lu_h + phip_avg) * inv
    bsupu = lamscale * lv_h * inv + chip * inv             # chi' from the solve
    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv
    bsq = bsupu * bsubu + bsupv * bsubv
    return {"r12": r12, "z12": z12, "guu": guu, "guv": guv, "gvv": gvv,
            "gsqrt": gsqrt, "lu_h": lu_h, "lv_h": lv_h, "bsupu": bsupu,
            "bsupv": bsupv, "bmag": np.sqrt(np.maximum(bsq, 0.0))}


def make_profiles(cfg, ns):
    """Radial profile evaluators exactly as profile_functions.hpp +
    kernels/profiles_impl.cuh: maxToroidalFlux = signJ·phiedge/(2π·torflux(1)),
    phipF(j) = maxTF·torfluxDeriv(Δs·j), lamscale = sqrt(Δs·Σ phipH²),
    iota/curr Horner evaluations with the normX = min(|x·bloat|, 1) clamp,
    and the ncurr=1 normalization Itor = signJ·μ0·curtor/(2π·C_edge). The
    aphi=[1.0] case keeps the historic constant-φ' arithmetic bit-stable.
    Returns a dict with phip_avg(jh), lamscale, chip_iota(jh) (ncurr=0) or
    itor/curr_h(jh) (ncurr=1)."""
    aphi = cfg["aphi"]
    ds = 1.0 / (ns - 1)

    def torflux(x):
        ret = 0.0
        for c in reversed(aphi):
            ret = x * ret + c
        return x * ret

    def torflux_deriv(x):
        return sum((i + 1) * c * x ** i for i, c in enumerate(aphi))

    def iota_eval(x):
        ret = 0.0
        for c in reversed(cfg["ai"]):
            ret = x * ret + c
        return ret

    def curr_eval(x):
        norm = min(abs(x * cfg["bloat"]), 1.0)
        ret = 0.0
        for i in range(len(cfg["ac"]) - 1, -1, -1):
            ret = norm * ret + cfg["ac"][i] / (i + 1)
        return norm * ret

    maxTF = SIGN_J * cfg["phiedge"] / (2.0 * np.pi * torflux(1.0))
    if aphi == [1.0]:
        # Constant φ' (torfluxDeriv ≡ 1): the historic arithmetic, kept
        # bit-stable with the pre-generalization formula.
        phip = maxTF
        lamscale = abs(phip) * np.sqrt((ns - 1) * ds)
        prof = {
            "phip_avg": lambda jh: phip,
            "lamscale": lamscale,
        }
    else:
        phip_F = np.array([maxTF * torflux_deriv(ds * j) for j in range(ns)])
        phip_H = 0.5 * (phip_F[:-1] + phip_F[1:])
        lamscale = float(np.sqrt(ds * np.sum(phip_H * phip_H)))
        prof = {
            "phip_avg": lambda jh: 0.5 * (phip_F[jh] + phip_F[jh + 1]),
            "lamscale": lamscale,
        }
    if cfg["ncurr"] == 0:
        def chip_iota(jh):
            sh = ds * (jh + 0.5)
            tf = min(torflux(sh), 1.0)
            return maxTF * iota_eval(tf) * torflux_deriv(sh)
        prof["chip_iota"] = chip_iota
    else:
        c_edge = curr_eval(1.0)
        prof["itor"] = SIGN_J * MU0 * cfg["curtor"] / (2.0 * np.pi * c_edge)

        def curr_h(jh):
            sh = ds * (jh + 0.5)
            return prof["itor"] * curr_eval(min(torflux(sh), 1.0))
        prof["curr_h"] = curr_h
    return prof


def solve_chip(fams, ns, jh, cfg, prof):
    """chi' for half-grid surface jh. ncurr=1: ncurr1FinalizeKernel —
    chi' = (currH − Σ(guu·B^θ_λ + guv·B^ζ)·w) / Σ(guu/√g·w), summed over the
    reduced-theta trapezoid with dnorm3 = 1/(nzeta·(nThetaRed−1)), currH
    evaluated at the FLUX coordinate sh (the solver's convention). ncurr=0:
    the prescribed-iota profile χ' = maxTF·ι(tf)·torfluxDeriv(sh)
    (kernels/profiles_impl.cuh)."""
    if cfg["ncurr"] == 0:
        return prof["chip_iota"](jh)
    ntheta = cfg["ntheta"]
    nz = cfg["nzeta"]
    ntheta_red = ntheta // 2 + 1
    # eval_state needs a uniform full-period grid; the reduced-theta grid is
    # the first nThetaRed rows of the solver's ntheta grid, so evaluate on
    # the full grid and slice.
    th = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zt = 2.0 * np.pi * np.arange(nz) / nz
    a = [eval_state(fams, ns, j, th, zt, cfg["ntor"], cfg["nfp"])
         for j in (jh, jh + 1)]
    a = [{k: v[:ntheta_red] for k, v in e.items() if k not in ("th", "zt")}
         for e in a]
    h = half_grid(a, ns, jh, 0.0, prof["phip_avg"](jh), prof["lamscale"])
    w = np.full(ntheta_red, 1.0 / (nz * (ntheta_red - 1)))
    w[0] *= 0.5
    w[-1] *= 0.5
    jv = np.sum(w[:, None] * (h["guu"] * h["bsupu"] + h["guv"] * h["bsupv"]))
    one_over = np.zeros_like(h["gsqrt"])
    np.divide(1.0, h["gsqrt"], out=one_over, where=np.abs(h["gsqrt"]) > 1e-30)
    avg = np.sum(w[:, None] * h["guu"] * one_over)
    curr = prof["curr_h"](jh)
    return (curr - jv) / avg if avg != 0.0 else 0.0


def boundary_from_params(params, th, zt):
    """Boundary R/Z from the embedded raw harmonics (signed-n VMEC
    convention: cos(mθ − nζ) / sin(mθ − nζ))."""
    R = np.zeros_like(th)
    Z = np.zeros_like(th)
    for m, n, value in params["rbc"]:
        R += value * np.cos(m * th - n * zt)
    for m, n, value in params["zbs"]:
        Z += value * np.sin(m * th - n * zt)
    return R, Z


def periodic_close(A):
    """Close a 2-D periodic grid: append the first row (theta = 2pi) and the
    first column (zeta = 2pi) so the rendered surface mesh has no seam."""
    A = np.concatenate([A, A[:1, :]], axis=0)
    A = np.concatenate([A, A[:, :1]], axis=1)
    return A


def trim_white(path, pad_px=12, title_gap_px=8, center="title"):
    """Post-process the saved PNG: crop to the non-white content, collapse
    the white strip between the title and the figure (3-D projections
    reserve margin for rotation, so bbox_inches='tight' cannot remove it),
    and pad the narrower side so the requested anchor ends up horizontally
    centered. center='title' centers on the topmost ink band (the title);
    center='body' centers on the median x of the body ink below it, which
    is robust against an asymmetric colorbar and centers the torus itself
    (with an axes title, both coincide)."""
    img = PILImage.open(path).convert("RGB")
    a = np.asarray(img)
    nonwhite = ~(a > 250).all(axis=2)
    ys, xs = np.where(nonwhite)
    if len(ys) == 0:
        return
    rows = nonwhite.any(axis=1)
    row_ys = np.where(rows)[0]
    gap0 = np.where(~rows[row_ys[0]:])[0]
    t1 = row_ys[0] + (gap0[0] if len(gap0) else a.shape[0] - row_ys[0])
    band = nonwhite[row_ys[0]:t1]
    band_xs = np.where(band)[1]
    title_cx = 0.5 * (band_xs.min() + band_xs.max()) if len(band_xs) else \
        0.5 * (xs.min() + xs.max())
    below = np.where(rows[t1:])[0]
    body_top = t1 + (below[0] if len(below) else 0)
    # Anchor = the horizontal midpoint of the body ink extent, colorbar
    # included, so the whole figure block (torus + colorbar) is centered
    # and both side margins balance.
    cut = int(0.92 * a.shape[1])  # separates the title text from the colorbar
    body_xs = xs[ys >= body_top]
    anchor = 0.5 * (body_xs.min() + body_xs.max()) \
        if center == "body" and len(body_xs) else title_cx
    # The 3-D box is not centered in the axes window, so the title (placed
    # at the window center by matplotlib) must be shifted onto the torus
    # center in post. The shift moves the title text only; the colorbar
    # label inside the same band stays put.
    title_shift = 0
    tx0 = tx1 = 0
    if center == "body":
        band_mask = band_xs < cut
        if band_mask.any():
            tx0, tx1 = band_xs[band_mask].min(), band_xs[band_mask].max()
            title_shift = int(round(anchor - 0.5 * (tx0 + tx1)))
    # Title piece and body piece; merge when they already touch. The body
    # piece ends at its own ink bottom (the pre-crop image may carry the
    # axes window's white margin below the projection).
    pieces = [(max(row_ys[0] - pad_px, 0), t1 + pad_px),
              (max(body_top - pad_px, 0), min(ys.max() + pad_px, a.shape[0]))]
    merged = []
    for lo, hi in pieces:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    x0 = max(xs.min() - pad_px, 0)
    x1 = min(xs.max() + pad_px, a.shape[1] - 1)
    total_h = sum(hi - lo for lo, hi in merged) + title_gap_px * (len(merged) - 1)
    canvas = PILImage.new("RGB", (x1 + 1 - x0, total_h), (255, 255, 255))
    y_cursor = 0
    for i, (lo, hi) in enumerate(merged):
        piece = img.crop((x0, lo, x1 + 1, hi))
        if i == 0 and title_shift:
            white = PILImage.new("RGB", (tx1 - tx0 + 1, piece.height),
                                 (255, 255, 255))
            piece.paste(white, (tx0 - x0, 0))
            text = img.crop((tx0, lo, tx1 + 1, hi))
            piece.paste(text, (tx0 - x0 + title_shift, 0))
        canvas.paste(piece, (0, y_cursor))
        y_cursor += piece.height + (title_gap_px if i < len(merged) - 1 else 0)
    w, h = canvas.size
    # Anchor center inside the canvas: c = anchor - x0. Padding the canvas
    # with L left / R right (L + c = (w + L + R - 1)/2) gives
    # L = w - 1 - 2c for L >= 0, else R = 2c - (w - 1).
    c = anchor - x0
    shift = int(round((w - 1) - 2.0 * c))
    out = PILImage.new("RGB", (w + abs(shift), h), (255, 255, 255))
    out.paste(canvas, (max(shift, 0), 0))
    out.save(path)


def compose_combined(path, boxes, title_box, cbar_box, pad_px=12,
                     title_gap_px=8, panel_gap_px=64, cbar_gap_px=16):
    """Post-process the combined PNG: crop each panel's axes window (given
    as saved-image boxes) and the colorbar window, tight-trim each to its
    ink, and recompose — panels side by side with a fixed gap, colorbar
    centered underneath — with the title re-centered over the composite.
    Using the known artist boxes avoids any fragile ink detection; the
    3-D boxes do not fill their axes windows, so no layout setting can
    remove the white space alone."""
    img = PILImage.open(path).convert("RGB")

    def tight(piece):
        arr = np.asarray(piece)
        nw = ~(arr > 250).all(axis=2)
        ys, xs = np.where(nw)
        if len(ys) == 0:
            return None
        y0 = max(ys.min() - pad_px, 0)
        y1 = min(ys.max() + pad_px, arr.shape[0] - 1)
        x0 = max(xs.min() - pad_px, 0)
        x1 = min(xs.max() + pad_px, arr.shape[1] - 1)
        return piece.crop((x0, y0, x1 + 1, y1 + 1))

    pieces = []
    for box in boxes:
        p = tight(img.crop((box[0], box[1], box[2] + 1, box[3] + 1)))
        if p is not None:
            pieces.append(p)
    cbar_piece = tight(img.crop((cbar_box[0], cbar_box[1],
                                 cbar_box[2] + 1, cbar_box[3] + 1)))
    title = img.crop((title_box[0], title_box[1],
                      title_box[2] + 1, title_box[3] + 1))
    # Panels row, vertically centered (the two projections have different
    # natural heights; centering balances them).
    body_w = sum(p.width for p in pieces) + panel_gap_px * (len(pieces) - 1)
    body_h = max(p.height for p in pieces)
    body = PILImage.new("RGB", (body_w, body_h), (255, 255, 255))
    x = 0
    for p in pieces:
        body.paste(p, (x, (body_h - p.height) // 2))
        x += p.width + panel_gap_px
    total_w = body_w
    total_h = body_h
    if cbar_piece is not None:
        total_w = max(total_w, cbar_piece.width)
        total_h += cbar_gap_px + cbar_piece.height
    out = PILImage.new("RGB", (total_w, title.height + title_gap_px + total_h),
                       (255, 255, 255))
    out.paste(title, ((total_w - title.width) // 2, 0))
    y = title.height + title_gap_px
    out.paste(body, ((total_w - body_w) // 2, y))
    if cbar_piece is not None:
        y += body_h + cbar_gap_px
        out.paste(cbar_piece, ((total_w - cbar_piece.width) // 2, y))
    final = PILImage.new("RGB", (out.width + 2 * pad_px, out.height + 2 * pad_px),
                         (255, 255, 255))
    final.paste(out, (pad_px, pad_px))
    final.save(path)


def surface_intensity(X, Y, Z, light_dir, lo=0.55):
    """Lambertian illumination from the GEOMETRIC surface normals (cross
    product of the grid tangents), rescaled to [lo, 1] so the side facing
    away from the light stays bright. Unlike LightSource.hillshade over the
    |B| scalar this produces real 3-D relief: the |B| heightfield is so
    smooth that its normal-intensity range collapses and (with fraction<1)
    the whole map clips to black."""
    gx1, gy1, gz1 = np.gradient(X, axis=0), np.gradient(Y, axis=0), np.gradient(Z, axis=0)
    gx2, gy2, gz2 = np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)
    nx = gy1 * gz2 - gz1 * gy2
    ny = gz1 * gx2 - gx1 * gz2
    nz = gx1 * gy2 - gy1 * gx2
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = (nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]) / mag
    lam = (lam - lam.min()) / (lam.max() - lam.min())
    return lo + (1.0 - lo) * lam


def to_cartesian(R, Z, nfp):
    """Replicate one field period into the full torus (phi = zeta/nfp),
    closing both periodic grid directions."""
    nz = R.shape[1]
    zt = np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False)
    Rfull = np.concatenate([R] * nfp, axis=1)
    Zfull = np.concatenate([Z] * nfp, axis=1)
    phi = np.concatenate([(zt + 2.0 * np.pi * k) / nfp for k in range(nfp)])
    Rfull = periodic_close(Rfull)
    Zfull = periodic_close(Zfull)
    phi = np.append(phi, 2.0 * np.pi)
    return Rfull * np.cos(phi[None, :]), Rfull * np.sin(phi[None, :]), Zfull


def _circular_filament(radius, center_z, circular_points):
    phi = np.linspace(0.0, 2.0 * np.pi, circular_points, endpoint=False)
    return np.column_stack(
        [radius * np.cos(phi), radius * np.sin(phi),
         np.full_like(phi, center_z)])


def _load_json_coil_filaments(path, circular_points):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: could not read coils file {path}: {exc}") \
            from exc

    if not isinstance(document, dict) or \
            document.get("schema") != "cumes-coils-v1":
        raise SystemExit(
            f"error: {path}: expected schema 'cumes-coils-v1'")
    periods = document.get("field_periods")
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        raise SystemExit(f"error: {path}: field_periods must be positive")
    circuits = document.get("circuits")
    if not isinstance(circuits, list) or not circuits:
        raise SystemExit(f"error: {path}: circuits must be a nonempty array")
    if any(not isinstance(circuit, dict) for circuit in circuits):
        raise SystemExit(f"error: {path}: each circuit must be an object")
    indices = [circuit.get("current_index") for circuit in circuits]
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in indices):
        raise SystemExit(
            f"error: {path}: every current_index must be an integer")
    circuits = sorted(circuits, key=lambda circuit: circuit.get(
        "current_index", -1))
    indices = [circuit.get("current_index") for circuit in circuits]
    if indices != list(range(len(circuits))):
        raise SystemExit(
            f"error: {path}: current_index values must be contiguous from zero")

    filaments = []
    for circuit in circuits:
        circuit_filaments = circuit.get("filaments")
        if not isinstance(circuit_filaments, list) or not circuit_filaments:
            raise SystemExit(
                f"error: {path}: every circuit needs a nonempty filaments array")
        for filament_data in circuit_filaments:
            if not isinstance(filament_data, dict):
                raise SystemExit(
                    f"error: {path}: each filament must be an object")
            winding = filament_data.get("winding_number")
            if isinstance(winding, bool) or not isinstance(winding, (int, float)) \
                    or not np.isfinite(winding) or winding == 0.0:
                raise SystemExit(
                    f"error: {path}: winding_number must be finite and nonzero")
            filament_type = filament_data.get("type")
            if filament_type == "polygon":
                try:
                    vertices = np.asarray(filament_data.get("vertices"),
                                          dtype=float)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"error: {path}: polygon vertices must be numeric") \
                        from exc
                if vertices.ndim != 2 or vertices.shape[0] < 3 or \
                        vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    raise SystemExit(
                        f"error: {path}: polygon vertices must be a finite Nx3 "
                        "array with N >= 3")
                if np.array_equal(vertices[0], vertices[-1]):
                    raise SystemExit(
                        f"error: {path}: JSON polygon closure is implicit; do "
                        "not repeat the first vertex")
                filaments.append(np.vstack([vertices, vertices[0]]))
            elif filament_type == "axisymmetric_circle":
                radius = filament_data.get("radius")
                center_z = filament_data.get("center_z")
                if isinstance(radius, bool) or \
                        not isinstance(radius, (int, float)) or \
                        not np.isfinite(radius) or radius <= 0.0:
                    raise SystemExit(
                        f"error: {path}: circle radius must be finite and positive")
                if isinstance(center_z, bool) or \
                        not isinstance(center_z, (int, float)) or \
                        not np.isfinite(center_z):
                    raise SystemExit(
                        f"error: {path}: circle center_z must be finite")
                filaments.append(_circular_filament(
                    radius, center_z, circular_points))
            else:
                raise SystemExit(
                    f"error: {path}: unsupported filament type {filament_type!r}")
    return periods, filaments


def load_coil_filaments(path, circular_points=160):
    """Read the polygon and axis-aligned circular filaments accepted by
    MAKEGRID coils-dot and cumes-coils-v1 JSON files.  Coordinates are
    Cartesian (x, y, z); circuit currents affect the field but not the
    geometry rendered here."""
    if os.path.splitext(path)[1].lower() == ".json":
        return _load_json_coil_filaments(path, circular_points)

    filaments = []
    vertices = []
    periods = None
    in_filaments = False
    found_end = False

    try:
        stream = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"error: could not open coils file {path}: {exc}") \
            from exc

    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            lower = line.lower()
            if not in_filaments:
                if lower.startswith("periods"):
                    fields = line.split()
                    if len(fields) != 2:
                        raise SystemExit(
                            f"error: {path}:{line_number}: expected "
                            "'periods <positive integer>'")
                    try:
                        periods = int(fields[1])
                    except ValueError as exc:
                        raise SystemExit(
                            f"error: {path}:{line_number}: invalid periods") \
                            from exc
                elif lower.startswith("begin filament"):
                    in_filaments = True
                continue

            if lower.startswith("mirror"):
                if not (lower.endswith("nil") or lower.endswith("nul")):
                    raise SystemExit(
                        f"error: {path}:{line_number}: coil mirroring is "
                        "unsupported; expected mirror NIL or NUL")
                continue
            if lower.startswith("end"):
                found_end = True
                break

            fields = line.split()
            if len(fields) not in (4, 6):
                raise SystemExit(
                    f"error: {path}:{line_number}: expected four or six "
                    "filament fields")
            try:
                vertex = tuple(float(value) for value in fields[:3])
            except ValueError as exc:
                raise SystemExit(
                    f"error: {path}:{line_number}: invalid filament vertex") \
                    from exc
            vertices.append(vertex)
            if len(fields) == 6:
                if len(vertices) == 1:
                    radius, _, center_z = vertices[0]
                    if radius <= 0.0:
                        raise SystemExit(
                            f"error: {path}:{line_number}: circular filament "
                            "radius must be positive")
                    filament = _circular_filament(
                        radius, center_z, circular_points)
                else:
                    filament = np.asarray(vertices, dtype=float)
                filaments.append(filament)
                vertices = []

    if not in_filaments:
        raise SystemExit(f"error: {path}: no 'begin filament' section")
    if not found_end:
        raise SystemExit(f"error: {path}: no terminating 'end' line")
    if vertices:
        raise SystemExit(f"error: {path}: file ends inside a filament")
    if periods is None or periods < 1:
        raise SystemExit(f"error: {path}: no positive periods value")
    if not filaments:
        raise SystemExit(f"error: {path}: no coil filaments")
    return periods, filaments


def resolve_coils_path(requested, params):
    """Resolve an explicit or embedded coil-configuration path. Relative
    paths are first interpreted exactly as the solver does (from the current
    working directory), then relative to the recorded input file when
    available."""
    path = requested or params.get("coils_file", "")
    if not path:
        raise SystemExit(
            "error: --coils needs PATH because this state contains a "
            "precomputed mgrid_file but no embedded coils_file path")
    candidates = [path]
    source_path = params.get("_source_path", "")
    if not os.path.isabs(path) and source_path:
        candidates.append(os.path.join(os.path.dirname(source_path), path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    tried = ", ".join(os.path.abspath(candidate) for candidate in candidates)
    raise SystemExit(f"error: coils file not found; tried {tried}")


def tube_mesh(points, radius, n_sides=10):
    """Build a closed tube around a filament centerline using a smoothly
    transported local frame.  Coils-dot polygon filaments are closed by
    convention; explicitly repeated terminal vertices are collapsed before
    the periodic mesh is formed."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("filament points must have shape (n, 3)")
    keep = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    points = points[keep]
    if len(points) > 2 and np.linalg.norm(points[-1] - points[0]) <= 1e-12:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("a tube filament needs at least three unique points")

    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm <= 1e-14):
        raise ValueError("filament has a degenerate tangent")
    tangent /= tangent_norm[:, None]

    axes = np.eye(3)
    reference = axes[np.argmin(np.abs(axes @ tangent[0]))]
    normal = np.empty_like(points)
    normal[0] = np.cross(tangent[0], reference)
    normal[0] /= np.linalg.norm(normal[0])
    for i in range(1, len(points)):
        transported = normal[i - 1] - np.dot(normal[i - 1], tangent[i]) * \
            tangent[i]
        magnitude = np.linalg.norm(transported)
        if magnitude <= 1e-12:
            reference = axes[np.argmin(np.abs(axes @ tangent[i]))]
            transported = np.cross(tangent[i], reference)
            magnitude = np.linalg.norm(transported)
        normal[i] = transported / magnitude
    binormal = np.cross(tangent, normal)

    points = np.vstack([points, points[0]])
    normal = np.vstack([normal, normal[0]])
    binormal = np.vstack([binormal, binormal[0]])
    angle = np.linspace(0.0, 2.0 * np.pi, n_sides + 1)
    mesh = points[:, :, None] + radius * (
        normal[:, :, None] * np.cos(angle)[None, None, :] +
        binormal[:, :, None] * np.sin(angle)[None, None, :])
    return mesh[:, 0, :], mesh[:, 1, :], mesh[:, 2, :]


def converged_axis(fams, ns, ntor, nfp, n=240):
    """The converged magnetic axis: the m=0 content of the innermost
    computed surface (j=1). The axis is theta-independent by construction
    (only m=0 modes enter), so the curve runs through the center of every
    cross-section and cannot leave the plasma:

        R_ax(zeta) = sum_n rmncc(0, n, j=1) cos(n zeta)
        Z_ax(zeta) = sum_n zmncs(0, n, j=1) sin(n zeta)

    with zeta the per-field-period angle, replicated nfp times around the
    full torus (phi = zeta/nfp). The axis carries genuine nfp-fold
    modulation, matching the seed convention in seed_state.hpp
    (raxis_c/zaxis_s interpolate against the same folded toroidal
    mode n)."""
    zt = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R = np.zeros_like(zt)
    Z = np.zeros_like(zt)
    for nn in range(ntor + 1):
        R += fams["rmncc"][nn * ns + 1] * np.cos(nn * zt)
        Z += fams["zmncs"][nn * ns + 1] * np.sin(nn * zt)
    Rfull = np.concatenate([R] * nfp)
    Zfull = np.concatenate([Z] * nfp)
    phi = np.concatenate([(zt + 2.0 * np.pi * k) / nfp for k in range(nfp)])
    X = Rfull * np.cos(phi)
    Y = Rfull * np.sin(phi)
    return np.append(X, X[0]), np.append(Y, Y[0]), np.append(Zfull, Zfull[0])


def field_lines(edge, th, zt, seeds, nfp, zeta_span=6 * 2.0 * np.pi,
                n_steps=2400):
    """Trace d(theta)/d(zeta) = B^theta/B^zeta on the edge half-grid surface
    (RK4, periodic in both angles), lifted onto the (r12, z12) geometry.
    `edge` is the half_grid() result on the (th, zt) solver grid."""
    ratio = edge["bsupu"] / edge["bsupv"]
    ri = RegularGridInterpolator((th, zt), ratio, method="linear",
                                 bounds_error=False, fill_value=None)
    rgi = RegularGridInterpolator((th, zt), edge["r12"], method="linear",
                                  bounds_error=False, fill_value=None)
    zgi = RegularGridInterpolator((th, zt), edge["z12"], method="linear",
                                  bounds_error=False, fill_value=None)
    lines = []
    for seed in seeds:
        zeta = np.linspace(0.0, zeta_span, n_steps)
        theta = np.empty_like(zeta)
        theta[0] = seed
        h = zeta[1] - zeta[0]
        for i in range(n_steps - 1):
            th_i, zt_i = theta[i] % (2 * np.pi), zeta[i] % (2 * np.pi)
            k1 = ri([th_i, zt_i])[0]
            k2 = ri([(theta[i] + 0.5 * h * k1) % (2 * np.pi),
                     (zeta[i] + 0.5 * h) % (2 * np.pi)])[0]
            k3 = ri([(theta[i] + 0.5 * h * k2) % (2 * np.pi),
                     (zeta[i] + 0.5 * h) % (2 * np.pi)])[0]
            k4 = ri([(theta[i] + h * k3) % (2 * np.pi),
                     (zeta[i] + h) % (2 * np.pi)])[0]
            theta[i + 1] = theta[i] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts = np.stack([theta % (2 * np.pi), zeta % (2 * np.pi)], axis=1)
        r_line = rgi(pts)
        z_line = zgi(pts)
        phi = zeta / nfp
        lines.append((r_line * np.cos(phi), r_line * np.sin(phi), z_line))
    return lines


def _render_machinery(S):
    """Per-process color machinery (each render process builds its own —
    matplotlib state is not shared across processes)."""
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=S["bmin"], vmax=S["bmax"])
    light_dir = mcolors.LightSource(azdeg=330, altdeg=55).direction
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return cmap, norm, light_dir, mappable


def mesh_quads(X, Y, Z):
    """Convert a structured surface mesh into uniformly shaped quads."""
    points = np.stack([X, Y, Z], axis=-1)
    return np.stack(
        [points[:-1, :-1], points[1:, :-1],
         points[1:, 1:], points[:-1, 1:]], axis=2).reshape(-1, 4, 3)


def quad_face_colors(vertex_colors):
    """Average structured RGBA vertex colors onto the corresponding quads."""
    return (vertex_colors[:-1, :-1] + vertex_colors[1:, :-1] +
            vertex_colors[1:, 1:] + vertex_colors[:-1, 1:]).reshape(
                -1, 4) * 0.25


def shade_quads(quads, color, light_dir):
    """Apply Matplotlib-compatible diffuse shading to uniformly colored
    quads before they enter the shared depth-sorted collection."""
    normals = np.cross(quads[:, 0] - quads[:, 1],
                       quads[:, 1] - quads[:, 2])
    magnitudes = np.linalg.norm(normals, axis=1)
    shade = np.zeros(len(quads), dtype=float)
    valid = magnitudes > 0.0
    shade[valid] = normals[valid] @ light_dir / magnitudes[valid]
    # Axes3D._shade_colors maps a [-1, 1] dot product onto [0.3, 1].
    intensity = 0.65 + 0.35 * np.clip(shade, -1.0, 1.0)
    rgba = np.tile(mcolors.to_rgba(color), (len(quads), 1))
    rgba[:, :3] *= intensity[:, None]
    return rgba


def draw_scene(ax, vw, S, cmap, norm, light_dir):
    """Draw plasma and coils as one globally face-sorted collection.

    Matplotlib has no depth buffer: separate Poly3DCollection objects receive
    one painter-order depth each. Combining every plasma and coil face lets
    its internal polygon sorter interleave the two geometries by view depth.
    """
    geometry = []
    colors = []
    antialiased = []
    for s in S["surf_3d"]:  # a single opaque surface (the plasma boundary)
        X, Y, Z = to_cartesian(s["r12"], s["z12"], S["nfp"])
        B = periodic_close(np.concatenate([s["bmag"]] * S["nfp"], axis=1))
        lam = surface_intensity(X, Y, Z, light_dir)
        hsv = mcolors.rgb_to_hsv(cmap(norm(B))[:, :, :3])
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * lam, 0.0, 1.0)
        face = np.concatenate(
            [mcolors.hsv_to_rgb(hsv), np.ones(hsv.shape[:2] + (1,))], axis=2)
        # Match the former plot_surface cstride=2 while keeping uniform quads.
        plasma_quads = mesh_quads(X[:, ::2], Y[:, ::2], Z[:, ::2])
        geometry.append(plasma_quads)
        colors.append(quad_face_colors(face[:, ::2]))
        antialiased.append(np.zeros(len(plasma_quads), dtype=bool))
    for filament in S["coils"]:
        X, Y, Z = tube_mesh(filament, S["coil_radius"])
        coil_quads = mesh_quads(X, Y, Z)
        geometry.append(coil_quads)
        colors.append(shade_quads(coil_quads, BRONZE, light_dir))
        antialiased.append(np.ones(len(coil_quads), dtype=bool))
    if geometry:
        surfaces = Poly3DCollection(
            np.concatenate(geometry),
            facecolors=np.concatenate(colors), edgecolors="none",
            linewidths=0.0, antialiaseds=np.concatenate(antialiased),
            zsort="average")
        ax.add_collection3d(surfaces)
    for (x, y, z) in S["lines"]:
        ax.plot(x, y, z, color="#0b0b0b", linewidth=1.1, alpha=0.85, zorder=50)
    ax.plot(S["axx"], S["axy"], S["axz"], color="#0b0b0b", linewidth=1.4,
            zorder=60)
    # Data-derived limits hug the plasma so the torus fills the frame
    # instead of floating in white space.
    lim = S["lim"]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-S["zlim"], S["zlim"])
    ax.set_box_aspect(S["box_aspect"])
    ax.set_axis_off()
    ax.view_init(**vw)


def render_single_view(base, suffix, label, vw, S):
    """One 3-D view per file (perspective / top); runs in its own process."""
    print(f"rendering {label} ...", flush=True)
    t0 = time.perf_counter()
    cmap, norm, light_dir, mappable = _render_machinery(S)
    # constrained_layout: 3-D projections reserve wide margins for
    # rotation; constrained_layout + bbox_inches='tight' mitigates them
    # (the rest is handled by trim_white).
    fig = plt.figure(figsize=(7.4, 5.6), dpi=100, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax, vw, S, cmap, norm, light_dir)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, aspect=20, pad=0.01)
    cbar.set_label("|B| (T)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    # The title is re-centered onto the torus by trim_white (the 3-D
    # box is not centered in the axes window, so matplotlib-side
    # placement cannot do it).
    fig.suptitle(f"{S['title']} ({label})", fontsize=11, y=1.01)
    out_png = f"{base}_{suffix}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def render_combined(base, S):
    """Two side-by-side views with one shared horizontal colorbar."""
    print("rendering combined view ...", flush=True)
    t0 = time.perf_counter()
    cmap, norm, light_dir, mappable = _render_machinery(S)
    # Plain side-by-side subplots; the figure is sized so each panel window
    # is nearly square (the top-view box then fills it and the inter-panel
    # slack vanishes). trim_white removes the remaining projection margins.
    fig = plt.figure(figsize=(8.8, 5.4), dpi=100)
    axes = []
    for i, (suffix, label, vw) in enumerate(S["views"]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        axes.append(ax)
        draw_scene(ax, vw, S, cmap, norm, light_dir)
        # mplot3d ignores the title pad, so place the label manually just
        # above the rendered box top (measured ~0.62 of the window).
        label_y = 0.92 if S["coils"] else 0.70
        ax.text2D(0.5, label_y, label, transform=ax.transAxes, ha="center",
                  va="bottom", fontsize=10)
    cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal",
                        shrink=0.6, aspect=24, pad=0.16)
    cbar.set_label("|B| (T)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=0.0)
    # tight_layout overrides the colorbar pad; pin the strip just below
    # the rendered 3-D box. The box's projected bottom sits at ~0.188 of
    # the figure height (mplot3d leaves a margin inside the window), so
    # the strip top = 0.188 - 60px gap.
    cbar.ax.set_position([0.30, 0.106, 0.40, 0.045])
    fig.suptitle(S["title"], fontsize=11, y=0.98)
    out_png = f"{base}_combined.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def render_slices(base, S):
    """Top view + six RZ poloidal cross-sections (two rows of three,
    spanning one field period), each with the nested flux-surface contour
    family."""
    print("rendering top view + RZ slices ...", flush=True)
    t0 = time.perf_counter()
    cmap, norm, light_dir, mappable = _render_machinery(S)
    surf_slices = S["surf_slices"]
    fams, ns, nzt_r = S["fams"], S["ns"], S["nzt_r"]
    nfp, ntor = S["nfp"], S["ntor"]
    zeta_cuts = [(k * np.pi / 3.0,
                  f"φ = {np.degrees(k * np.pi / (3.0 * nfp)):.0f}°")
                 for k in range(6)]
    # Explicit layout: the slice block gets a bounded region that is
    # SHORTER than the 3-D panel, so the two-row slice block ends up
    # smaller than the top-view torus.
    fig = plt.figure(figsize=(10.0, 5.6), dpi=100)
    gs = fig.add_gridspec(2, 6, left=0.05, right=0.44, top=0.72,
                          bottom=0.30, hspace=0.35, wspace=0.06)
    # Two rows of three slices (2 columns each).
    axs = [fig.add_subplot(gs[0, 0:2]),
           fig.add_subplot(gs[0, 2:4]),
           fig.add_subplot(gs[0, 4:6]),
           fig.add_subplot(gs[1, 0:2]),
           fig.add_subplot(gs[1, 2:4]),
           fig.add_subplot(gs[1, 4:6])]
    # Nearly square window (matches the top-view box) so the torus fills
    # it and the gap to the slice block shrinks.
    ax3d = fig.add_axes([0.455, 0.06, 0.48, 0.86], projection="3d")
    draw_scene(ax3d, dict(elev=90.0, azim=-90.0), S, cmap, norm, light_dir)
    # Mark the toroidal positions of the cross-sections on the top view
    # with radial dashed lines (one period only).
    lim = S["lim"]
    for zc, _ in zeta_cuts:
        phi = zc / nfp
        ax3d.plot([0.0, lim * np.cos(phi)], [0.0, lim * np.sin(phi)],
                  [0.0, 0.0], color="#0b0b0b", linestyle=(0, (5, 4)),
                  linewidth=0.9, zorder=55)
    r_all = np.concatenate([s["r12"].ravel() for s in surf_slices])
    z_all = np.concatenate([s["z12"].ravel() for s in surf_slices])
    Rmin, Rmax = r_all.min(), r_all.max()
    Zmax = max(abs(z_all.min()), abs(z_all.max()))
    for i, (zc, name) in enumerate(zeta_cuts):
        idx = int(round(zc / (2.0 * np.pi) * nzt_r)) % nzt_r
        ax = axs[i]
        for k, s in enumerate(surf_slices):
            R = np.append(s["r12"][:, idx], s["r12"][0, idx])
            Z = np.append(s["z12"][:, idx], s["z12"][0, idx])
            B = np.append(s["bmag"][:, idx], s["bmag"][0, idx])
            pts = np.stack([R, Z], axis=1)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm,
                                linewidths=2.4 if k == 0 else 1.0)
            lc.set_array(B[:-1])
            ax.add_collection(lc)
        Rax = sum(fams["rmncc"][nn * ns + 1] * np.cos(nn * zc)
                  for nn in range(ntor + 1))
        Zax = sum(fams["zmncs"][nn * ns + 1] * np.sin(nn * zc)
                  for nn in range(ntor + 1))
        ax.plot([Rax], [Zax], "o", color="#0b0b0b", markersize=2.5)
        ax.set_xlim(Rmin - 0.05, Rmax + 0.05)
        # Mild vertical headroom; the slice plots fill their cell heights
        # either way, so this only controls their width.
        ax.set_ylim(-1.1 * Zmax, 1.1 * Zmax)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=8.5, pad=2)
        ax.tick_params(labelsize=7.5)
        # One vertical frame label per row (first slice of each row only).
        if i in (0, 3):
            ax.set_ylabel("Z (m)", fontsize=8)
        if i < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("R (m)", fontsize=8)
    cbar = fig.colorbar(mappable, ax=ax3d, shrink=0.6, aspect=20, pad=0.01)
    cbar.set_label("|B| (T)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(f"{S['title']} (top view + poloidal cross-sections)",
                 fontsize=11, y=1.01)
    out_png = f"{base}_slices.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white(out_png, center="body")
    print(f"saved {out_png} ({time.perf_counter() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="figure_data/w7x_state.bin")
    ap.add_argument("--out", default="figure_data/equilibrium.png")
    ap.add_argument("--field-lines", action="store_true",
                    help="also trace and draw field lines on the edge surface "
                         "(off by default: the figure stays cleaner)")
    ap.add_argument("--coils", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="for a free-boundary state, overlay the MAKEGRID "
                         "coil filaments as bronze tubes; omit PATH to use "
                         "the state's embedded coils_file")
    ap.add_argument("--coil-radius", type=float, default=None,
                    metavar="METERS",
                    help="coil tube radius in meters (default: 0.6%% of the "
                         "plasma's major radial extent)")
    args = ap.parse_args()

    if args.coil_radius is not None and args.coil_radius <= 0.0:
        ap.error("--coil-radius must be positive")
    if args.coil_radius is not None and args.coils is None:
        ap.error("--coil-radius requires --coils")

    output_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_dir, exist_ok=True)

    ns, mnmax, fams, params, name = load_state(args.state)
    if args.coils is not None and not params["lfreeb"]:
        ap.error("--coils is only valid for free-boundary states")
    ntor, nfp = params["ntor"], params["nfp"]
    if mnmax != params["mpol"] * (ntor + 1):
        raise SystemExit(
            f"error: mnmax {mnmax} != mpol*(ntor+1) "
            f"{params['mpol']}*{ntor + 1} — corrupt embedded input record")
    prof = make_profiles(params, ns)
    print(f"loaded state: ns={ns}, mnmax={mnmax}, mpol={params['mpol']}, "
          f"ntor={ntor}, nfp={nfp}, ntheta={params['ntheta']}, "
          f"nzeta={params['nzeta']}, ncurr={params['ncurr']}, "
          f"phiedge={params['phiedge']:g}, lfreeb={params['lfreeb']}",
          flush=True)

    # ---- self-check 1: state LCFS vs the embedded initial boundary ------
    print("self-check 1/2: state LCFS vs embedded initial rbc/zbs boundary ...",
          flush=True)
    th_ax = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    zt_ax = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    b = eval_state(fams, ns, ns - 1, th_ax, zt_ax, ntor, nfp)
    maxsc = max(np.sqrt((ns - 1) / (ns - 1)), np.sqrt(1.0 / (ns - 1)))
    R_phys = b["re"] + maxsc * b["ro"]
    Z_phys = b["ze"] + maxsc * b["zo"]
    TH_c, ZT_c = np.meshgrid(th_ax, zt_ax, indexing="ij")
    Rb, Zb = boundary_from_params(params, TH_c, ZT_c)
    err = max(np.max(np.abs(R_phys - Rb)), np.max(np.abs(Z_phys - Zb)))
    if not np.isfinite(err):
        raise SystemExit("error: non-finite LCFS boundary reconstruction")
    if params["lfreeb"]:
        print(f"  initial-to-converged max displacement = {err:.3e} "
              "(expected for free boundary)", flush=True)
    else:
        print(f"  max err = {err:.3e}", flush=True)
        assert err < 1e-8, "state LCFS does not match the embedded boundary"

    # ---- the plotted flux surface (plasma boundary, edge half-grid) ------
    print(f"edge surface: solving chi' + half-grid geometry (jh = {ns - 2}) ...",
          flush=True)
    th = np.linspace(0.0, 2.0 * np.pi, params["ntheta"], endpoint=False)
    zt = np.linspace(0.0, 2.0 * np.pi, params["nzeta"], endpoint=False)
    chip = solve_chip(fams, ns, ns - 2, params, prof)
    a = [eval_state(fams, ns, j, th, zt, ntor, nfp)
         for j in (ns - 2, ns - 1)]
    edge = half_grid(a, ns, ns - 2, chip, prof["phip_avg"](ns - 2),
                     prof["lamscale"])
    print(f"  jh={ns - 2:3d}: chip={chip:+.4e}, |B| "
          f"{edge['bmag'].min():.3f}-{edge['bmag'].max():.3f} T", flush=True)

    # ---- fine render grid (analytic reconstruction, same formulas) -------
    # One flux surface for the 3-D figures (the plasma boundary); the RZ
    # slices get a family of nested contours from the edge down to s ~ 0.1.
    print("fine render grid: evaluating geometry + |B| on 240x120 ...", flush=True)
    nth_r, nzt_r = 240, 120
    th_r = np.linspace(0.0, 2.0 * np.pi, nth_r, endpoint=False)
    zt_r = np.linspace(0.0, 2.0 * np.pi, nzt_r, endpoint=False)
    jh_slices = tuple(range(ns - 2, 8, -8))   # the edge down to s ~ 0.1
    surf_slices = []
    for jh in jh_slices:
        chip = solve_chip(fams, ns, jh, params, prof)
        a = [eval_state(fams, ns, j, th_r, zt_r, ntor, nfp)
             for j in (jh, jh + 1)]
        surf_slices.append(half_grid(a, ns, jh, chip, prof["phip_avg"](jh),
                                     prof["lamscale"]))
    surf_3d = [surf_slices[0]]               # the edge is the first contour
    b_all = np.concatenate([s["bmag"].ravel() for s in surf_slices])
    print(f"  3D: 1 flux surface, slices: {len(surf_slices)} contours, "
          f"|B| range {b_all.min():.3f} - {b_all.max():.3f} T", flush=True)

    # ---- field lines on the edge surface (opt-in) ------------------------
    lines = []
    if args.field_lines:
        print("field lines: tracing d(theta)/d(zeta) = B^theta/B^zeta (RK4) ...",
              flush=True)
        try:
            lines = field_lines(edge, th, zt, nfp=nfp,
                                seeds=[0.0, 2.0 * np.pi / 3, 4.0 * np.pi / 3])
            print(f"  {len(lines)} lines traced", flush=True)
        except Exception as exc:  # decorative; the figure survives without them
            lines = []
            print(f"  field lines skipped: {exc}", flush=True)

    # ---- converged magnetic axis (m=0 content of the innermost surface) --
    axx, axy, axz = converged_axis(fams, ns, ntor, nfp)
    print(f"magnetic axis: R {axx.min():.3f}-{axx.max():.3f}, "
          f"Z {axz.min():.3f}-{axz.max():.3f}", flush=True)

    # ---- rendering (four figures, one process per figure) -----------------
    # matplotlib's Agg rasterizer largely holds the GIL and pyplot state is
    # not thread-safe across concurrent figures, so threads would serialize;
    # forked processes run in parallel and inherit the shared data
    # copy-on-write (no pickling of the surface arrays). High-contrast
    # viridis (bright yellow = large |B|, deep purple = small |B|) is mapped
    # over the actual data range; shading uses the geometric surface normals
    # (surface_intensity), never the |B| heightfield.
    # Data-derived framing: the render window hugs the plotted surface with
    # a small margin, so any device fills the frame.
    r_edge = surf_3d[0]["r12"]
    z_edge = surf_3d[0]["z12"]
    plasma_lim = float(np.max(r_edge))
    plasma_zlim = float(max(abs(z_edge.min()), abs(z_edge.max())))
    coils = []
    coil_radius = 0.0
    if args.coils is not None:
        coils_path = resolve_coils_path(args.coils, params)
        coil_periods, loaded_coils = load_coil_filaments(coils_path)
        if coil_periods != nfp:
            raise SystemExit(
                f"error: coils file has {coil_periods} field periods, state "
                f"has nfp={nfp}")
        # Idealized axis-current filaments can use million-meter closure
        # segments.  They produce the intended field but are not physical
        # coils and would collapse the useful plot scale, so omit only such
        # unmistakable far-field constructions.
        visible_extent = 50.0 * max(plasma_lim, plasma_zlim)
        coils = [filament for filament in loaded_coils
                 if np.max(np.linalg.norm(filament, axis=1)) <= visible_extent]
        skipped = len(loaded_coils) - len(coils)
        if not coils:
            raise SystemExit("error: no coil filaments lie within the useful "
                             "plot extent")
        coil_radius = args.coil_radius \
            if args.coil_radius is not None else 0.006 * plasma_lim
        print(f"coils: loaded {len(coils)} filament(s) from {coils_path}; "
              f"bronze tube radius={coil_radius:.4g} m" +
              (f"; skipped {skipped} far-field filament(s)" if skipped else ""),
              flush=True)

    coil_xy_lim = max((np.max(np.hypot(f[:, 0], f[:, 1]))
                       for f in coils), default=0.0)
    coil_zlim = max((np.max(np.abs(f[:, 2])) for f in coils), default=0.0)
    lim = 1.06 * max(plasma_lim, coil_xy_lim + coil_radius)
    zlim = 1.10 * max(plasma_zlim, coil_zlim + coil_radius)
    views = [("perspective", "perspective view", dict(elev=24.0, azim=-58.0)),
             ("top", "top view", dict(elev=90.0, azim=-90.0))]
    S = {
        "surf_3d": surf_3d, "surf_slices": surf_slices,
        "fams": fams, "ns": ns, "nzt_r": nzt_r,
        "nfp": nfp, "ntor": ntor,
        "lines": lines, "axx": axx, "axy": axy, "axz": axz,
        "coils": coils, "coil_radius": coil_radius,
        "bmin": b_all.min(), "bmax": b_all.max(),
        "views": views,
        "title": f"cuMES converged equilibrium — {name}",
        "lim": lim, "zlim": zlim,
        "box_aspect": (1.0, 1.0, zlim / lim),
    }
    base = os.path.splitext(args.out)[0]
    jobs = [
        (render_single_view, (base, "perspective", "perspective view",
                              dict(elev=24.0, azim=-58.0), S)),
        (render_single_view, (base, "top", "top view",
                              dict(elev=90.0, azim=-90.0), S)),
        (render_combined, (base, S)),
        (render_slices, (base, S)),
    ]
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # no fork on this platform: spawn pickles S instead
        ctx = multiprocessing.get_context()
    procs = [ctx.Process(target=fn, args=args) for fn, args in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise SystemExit(f"error: {len(failed)} render process(es) failed "
                         f"(exit codes {failed})")
    print("all figures saved", flush=True)

if __name__ == "__main__":
    main()
