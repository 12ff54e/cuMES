// output_netcdf.cpp — netCDF classic-3 output of the solved state.
// Compiled only when CUMES_HAVE_NETCDF is defined (CMake, CUMES_USE_NETCDF
// option + find_package/pkg-config). Content mirrors outputSaveBinary (the
// 6 coefficient families, on disk double regardless of T) plus grid params,
// convergence and the full InputParams provenance, with global attributes
// input_file / precision.
//
// Layout note: the in-memory coefficient buffers are mode-major (index
// m*ns + j, column-major over surfaces), while the file variables use dims
// (ns, mnmax) in C order. A naive whole-buffer put would silently
// transpose, so each mode is written as a (ns, 1) hyperslab with the
// contiguous column at dbuf + m*ns.
#include "output.cuh"
#include "cumes/io/legacy_provenance.hpp"
#define CUMES_IO_DEVICE_STAGE 1  // opt in to FamilyStage (needs CUDA headers)
#include "cumes/io/writer_helpers.hpp"  // io_detail::tempPathFor/renamePublish, FamilyStage
#include "solver.cuh"
#include <cuda_runtime.h>  // cudaMemcpy (host runtime API)
#include <netcdf.h>
#include <cstdio>
#include <cstdlib>   // getpid
#include <cstring>
#include <string>
#include <unistd.h>   // getpid, rename

#include "cumes/runtime/cuda_status.hpp"

template <typename T>
bool outputSaveNetcdf(const cumes::SpectralStorage<T>& storage, const DeviceParams<T>& p,
                      const cumes::ValidatedProblem& vp, const SolverResult<T>& result,
                      const char* path, const char* input_file) {
    // Fixed-capacity provenance for the v0 layout (padded arrays, byte-identical
    // to the pre-overhaul InputParams).
    const cumes::LegacyInputProvenance pv =
        cumes::LegacyInputProvenance::from_validated(vp);
    // Atomic publication: create the file at a same-directory temp path, then
    // rename() it over `path` after a successful close, so a reader never sees
    // a half-written file and a failure leaves the target untouched. The temp
    // path must live in the same directory as the target for an atomic rename.
    const std::string tmp = cumes::io_detail::tempPathFor(path);
    // NC_CLOBBER alone gives classic-3 (CDF-1); ncdump -k reports "classic".
    int ncid = -1;
    int rc = nc_create(tmp.c_str(), NC_CLOBBER, &ncid);
    if (rc != NC_NOERR) {
        fprintf(stderr, "NetCDF error [nc_create %s]: %s\n", tmp.c_str(),
                nc_strerror(rc));
        return false;
    }
    // On any later error: close, delete the half-written file, return false
    // (the caller folds output success into the CLI exit code). A failure OF
    // nc_close itself must not be re-closed (double-close UB) — the closing
    // call is handled at the call site, not through this macro. The same fail
    // path also serves the device-copy failure (the D2H copy reports instead
    // of throwing, so a GPU fault still removes the temp and returns false).
    auto fail = [&](const char* tag, const std::string& msg) -> bool {
        fprintf(stderr, "NetCDF error [%s]: %s\n", tag, msg.c_str());
        nc_close(ncid);
        remove(tmp.c_str());
        return false;
    };
#define NC_CHECK(rc_, tag)                                                   \
    do {                                                                     \
        int _rc = (rc_);                                                     \
        if (_rc != NC_NOERR) return fail(tag, nc_strerror(_rc));             \
    } while (0)

    // ---- dimensions ----
    int dim_ns, dim_mnmax, dim_ngrids, dim_ncoeff, dim_naxis, dim_nbm, dim_nbn;
    NC_CHECK(nc_def_dim(ncid, "ns", (size_t)p.ns, &dim_ns), "def dim ns");
    NC_CHECK(nc_def_dim(ncid, "mnmax", (size_t)p.mnmax, &dim_mnmax), "def dim mnmax");
    NC_CHECK(nc_def_dim(ncid, "ngrids", cumes::LegacyInputProvenance::kMaxGrids, &dim_ngrids), "def dim ngrids");
    NC_CHECK(nc_def_dim(ncid, "ncoeff", cumes::LegacyInputProvenance::kMaxCoeff, &dim_ncoeff), "def dim ncoeff");
    NC_CHECK(nc_def_dim(ncid, "naxis", cumes::LegacyInputProvenance::kMaxAxis, &dim_naxis), "def dim naxis");
    NC_CHECK(nc_def_dim(ncid, "nbm", cumes::LegacyInputProvenance::kMaxM, &dim_nbm), "def dim nbm");
    NC_CHECK(nc_def_dim(ncid, "nbn", cumes::LegacyInputProvenance::kMaxN, &dim_nbn), "def dim nbn");

    // ---- state variables (ns, mnmax) ----
    const int state_dims[2] = {dim_ns, dim_mnmax};
    int v_rmncc, v_zmnsc, v_lmnsc, v_rmnss, v_zmncs, v_lmncs;
    NC_CHECK(nc_def_var(ncid, "rmncc", NC_DOUBLE, 2, state_dims, &v_rmncc), "def rmncc");
    NC_CHECK(nc_def_var(ncid, "zmnsc", NC_DOUBLE, 2, state_dims, &v_zmnsc), "def zmnsc");
    NC_CHECK(nc_def_var(ncid, "lmnsc", NC_DOUBLE, 2, state_dims, &v_lmnsc), "def lmnsc");
    NC_CHECK(nc_def_var(ncid, "rmnss", NC_DOUBLE, 2, state_dims, &v_rmnss), "def rmnss");
    NC_CHECK(nc_def_var(ncid, "zmncs", NC_DOUBLE, 2, state_dims, &v_zmncs), "def zmncs");
    NC_CHECK(nc_def_var(ncid, "lmncs", NC_DOUBLE, 2, state_dims, &v_lmncs), "def lmncs");

    // ---- scalar variables (0 dims) ----
    struct IntScalar { const char* name; int value; };
    const IntScalar int_scalars[] = {
        {"mpol", p.mpol}, {"ntor", p.ntor}, {"nfp", p.nfp},
        {"ntheta", p.ntheta}, {"nzeta", p.nzeta},
        {"ns", p.ns}, {"mnmax", p.mnmax}, {"nZnT", p.nZnT},
        {"ncurr", p.ncurr}, {"max_iter", p.max_iter},
        {"n_grids", pv.n_grids},
        {"am_n", pv.am_n}, {"ac_n", pv.ac_n}, {"ai_n", pv.ai_n},
        {"aphi_n", pv.aphi_n}, {"raxis_n", pv.raxis_n},
        {"rbc_n", pv.rbc_n}, {"zbs_n", pv.zbs_n},
        {"iterations", result.iterations},
        {"converged", result.converged ? 1 : 0},
    };
    int v_int_scalar[sizeof(int_scalars) / sizeof(int_scalars[0])];
    for (size_t i = 0; i < sizeof(int_scalars) / sizeof(int_scalars[0]); ++i) {
        NC_CHECK(nc_def_var(ncid, int_scalars[i].name, NC_INT, 0, nullptr,
                            &v_int_scalar[i]), "def int scalar");
    }
    struct DblScalar { const char* name; double value; };
    const DblScalar dbl_scalars[] = {
        {"delt", (double)p.delt}, {"ftol", (double)p.ftol},
        {"lamscale", (double)p.lamscale},
        {"phiedge", pv.phiedge}, {"pres_scale", pv.pres_scale},
        {"adiabatic_index", pv.adiabatic_index}, {"spres_ped", pv.spres_ped},
        {"bloat", pv.bloat}, {"curtor", pv.curtor}, {"tcon0", pv.tcon0},
        {"fsqr", (double)result.fsqr}, {"fsqz", (double)result.fsqz},
        {"fsql", (double)result.fsql},
    };
    int v_dbl_scalar[sizeof(dbl_scalars) / sizeof(dbl_scalars[0])];
    for (size_t i = 0; i < sizeof(dbl_scalars) / sizeof(dbl_scalars[0]); ++i) {
        NC_CHECK(nc_def_var(ncid, dbl_scalars[i].name, NC_DOUBLE, 0, nullptr,
                            &v_dbl_scalar[i]), "def double scalar");
    }

    // ---- provenance arrays ----
    int v_ns_array, v_niter_array, v_ftol_array;
    NC_CHECK(nc_def_var(ncid, "ns_array", NC_INT, 1, &dim_ngrids, &v_ns_array), "def ns_array");
    NC_CHECK(nc_def_var(ncid, "niter_array", NC_INT, 1, &dim_ngrids, &v_niter_array), "def niter_array");
    NC_CHECK(nc_def_var(ncid, "ftol_array", NC_DOUBLE, 1, &dim_ngrids, &v_ftol_array), "def ftol_array");
    int v_am, v_ac, v_ai, v_aphi;
    NC_CHECK(nc_def_var(ncid, "am", NC_DOUBLE, 1, &dim_ncoeff, &v_am), "def am");
    NC_CHECK(nc_def_var(ncid, "ac", NC_DOUBLE, 1, &dim_ncoeff, &v_ac), "def ac");
    NC_CHECK(nc_def_var(ncid, "ai", NC_DOUBLE, 1, &dim_ncoeff, &v_ai), "def ai");
    NC_CHECK(nc_def_var(ncid, "aphi", NC_DOUBLE, 1, &dim_ncoeff, &v_aphi), "def aphi");
    int v_raxis_c, v_zaxis_s;
    NC_CHECK(nc_def_var(ncid, "raxis_c", NC_DOUBLE, 1, &dim_naxis, &v_raxis_c), "def raxis_c");
    NC_CHECK(nc_def_var(ncid, "zaxis_s", NC_DOUBLE, 1, &dim_naxis, &v_zaxis_s), "def zaxis_s");
    const int bnd_dims[2] = {dim_nbm, dim_nbn};
    int v_rbcc, v_rbss, v_zbsc, v_zbcs;
    NC_CHECK(nc_def_var(ncid, "rbcc", NC_DOUBLE, 2, bnd_dims, &v_rbcc), "def rbcc");
    NC_CHECK(nc_def_var(ncid, "rbss", NC_DOUBLE, 2, bnd_dims, &v_rbss), "def rbss");
    NC_CHECK(nc_def_var(ncid, "zbsc", NC_DOUBLE, 2, bnd_dims, &v_zbsc), "def zbsc");
    NC_CHECK(nc_def_var(ncid, "zbcs", NC_DOUBLE, 2, bnd_dims, &v_zbcs), "def zbcs");

    // ---- global attributes ----
    const char* input = input_file ? input_file : "";
    NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "input_file", strlen(input), input),
             "attr input_file");
    const char* prec = (sizeof(T) == sizeof(double)) ? "double" : "float";
    NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "precision", strlen(prec), prec),
             "attr precision");

    NC_CHECK(nc_enddef(ncid), "nc_enddef");

    // ---- state data (per-mode hyperslabs, see layout note at the top) ----
    const auto n_opt = cumes::io_detail::familyCount(p.ns, p.mnmax);
    if (!n_opt) return fail("write state", "ns * mnmax overflows size_t");
    const std::size_t n = *n_opt;
    // RAII staging (FamilyStage): the buffers cannot leak on a write failure,
    // and a CUDA copy error is reported (fail path) instead of thrown.
    cumes::io_detail::FamilyStage<T> stage(n);
    auto writeFam = [&](const T* d, int varid, const char* tag) -> bool {
        std::string reason;
        if (!stage.copy(d, tag, reason)) return fail(tag, reason);
        const double* dbuf = stage.data();
        for (int m = 0; m < p.mnmax; ++m) {
            const size_t start[2] = {0, (size_t)m};
            const size_t count[2] = {(size_t)p.ns, 1};
            int rc2 = nc_put_vara_double(ncid, varid, start, count,
                                         dbuf + (size_t)m * p.ns);
            if (rc2 != NC_NOERR) return fail(tag, nc_strerror(rc2));
        }
        return true;
    };
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Rcc), v_rmncc, "write rmncc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Zsc), v_zmnsc, "write zmnsc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Lsc), v_lmnsc, "write lmnsc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Rss), v_rmnss, "write rmnss")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Zcs), v_zmncs, "write zmncs")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Lcs), v_lmncs, "write lmncs")) return false;

    // ---- scalar data ----
    for (size_t i = 0; i < sizeof(int_scalars) / sizeof(int_scalars[0]); ++i) {
        NC_CHECK(nc_put_var_int(ncid, v_int_scalar[i], &int_scalars[i].value),
                 "put int scalar");
    }
    for (size_t i = 0; i < sizeof(dbl_scalars) / sizeof(dbl_scalars[0]); ++i) {
        NC_CHECK(nc_put_var_double(ncid, v_dbl_scalar[i], &dbl_scalars[i].value),
                 "put double scalar");
    }

    // ---- provenance data (full fixed-size arrays; n_grids records usage) ----
    NC_CHECK(nc_put_var_int(ncid, v_ns_array, pv.ns_array), "put ns_array");
    NC_CHECK(nc_put_var_int(ncid, v_niter_array, pv.niter_array), "put niter_array");
    NC_CHECK(nc_put_var_double(ncid, v_ftol_array, pv.ftol_array), "put ftol_array");
    NC_CHECK(nc_put_var_double(ncid, v_am, pv.am), "put am");
    NC_CHECK(nc_put_var_double(ncid, v_ac, pv.ac), "put ac");
    NC_CHECK(nc_put_var_double(ncid, v_ai, pv.ai), "put ai");
    NC_CHECK(nc_put_var_double(ncid, v_aphi, pv.aphi), "put aphi");
    NC_CHECK(nc_put_var_double(ncid, v_raxis_c, pv.raxis_c), "put raxis_c");
    NC_CHECK(nc_put_var_double(ncid, v_zaxis_s, pv.zaxis_s), "put zaxis_s");
    NC_CHECK(nc_put_var_double(ncid, v_rbcc, &pv.rbcc[0][0]), "put rbcc");
    NC_CHECK(nc_put_var_double(ncid, v_rbss, &pv.rbss[0][0]), "put rbss");
    NC_CHECK(nc_put_var_double(ncid, v_zbsc, &pv.zbsc[0][0]), "put zbsc");
    NC_CHECK(nc_put_var_double(ncid, v_zbcs, &pv.zbcs[0][0]), "put zbcs");

    // A failed close is not re-closed (the stream is already being torn
    // down); remove the partial temp and report the failure.
    {
        int _rc = nc_close(ncid);
        if (_rc != NC_NOERR) {
            fprintf(stderr, "NetCDF error [nc_close %s]: %s\n", tmp.c_str(),
                    nc_strerror(_rc));
            remove(tmp.c_str());
            return false;
        }
    }
    // Atomic publish: the temp is fully written and closed, so rename it over
    // the target. On failure remove the temp; the target is untouched.
    if (!cumes::io_detail::renamePublish(tmp, path).empty()) {
        fprintf(stderr, "NetCDF error [rename %s -> %s]\n", tmp.c_str(), path);
        return false;
    }
#undef NC_CHECK
    printf("Saved netCDF state to %s\n", path);
    return true;
}

// ---- Explicit instantiation (double + float) ----------------------------
template bool outputSaveNetcdf<double>(const cumes::SpectralStorage<double>&, const DeviceParams<double>&, const cumes::ValidatedProblem&, const SolverResult<double>&, const char*, const char*);
template bool outputSaveNetcdf<float>(const cumes::SpectralStorage<float>&, const DeviceParams<float>&, const cumes::ValidatedProblem&, const SolverResult<float>&, const char*, const char*);
