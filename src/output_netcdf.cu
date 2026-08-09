// output_netcdf.cu — netCDF classic-3 output of the solved state.
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
#include "input.h"
#include "solver.cuh"
#include <netcdf.h>
#include <cstdio>
#include <cstring>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void outputSaveNetcdf(const SpectralState<T>& st, const GridParams<T>& p,
                      const InputParams& ip, const SolverResult<T>& result,
                      const char* path, const char* input_file) {
    // NC_CLOBBER alone gives classic-3 (CDF-1); ncdump -k reports "classic".
    int ncid = -1;
    int rc = nc_create(path, NC_CLOBBER, &ncid);
    if (rc != NC_NOERR) {
        fprintf(stderr, "NetCDF error [nc_create %s]: %s\n", path,
                nc_strerror(rc));
        return;
    }
    // On any later error: close, delete the half-written file, return.
#define NC_CHECK(rc_, tag)                                                   \
    do {                                                                     \
        int _rc = (rc_);                                                     \
        if (_rc != NC_NOERR) {                                               \
            fprintf(stderr, "NetCDF error [%s]: %s\n", tag, nc_strerror(_rc)); \
            nc_close(ncid);                                                  \
            remove(path);                                                    \
            return;                                                          \
        }                                                                    \
    } while (0)

    // ---- dimensions ----
    int dim_ns, dim_mnmax, dim_ngrids, dim_ncoeff, dim_naxis, dim_nbm, dim_nbn;
    NC_CHECK(nc_def_dim(ncid, "ns", (size_t)p.ns, &dim_ns), "def dim ns");
    NC_CHECK(nc_def_dim(ncid, "mnmax", (size_t)p.mnmax, &dim_mnmax), "def dim mnmax");
    NC_CHECK(nc_def_dim(ncid, "ngrids", InputParams::kMaxGrids, &dim_ngrids), "def dim ngrids");
    NC_CHECK(nc_def_dim(ncid, "ncoeff", InputParams::kMaxCoeff, &dim_ncoeff), "def dim ncoeff");
    NC_CHECK(nc_def_dim(ncid, "naxis", 32, &dim_naxis), "def dim naxis");
    NC_CHECK(nc_def_dim(ncid, "nbm", 16, &dim_nbm), "def dim nbm");
    NC_CHECK(nc_def_dim(ncid, "nbn", 16, &dim_nbn), "def dim nbn");

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
        {"n_grids", ip.n_grids},
        {"am_n", ip.am_n}, {"ac_n", ip.ac_n}, {"ai_n", ip.ai_n},
        {"aphi_n", ip.aphi_n}, {"raxis_n", ip.raxis_n},
        {"rbc_n", ip.rbc_n}, {"zbs_n", ip.zbs_n},
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
        {"phiedge", ip.phiedge}, {"pres_scale", ip.pres_scale},
        {"adiabatic_index", ip.adiabatic_index}, {"spres_ped", ip.spres_ped},
        {"bloat", ip.bloat}, {"curtor", ip.curtor}, {"tcon0", ip.tcon0},
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
    const size_t n = (size_t)p.ns * p.mnmax;
    auto* buf = new T[n];
    auto* dbuf = new double[n];
    const size_t nb = n * sizeof(T);
    auto writeFam = [&](const T* d, int varid) -> int {
        checkCuda(cudaMemcpy(buf, d, nb, cudaMemcpyDeviceToHost), "cpy fam");
        for (size_t i = 0; i < n; ++i) { dbuf[i] = (double)buf[i]; }
        for (int m = 0; m < p.mnmax; ++m) {
            const size_t start[2] = {0, (size_t)m};
            const size_t count[2] = {(size_t)p.ns, 1};
            int rc2 = nc_put_vara_double(ncid, varid, start, count,
                                         dbuf + (size_t)m * p.ns);
            if (rc2 != NC_NOERR) { return rc2; }
        }
        return NC_NOERR;
    };
    NC_CHECK(writeFam(st.d_rmncc, v_rmncc), "write rmncc");
    NC_CHECK(writeFam(st.d_zmnsc, v_zmnsc), "write zmnsc");
    NC_CHECK(writeFam(st.d_lmnsc, v_lmnsc), "write lmnsc");
    NC_CHECK(writeFam(st.d_rmnss, v_rmnss), "write rmnss");
    NC_CHECK(writeFam(st.d_zmncs, v_zmncs), "write zmncs");
    NC_CHECK(writeFam(st.d_lmncs, v_lmncs), "write lmncs");
    delete[] dbuf;
    delete[] buf;

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
    NC_CHECK(nc_put_var_int(ncid, v_ns_array, ip.ns_array), "put ns_array");
    NC_CHECK(nc_put_var_int(ncid, v_niter_array, ip.niter_array), "put niter_array");
    NC_CHECK(nc_put_var_double(ncid, v_ftol_array, ip.ftol_array), "put ftol_array");
    NC_CHECK(nc_put_var_double(ncid, v_am, ip.am), "put am");
    NC_CHECK(nc_put_var_double(ncid, v_ac, ip.ac), "put ac");
    NC_CHECK(nc_put_var_double(ncid, v_ai, ip.ai), "put ai");
    NC_CHECK(nc_put_var_double(ncid, v_aphi, ip.aphi), "put aphi");
    NC_CHECK(nc_put_var_double(ncid, v_raxis_c, ip.raxis_c), "put raxis_c");
    NC_CHECK(nc_put_var_double(ncid, v_zaxis_s, ip.zaxis_s), "put zaxis_s");
    NC_CHECK(nc_put_var_double(ncid, v_rbcc, &ip.rbcc[0][0]), "put rbcc");
    NC_CHECK(nc_put_var_double(ncid, v_rbss, &ip.rbss[0][0]), "put rbss");
    NC_CHECK(nc_put_var_double(ncid, v_zbsc, &ip.zbsc[0][0]), "put zbsc");
    NC_CHECK(nc_put_var_double(ncid, v_zbcs, &ip.zbcs[0][0]), "put zbcs");

    NC_CHECK(nc_close(ncid), "nc_close");
#undef NC_CHECK
    printf("Saved netCDF state to %s\n", path);
}

// ---- Explicit instantiation (double + float) ----------------------------
template void outputSaveNetcdf<double>(const SpectralState<double>&, const GridParams<double>&, const InputParams&, const SolverResult<double>&, const char*, const char*);
template void outputSaveNetcdf<float>(const SpectralState<float>&, const GridParams<float>&, const InputParams&, const SolverResult<float>&, const char*, const char*);
