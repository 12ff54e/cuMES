// output_hdf5.cpp — serial HDF5 output of the solved state.
// Compiled only when CUMES_HAVE_HDF5 is defined (CMake, CUMES_USE_HDF5
// option + find_package(HDF5 COMPONENTS C)). Same content as the netCDF
// writer (src/output_netcdf.cpp): the 6 coefficient families (on disk
// double regardless of T) plus grid params, convergence and the full
// InputParams provenance; scalars are root-group attributes, arrays are
// datasets.
//
// Layout note: the in-memory coefficient buffers are mode-major (index
// m*ns + j), while the datasets use dims (ns, mnmax) in C order. A naive
// whole-buffer H5Dwrite would silently transpose, so each mode is written
// as a hyperslab with the contiguous column at dbuf + m*ns.
#include "output.cuh"
#include "input.h"
#include "solver.cuh"
#include <cuda_runtime.h>  // cudaMemcpy (host runtime API)
#include <hdf5.h>
#include <cstdio>
#include <cstdlib>   // getpid
#include <cstring>
#include <string>
#include <unistd.h>   // getpid, rename

#include "cumes/runtime/cuda_status.hpp"

namespace {

// Write one scalar (or fixed-size string) attribute on the root group.
herr_t putAttr(hid_t loc, const char* name, hid_t dtype, const void* val) {
    hid_t sid = H5Screate(H5S_SCALAR);
    if (sid < 0) { return -1; }
    hid_t aid = H5Acreate2(loc, name, dtype, sid, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(sid);
    if (aid < 0) { return -1; }
    herr_t r = H5Awrite(aid, dtype, val);
    H5Aclose(aid);
    return r;
}

}  // namespace

template <typename T>
bool outputSaveHdf5(const SpectralState<T>& st, const GridParams<T>& p,
                    const InputParams& ip, const SolverResult<T>& result,
                    const char* path, const char* input_file) {
    // Atomic publication: create at a same-directory temp path, then rename()
    // over `path` after a successful close, so a reader never sees a
    // half-written file and a failure leaves the target untouched.
    const std::string tmp = std::string(path) + ".tmp." +
                            std::to_string((long)getpid());
    hid_t fid = H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "HDF5 error [H5Fcreate %s]\n", tmp.c_str());
        return false;
    }
    // On any later error: close, delete the half-written file, return false
    // (the caller folds output success into the CLI exit code). A failure OF
    // H5Fclose itself must not be re-closed (double-close UB) — the closing
    // call is handled at the call site, not through this macro.
#define H5_CHECK(expr, tag)                                                   \
    do {                                                                      \
        if ((expr) < 0) {                                                     \
            fprintf(stderr, "HDF5 error [%s]\n", tag);                        \
            H5Fclose(fid);                                                    \
            remove(tmp.c_str());                                              \
            return false;                                                     \
        }                                                                     \
    } while (0)

    // ---- scalar attributes ----
    struct IntAttr { const char* name; int value; };
    const IntAttr ints[] = {
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
    for (const auto& a : ints) {
        H5_CHECK(putAttr(fid, a.name, H5T_NATIVE_INT, &a.value), "int attr");
    }
    struct DblAttr { const char* name; double value; };
    const DblAttr dbls[] = {
        {"delt", (double)p.delt}, {"ftol", (double)p.ftol},
        {"lamscale", (double)p.lamscale},
        {"phiedge", ip.phiedge}, {"pres_scale", ip.pres_scale},
        {"adiabatic_index", ip.adiabatic_index}, {"spres_ped", ip.spres_ped},
        {"bloat", ip.bloat}, {"curtor", ip.curtor}, {"tcon0", ip.tcon0},
        {"fsqr", (double)result.fsqr}, {"fsqz", (double)result.fsqz},
        {"fsql", (double)result.fsql},
    };
    for (const auto& a : dbls) {
        H5_CHECK(putAttr(fid, a.name, H5T_NATIVE_DOUBLE, &a.value), "double attr");
    }
    const char* input = input_file ? input_file : "";
    hid_t s1 = H5Tcopy(H5T_C_S1);
    H5_CHECK(H5Tset_size(s1, strlen(input) + 1), "string type input_file");
    H5_CHECK(putAttr(fid, "input_file", s1, input), "attr input_file");
    H5Tclose(s1);
    const char* prec = (sizeof(T) == sizeof(double)) ? "double" : "float";
    s1 = H5Tcopy(H5T_C_S1);
    H5_CHECK(H5Tset_size(s1, strlen(prec) + 1), "string type precision");
    H5_CHECK(putAttr(fid, "precision", s1, prec), "attr precision");
    H5Tclose(s1);

    // ---- helper: define a 1-D/2-D dataset and write it whole ----
    auto writeArray = [&](const char* name, hid_t dtype, int nd,
                          const hsize_t* dims, const void* data) -> herr_t {
        hid_t sp = H5Screate_simple(nd, dims, nullptr);
        if (sp < 0) { return -1; }
        hid_t ds = H5Dcreate2(fid, name, dtype, sp, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);
        H5Sclose(sp);
        if (ds < 0) { return -1; }
        herr_t r = H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        return r;
    };
    const hsize_t d8[1] = {InputParams::kMaxGrids};
    const hsize_t d16[1] = {InputParams::kMaxCoeff};
    const hsize_t d32[1] = {32};
    const hsize_t d16x16[2] = {16, 16};
    H5_CHECK(writeArray("ns_array", H5T_NATIVE_INT, 1, d8, ip.ns_array), "write ns_array");
    H5_CHECK(writeArray("niter_array", H5T_NATIVE_INT, 1, d8, ip.niter_array), "write niter_array");
    H5_CHECK(writeArray("ftol_array", H5T_NATIVE_DOUBLE, 1, d8, ip.ftol_array), "write ftol_array");
    H5_CHECK(writeArray("am", H5T_NATIVE_DOUBLE, 1, d16, ip.am), "write am");
    H5_CHECK(writeArray("ac", H5T_NATIVE_DOUBLE, 1, d16, ip.ac), "write ac");
    H5_CHECK(writeArray("ai", H5T_NATIVE_DOUBLE, 1, d16, ip.ai), "write ai");
    H5_CHECK(writeArray("aphi", H5T_NATIVE_DOUBLE, 1, d16, ip.aphi), "write aphi");
    H5_CHECK(writeArray("raxis_c", H5T_NATIVE_DOUBLE, 1, d32, ip.raxis_c), "write raxis_c");
    H5_CHECK(writeArray("zaxis_s", H5T_NATIVE_DOUBLE, 1, d32, ip.zaxis_s), "write zaxis_s");
    H5_CHECK(writeArray("rbcc", H5T_NATIVE_DOUBLE, 2, d16x16, &ip.rbcc[0][0]), "write rbcc");
    H5_CHECK(writeArray("rbss", H5T_NATIVE_DOUBLE, 2, d16x16, &ip.rbss[0][0]), "write rbss");
    H5_CHECK(writeArray("zbsc", H5T_NATIVE_DOUBLE, 2, d16x16, &ip.zbsc[0][0]), "write zbsc");
    H5_CHECK(writeArray("zbcs", H5T_NATIVE_DOUBLE, 2, d16x16, &ip.zbcs[0][0]), "write zbcs");

    // ---- state datasets (ns, mnmax), per-mode hyperslab writes ----
    const size_t n = (size_t)p.ns * p.mnmax;
    auto* buf = new T[n];
    auto* dbuf = new double[n];
    const size_t nb = n * sizeof(T);
    const hsize_t state_dims[2] = {(hsize_t)p.ns, (hsize_t)p.mnmax};
    auto writeFam = [&](const T* d, const char* name) -> herr_t {
        cumes::check_cuda(cudaMemcpy(buf, d, nb, cudaMemcpyDeviceToHost), "cpy fam");
        for (size_t i = 0; i < n; ++i) { dbuf[i] = (double)buf[i]; }
        hid_t sp = H5Screate_simple(2, state_dims, nullptr);
        if (sp < 0) { return -1; }
        hid_t ds = H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(sp);
        if (ds < 0) { return -1; }
        for (int m = 0; m < p.mnmax; ++m) {
            hsize_t start[2] = {0, (hsize_t)m};
            hsize_t count[2] = {(hsize_t)p.ns, 1};
            hsize_t mdim[1] = {(hsize_t)p.ns};
            hid_t fs = H5Dget_space(ds);
            if (fs < 0) { H5Dclose(ds); return -1; }
            H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count,
                                nullptr);
            hid_t ms = H5Screate_simple(1, mdim, nullptr);
            herr_t r = H5Dwrite(ds, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT,
                                dbuf + (size_t)m * p.ns);
            H5Sclose(ms);
            H5Sclose(fs);
            if (r < 0) { H5Dclose(ds); return -1; }
        }
        H5Dclose(ds);
        return 0;
    };
    H5_CHECK(writeFam(st.d_rmncc, "rmncc"), "write rmncc");
    H5_CHECK(writeFam(st.d_zmnsc, "zmnsc"), "write zmnsc");
    H5_CHECK(writeFam(st.d_lmnsc, "lmnsc"), "write lmnsc");
    H5_CHECK(writeFam(st.d_rmnss, "rmnss"), "write rmnss");
    H5_CHECK(writeFam(st.d_zmncs, "zmncs"), "write zmncs");
    H5_CHECK(writeFam(st.d_lmncs, "lmncs"), "write lmncs");
    delete[] dbuf;
    delete[] buf;

    // A failed close is not re-closed (the file is already being torn down);
    // remove the partial temp and report the failure.
    if (H5Fclose(fid) < 0) {
        fprintf(stderr, "HDF5 error [H5Fclose %s]\n", tmp.c_str());
        remove(tmp.c_str());
        return false;
    }
    // Atomic publish: the temp is fully written and closed, so rename it over
    // the target. On failure remove the temp; the target is untouched.
    if (rename(tmp.c_str(), path) != 0) {
        fprintf(stderr, "HDF5 error [rename %s -> %s]\n", tmp.c_str(), path);
        remove(tmp.c_str());
        return false;
    }
#undef H5_CHECK
    printf("Saved HDF5 state to %s\n", path);
    return true;
}

// ---- Explicit instantiation (double + float) ----------------------------
template bool outputSaveHdf5<double>(const SpectralState<double>&, const GridParams<double>&, const InputParams&, const SolverResult<double>&, const char*, const char*);
template bool outputSaveHdf5<float>(const SpectralState<float>&, const GridParams<float>&, const InputParams&, const SolverResult<float>&, const char*, const char*);
