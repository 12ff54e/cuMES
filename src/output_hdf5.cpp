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
#include "cumes/io/legacy_provenance.hpp"
#define CUMES_IO_DEVICE_STAGE 1  // opt in to FamilyStage (needs CUDA headers)
#include "cumes/io/writer_helpers.hpp"  // io_detail::tempPathFor/renamePublish, FamilyStage
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
bool outputSaveHdf5(const cumes::SpectralStorage<T>& storage, const DeviceParams<T>& p,
                    const cumes::ValidatedProblem& vp, const SolverResult<T>& result,
                    const char* path, const char* input_file) {
    // Fixed-capacity provenance for the v0 layout (padded arrays, byte-identical
    // to the pre-overhaul InputParams).
    const cumes::LegacyInputProvenance pv =
        cumes::LegacyInputProvenance::from_validated(vp);
    // Atomic publication: create at a same-directory temp path, then rename()
    // over `path` after a successful close, so a reader never sees a
    // half-written file and a failure leaves the target untouched.
    const std::string tmp = cumes::io_detail::tempPathFor(path);
    hid_t fid = H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "HDF5 error [H5Fcreate %s]\n", tmp.c_str());
        return false;
    }
    // On any later error: close, delete the half-written file, return false
    // (the caller folds output success into the CLI exit code). A failure OF
    // H5Fclose itself must not be re-closed (double-close UB) — the closing
    // call is handled at the call site, not through this macro. The same fail
    // path also serves the device-copy failure (the D2H copy reports instead
    // of throwing, so a GPU fault still removes the temp and returns false).
    auto fail = [&](const char* tag, const std::string& msg) -> bool {
        if (msg.empty()) {
            fprintf(stderr, "HDF5 error [%s]\n", tag);
        } else {
            fprintf(stderr, "HDF5 error [%s]: %s\n", tag, msg.c_str());
        }
        H5Fclose(fid);
        remove(tmp.c_str());
        return false;
    };
#define H5_CHECK(expr, tag)                                                   \
    do {                                                                      \
        if ((expr) < 0) return fail(tag, "");                                 \
    } while (0)

    // ---- scalar attributes ----
    struct IntAttr { const char* name; int value; };
    const IntAttr ints[] = {
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
    for (const auto& a : ints) {
        H5_CHECK(putAttr(fid, a.name, H5T_NATIVE_INT, &a.value), "int attr");
    }
    struct DblAttr { const char* name; double value; };
    const DblAttr dbls[] = {
        {"delt", (double)p.delt}, {"ftol", (double)p.ftol},
        {"lamscale", (double)p.lamscale},
        {"phiedge", pv.phiedge}, {"pres_scale", pv.pres_scale},
        {"adiabatic_index", pv.adiabatic_index}, {"spres_ped", pv.spres_ped},
        {"bloat", pv.bloat}, {"curtor", pv.curtor}, {"tcon0", pv.tcon0},
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
    const hsize_t d8[1] = {cumes::LegacyInputProvenance::kMaxGrids};
    const hsize_t d16[1] = {cumes::LegacyInputProvenance::kMaxCoeff};
    const hsize_t d32[1] = {cumes::LegacyInputProvenance::kMaxAxis};
    const hsize_t d16x16[2] = {cumes::LegacyInputProvenance::kMaxM,
                               cumes::LegacyInputProvenance::kMaxN};
    H5_CHECK(writeArray("ns_array", H5T_NATIVE_INT, 1, d8, pv.ns_array), "write ns_array");
    H5_CHECK(writeArray("niter_array", H5T_NATIVE_INT, 1, d8, pv.niter_array), "write niter_array");
    H5_CHECK(writeArray("ftol_array", H5T_NATIVE_DOUBLE, 1, d8, pv.ftol_array), "write ftol_array");
    H5_CHECK(writeArray("am", H5T_NATIVE_DOUBLE, 1, d16, pv.am), "write am");
    H5_CHECK(writeArray("ac", H5T_NATIVE_DOUBLE, 1, d16, pv.ac), "write ac");
    H5_CHECK(writeArray("ai", H5T_NATIVE_DOUBLE, 1, d16, pv.ai), "write ai");
    H5_CHECK(writeArray("aphi", H5T_NATIVE_DOUBLE, 1, d16, pv.aphi), "write aphi");
    H5_CHECK(writeArray("raxis_c", H5T_NATIVE_DOUBLE, 1, d32, pv.raxis_c), "write raxis_c");
    H5_CHECK(writeArray("zaxis_s", H5T_NATIVE_DOUBLE, 1, d32, pv.zaxis_s), "write zaxis_s");
    H5_CHECK(writeArray("rbcc", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.rbcc[0][0]), "write rbcc");
    H5_CHECK(writeArray("rbss", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.rbss[0][0]), "write rbss");
    H5_CHECK(writeArray("zbsc", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.zbsc[0][0]), "write zbsc");
    H5_CHECK(writeArray("zbcs", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.zbcs[0][0]), "write zbcs");

    // ---- state datasets (ns, mnmax), per-mode hyperslab writes ----
    const auto n_opt = cumes::io_detail::familyCount(p.ns, p.mnmax);
    if (!n_opt) return fail("write state", "ns * mnmax overflows size_t");
    const std::size_t n = *n_opt;
    // RAII staging (FamilyStage): the buffers cannot leak on a write failure,
    // and a CUDA copy error is reported (fail path) instead of thrown.
    cumes::io_detail::FamilyStage<T> stage(n);
    const hsize_t state_dims[2] = {(hsize_t)p.ns, (hsize_t)p.mnmax};
    auto writeFam = [&](const T* d, const char* name) -> bool {
        std::string reason;
        if (!stage.copy(d, name, reason)) return fail(name, reason);
        const double* dbuf = stage.data();
        hid_t sp = H5Screate_simple(2, state_dims, nullptr);
        if (sp < 0) return fail(name, "H5Screate_simple failed");
        hid_t ds = H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(sp);
        if (ds < 0) return fail(name, "H5Dcreate2 failed");
        // The file dataspace is invariant across modes: fetch it once before
        // the per-mode loop instead of per mode.
        hid_t fs = H5Dget_space(ds);
        if (fs < 0) {
            H5Dclose(ds);
            return fail(name, "H5Dget_space failed");
        }
        const hsize_t mdim[1] = {(hsize_t)p.ns};
        for (int m = 0; m < p.mnmax; ++m) {
            hsize_t start[2] = {0, (hsize_t)m};
            hsize_t count[2] = {(hsize_t)p.ns, 1};
            if (H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count,
                                    nullptr) < 0) {
                H5Sclose(fs);
                H5Dclose(ds);
                return fail(name, "H5Sselect_hyperslab failed");
            }
            hid_t ms = H5Screate_simple(1, mdim, nullptr);
            herr_t r = H5Dwrite(ds, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT,
                                dbuf + (size_t)m * p.ns);
            H5Sclose(ms);
            if (r < 0) {
                H5Sclose(fs);
                H5Dclose(ds);
                return fail(name, "H5Dwrite failed");
            }
        }
        H5Sclose(fs);
        H5Dclose(ds);
        return true;
    };
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Rcc), "rmncc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Zsc), "zmnsc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Lsc), "lmnsc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Rss), "rmnss")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Zcs), "zmncs")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Lcs), "lmncs")) return false;

    // A failed close is not re-closed (the file is already being torn down);
    // remove the partial temp and report the failure.
    if (H5Fclose(fid) < 0) {
        fprintf(stderr, "HDF5 error [H5Fclose %s]\n", tmp.c_str());
        remove(tmp.c_str());
        return false;
    }
    // Atomic publish: the temp is fully written and closed, so rename it over
    // the target. On failure remove the temp; the target is untouched.
    if (!cumes::io_detail::renamePublish(tmp, path).empty()) {
        fprintf(stderr, "HDF5 error [rename %s -> %s]\n", tmp.c_str(), path);
        return false;
    }
#undef H5_CHECK
    printf("Saved HDF5 state to %s\n", path);
    return true;
}

// ---- Explicit instantiation (double + float) ----------------------------
template bool outputSaveHdf5<double>(const cumes::SpectralStorage<double>&, const DeviceParams<double>&, const cumes::ValidatedProblem&, const SolverResult<double>&, const char*, const char*);
template bool outputSaveHdf5<float>(const cumes::SpectralStorage<float>&, const DeviceParams<float>&, const cumes::ValidatedProblem&, const SolverResult<float>&, const char*, const char*);
