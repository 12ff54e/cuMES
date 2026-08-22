// test_io_restart_offsets.cpp — corrupted v1 restart-metadata hardening
// (completion-plan follow-up §2.2).
//
// The NetCDF/HDF5 v1 readers must validate serialized restart_stage_offset
// values (first == 0 when stages exist, nonnegative, monotonic, bounded by
// nrestarts) and the stage/restart array extents BEFORE indexing rst_iter.
// Before this gate, a corrupted file with negative, descending, or oversized
// offsets read out of bounds (or attributed restarts to the wrong stage).
//
// Host-only on purpose: the ASan/UBSan twin (asan_test_io_restart_offsets)
// runs this exact source under the host sanitizers.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

#include "cumes/io/reader.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/io/writer.hpp"
#ifdef CUMES_HAVE_NETCDF
#include <netcdf.h>
#endif
#ifdef CUMES_HAVE_HDF5
#include <hdf5.h>
#endif

// The harness is unconditional: main() uses check/summary/std::cout even in
// the nobackend build (only the backend-specific sections are guarded).
#include "cumes_test.h"
using namespace cumes::test;

using cumes::EquilibriumSnapshot;
using cumes::OutputFormat;
using cumes::Reader;
using cumes::RunReport;


// Per-test temp directory with RAII cleanup (never leaves repo-root debris;
// completion-plan follow-up §5).
class TempDir {
 public:
    TempDir() {
        char tmpl[] = "/tmp/cumes_io_restart_XXXXXX";
        dir_ = mkdtemp(tmpl);
    }
    ~TempDir() {
        if (dir_.empty()) return;
        const std::string cmd = "rm -rf '" + dir_ + "'";
        (void)!system(cmd.c_str());
    }
    const std::string& path() const { return dir_; }
    bool ok() const { return !dir_.empty(); }

 private:
    std::string dir_;
};

// A fixture writer: write a minimal-but-schema-complete v1 container with the
// given restart offsets (and an optional restart_iteration extent mismatch).
using FixtureWriter = bool (*)(const std::string& path,
                               const std::vector<int>& rst_off, int nrestarts,
                               bool wrong_iter_extent);

#if defined(CUMES_HAVE_NETCDF) || defined(CUMES_HAVE_HDF5)

// ---------------------------------------------------------------------------
// Fixture writers. The state values are irrelevant — the reader reaches the
// restart section only after the state read succeeds, which is exactly what
// these cases exercise.
// ---------------------------------------------------------------------------
#ifdef CUMES_HAVE_NETCDF
static bool writeNetcdfFixture(const std::string& path,
                               const std::vector<int>& rst_off, int nrestarts,
                               bool wrong_iter_extent) {
    int ncid = -1;
    if (nc_create(path.c_str(), NC_CLOBBER, &ncid) != NC_NOERR) return false;
#define NC_F(expr)                                                            \
    do {                                                                      \
        if ((expr) != NC_NOERR) return false;                                 \
    } while (0)
    const bool ok = [&]() -> bool {
    const int nstages = static_cast<int>(rst_off.size());
    int d_ns, d_mn, d_st, d_rs;
    NC_F(nc_def_dim(ncid, "ns", 3, &d_ns));
    NC_F(nc_def_dim(ncid, "mnmax", 2, &d_mn));
    NC_F(nc_def_dim(ncid, "nstages", nstages, &d_st));
    NC_F(nc_def_dim(ncid, "nrestarts", nrestarts, &d_rs));
    const int fam_dims[2] = {d_ns, d_mn};
    const char* fams[6] = {"rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs"};
    int v_fam[6];
    for (int c = 0; c < 6; ++c) {
        NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 2, fam_dims, &v_fam[c]));
    }
    int v_prec, v_status, v_total, v_dirty;
    NC_F(nc_def_var(ncid, "precision", NC_INT, 0, nullptr, &v_prec));
    NC_F(nc_def_var(ncid, "status", NC_INT, 0, nullptr, &v_status));
    NC_F(nc_def_var(ncid, "total_iterations", NC_INT, 0, nullptr, &v_total));
    NC_F(nc_def_var(ncid, "build_dirty", NC_INT, 0, nullptr, &v_dirty));
    int v_ns, v_iter, v_conv, v_fsqr, v_fsqz, v_fsql, v_off, v_riter;
    NC_F(nc_def_var(ncid, "stage_ns", NC_INT, 1, &d_st, &v_ns));
    NC_F(nc_def_var(ncid, "stage_iterations", NC_INT, 1, &d_st, &v_iter));
    NC_F(nc_def_var(ncid, "stage_converged", NC_INT, 1, &d_st, &v_conv));
    NC_F(nc_def_var(ncid, "stage_fsqr", NC_DOUBLE, 1, &d_st, &v_fsqr));
    NC_F(nc_def_var(ncid, "stage_fsqz", NC_DOUBLE, 1, &d_st, &v_fsqz));
    NC_F(nc_def_var(ncid, "stage_fsql", NC_DOUBLE, 1, &d_st, &v_fsql));
    NC_F(nc_def_var(ncid, "restart_stage_offset", NC_INT, 1, &d_st, &v_off));
    // Wrong-extent case: declare restart_iteration against nstages instead of
    // nrestarts — the extent check must reject it before any indexing.
    NC_F(nc_def_var(ncid, "restart_iteration", NC_INT, 1,
                    wrong_iter_extent ? &d_st : &d_rs, &v_riter));
    const char* str_attrs[][2] = {
        {"revision", "r1"},
        {"build_type", "Release"},
        {"precision_policy", "verify-double"},
        {"compile_flags", ""},   // zero-length: exercises the reader's guard
        {"source_path", "in.json"},
        {"source_hash", "h"},
        {"gpu_name", "g"},
        {"driver", "d"},
        {"runtime", "rt"},
        {"toolkit", "t"}};
    for (const auto& a : str_attrs) {
        NC_F(nc_put_att_text(ncid, NC_GLOBAL, a[0], strlen(a[1]), a[1]));
    }
    NC_F(nc_enddef(ncid));
    const double fambuf[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int c = 0; c < 6; ++c) {
        NC_F(nc_put_var_double(ncid, v_fam[c], fambuf));
    }
    const int zero = 0;
    NC_F(nc_put_var_int(ncid, v_prec, &zero));
    NC_F(nc_put_var_int(ncid, v_status, &zero));
    NC_F(nc_put_var_int(ncid, v_total, &zero));
    NC_F(nc_put_var_int(ncid, v_dirty, &zero));
    std::vector<int> stage_ns((size_t)nstages, 3);
    std::vector<int> stage_iter((size_t)nstages, 1);
    std::vector<int> stage_conv((size_t)nstages, 1);
    std::vector<double> stage_res((size_t)nstages, 0.0);
    std::vector<int> rst_iter((size_t)(wrong_iter_extent ? nstages : nrestarts),
                              7);
    NC_F(nc_put_var_int(ncid, v_ns, stage_ns.data()));
    NC_F(nc_put_var_int(ncid, v_iter, stage_iter.data()));
    NC_F(nc_put_var_int(ncid, v_conv, stage_conv.data()));
    NC_F(nc_put_var_double(ncid, v_fsqr, stage_res.data()));
    NC_F(nc_put_var_double(ncid, v_fsqz, stage_res.data()));
    NC_F(nc_put_var_double(ncid, v_fsql, stage_res.data()));
    NC_F(nc_put_var_int(ncid, v_off, rst_off.data()));
    NC_F(nc_put_var_int(ncid, v_riter, rst_iter.data()));
        return true;
    }();
    if (!ok) {
        nc_abort(ncid);
        return false;
    }
    const bool closed = nc_close(ncid) == NC_NOERR;
#undef NC_F
    return closed;
}
#endif  // CUMES_HAVE_NETCDF

#ifdef CUMES_HAVE_HDF5
static bool writeHdf5Fixture(const std::string& path,
                             const std::vector<int>& rst_off, int nrestarts,
                             bool wrong_iter_extent) {
    hid_t fid = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) return false;
#define H5_F(expr)                                                            \
    do {                                                                      \
        if ((expr) < 0) return false;                                         \
    } while (0)
    const bool ok = [&]() -> bool {
    auto putIntAttr = [&](const char* name, int value) -> herr_t {
        hid_t sid = H5Screate(H5S_SCALAR);
        if (sid < 0) return -1;
        hid_t aid = H5Acreate2(fid, name, H5T_NATIVE_INT, sid, H5P_DEFAULT,
                               H5P_DEFAULT);
        H5Sclose(sid);
        if (aid < 0) return -1;
        const herr_t r = H5Awrite(aid, H5T_NATIVE_INT, &value);
        H5Aclose(aid);
        return r;
    };
    auto putStrAttr = [&](const char* name, const std::string& value) -> herr_t {
        hid_t s1 = H5Tcopy(H5T_C_S1);
        if (s1 < 0) return -1;
        herr_t r0 = H5Tset_size(s1, value.size() + 1);
        if (r0 < 0) { H5Tclose(s1); return -1; }
        hid_t sid = H5Screate(H5S_SCALAR);
        if (sid < 0) { H5Tclose(s1); return -1; }
        hid_t aid = H5Acreate2(fid, name, s1, sid, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(sid);
        if (aid < 0) { H5Tclose(s1); return -1; }
        const herr_t r = H5Awrite(aid, s1, value.c_str());
        H5Aclose(aid);
        H5Tclose(s1);
        return r;
    };
    auto putArray = [&](const char* name, hid_t dtype, hsize_t len,
                        const void* data) -> herr_t {
        hid_t sp = H5Screate_simple(1, &len, nullptr);
        if (sp < 0) return -1;
        hid_t ds = H5Dcreate2(fid, name, dtype, sp, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);
        H5Sclose(sp);
        if (ds < 0) return -1;
        const herr_t r = H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                  data);
        H5Dclose(ds);
        return r;
    };

    const int nstages = static_cast<int>(rst_off.size());
    const hsize_t state_dims[2] = {3, 2};
    const char* fams[6] = {"rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs"};
    const double fambuf[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int c = 0; c < 6; ++c) {
        hid_t sp = H5Screate_simple(2, state_dims, nullptr);
        if (sp < 0) return false;
        hid_t ds = H5Dcreate2(fid, fams[c], H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(sp);
        if (ds < 0 || H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, fambuf) < 0) {
            if (ds >= 0) H5Dclose(ds);
            return false;
        }
        H5Dclose(ds);
    }
    H5_F(putIntAttr("precision", 0));
    H5_F(putIntAttr("status", 0));
    H5_F(putIntAttr("total_iterations", 0));
    H5_F(putIntAttr("build_dirty", 0));
    H5_F(putStrAttr("revision", "r1"));
    H5_F(putStrAttr("build_type", "Release"));
    H5_F(putStrAttr("precision_policy", "verify-double"));
    H5_F(putStrAttr("compile_flags", ""));
    H5_F(putStrAttr("source_path", "in.json"));
    H5_F(putStrAttr("source_hash", "h"));
    H5_F(putStrAttr("gpu_name", "g"));
    H5_F(putStrAttr("driver", "d"));
    H5_F(putStrAttr("runtime", "rt"));
    H5_F(putStrAttr("toolkit", "t"));
    std::vector<int> stage_ns((size_t)nstages, 3);
    // Wrong-extent case: unlike NetCDF (which stores nrestarts as a named
    // dimension), the HDF5 reader derives the stage/restart extents FROM the
    // datasets, so the mismatch is manufactured on a stage array instead.
    std::vector<int> stage_iter((size_t)(wrong_iter_extent ? nstages + 1
                                                           : nstages),
                                1);
    std::vector<int> stage_conv((size_t)nstages, 1);
    std::vector<double> stage_res((size_t)nstages, 0.0);
    std::vector<int> rst_iter((size_t)nrestarts, 7);
    H5_F(putArray("stage_ns", H5T_NATIVE_INT, (hsize_t)nstages,
                  stage_ns.data()));
    H5_F(putArray("stage_iterations", H5T_NATIVE_INT,
                  (hsize_t)stage_iter.size(), stage_iter.data()));
    H5_F(putArray("stage_converged", H5T_NATIVE_INT, (hsize_t)nstages,
                  stage_conv.data()));
    H5_F(putArray("stage_fsqr", H5T_NATIVE_DOUBLE, (hsize_t)nstages,
                  stage_res.data()));
    H5_F(putArray("stage_fsqz", H5T_NATIVE_DOUBLE, (hsize_t)nstages,
                  stage_res.data()));
    H5_F(putArray("stage_fsql", H5T_NATIVE_DOUBLE, (hsize_t)nstages,
                  stage_res.data()));
    H5_F(putArray("restart_stage_offset", H5T_NATIVE_INT, (hsize_t)nstages,
                  rst_off.data()));
    H5_F(putArray("restart_iteration", H5T_NATIVE_INT, (hsize_t)nrestarts,
                  rst_iter.data()));
        return true;
    }();
    if (!ok) {
        H5Fclose(fid);
        return false;
    }
    const bool closed = H5Fclose(fid) >= 0;
#undef H5_F
    return closed;
}
#endif  // CUMES_HAVE_HDF5

// ---------------------------------------------------------------------------
// One backend: the reader must round-trip valid offsets and fail cleanly
// (typed error, no crash/OOB) on every corrupted shape.
// ---------------------------------------------------------------------------
static void testBackend(const char* name, FixtureWriter writer,
                        OutputFormat fmt, const std::string& suffix,
                        const TempDir& dir) {
    auto runCase = [&](const char* label, const std::vector<int>& offsets,
                       int nrestarts, bool expect_ok,
                       bool wrong_iter_extent = false) {
        const std::string path = dir.path() + "/" + label + suffix;
        const bool written = writer(path, offsets, nrestarts, wrong_iter_extent);
        check(written, format("{}: fixture written ({})", name, label));

        std::unique_ptr<Reader> reader = make_reader(fmt);
        check(reader != nullptr, format("{}: reader factory exists ({})", name, label));

        RunReport rep;
        const auto res = reader->read(path, &rep);
        remove(path.c_str());
        if (expect_ok) {
            check(res.has_value(), format("{}: {}", name, label));
            if (res.has_value()) {
                bool restarts_ok = rep.stages.size() == offsets.size();
                for (size_t g = 0; restarts_ok && g < offsets.size(); ++g) {
                    const size_t begin = (size_t)offsets[g];
                    const size_t end = (g + 1 < offsets.size())
                                           ? (size_t)offsets[g + 1]
                                           : (size_t)nrestarts;
                    restarts_ok = rep.stages[g].restarts.size() == end - begin;
                }
                check(restarts_ok,
                      format("{}: {} (restarts land in the right stages)", name,
                             label));
            }
        } else if (!res.has_value()) {
            const std::string& err = res.error();
            // The NetCDF extent case fails in the restart-history read, the
            // HDF5 one in the stage-history read — either names the contract.
            check(err.find("restart") != std::string::npos ||
                      err.find("stage") != std::string::npos,
                  format("{}: {} failure names the restart contract", name,
                         label));
        } else {
            check(false, format("{}: {} rejected with typed failure", name, label));
        }
    };

    // Valid: two stages share three restarts via offsets {0, 2}.
    runCase("valid", {0, 2}, 3, /*expect_ok=*/true);
    // Corrupted shapes (completion-plan follow-up §2.2).
    runCase("negative", {0, -1}, 2, false);
    runCase("descending", {0, 2, 1}, 3, false);
    runCase("oversized", {0, 5}, 3, false);
    runCase("nonzero_first", {1, 2}, 2, false);
    runCase("extent_mismatch", {0, 1}, 1, false, /*wrong_iter_extent=*/true);
}
#endif  // CUMES_HAVE_NETCDF || CUMES_HAVE_HDF5

int main() {
    TempDir dir;
    check(dir.ok(), "temp directory created");
    if (!dir.ok()) return 1;

#ifdef CUMES_HAVE_NETCDF
    testBackend("netcdf", writeNetcdfFixture, OutputFormat::kNetCdf, ".nc",
                dir);
#else
    std::cout << "SKIP netcdf corrupted-offset cases (backend not compiled)\n";
#endif
#ifdef CUMES_HAVE_HDF5
    testBackend("hdf5", writeHdf5Fixture, OutputFormat::kHdf5, ".h5", dir);
#else
    std::cout << "SKIP hdf5 corrupted-offset cases (backend not compiled)\n";
#endif

    return summary();
}
