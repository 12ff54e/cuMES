// hdf5_writer.cpp — host-only HDF5 state adapters (completion plan steps
// 2.2/2.3). Mirrors src/cumes/io/netcdf_writer.cpp content-for-content:
//
//   - legacy-v0: byte/layout-exact replica of the pre-overhaul device-reading
//     writer (src/output_hdf5.cpp, deleted): padded fixed capacities,
//     [surface, mode] logical mapping, final-stage scalar pack, legacy
//     provenance — written purely from the host snapshot.
//   - v1: active dimensions + complete provenance (precision/status/total
//     iterations, build/input/runtime strings, source hash, per-stage
//     outcomes, full restart history, raw + folded boundary harmonics) with
//     a round-tripping reader.
//
// This TU (and only this TU) includes <hdf5.h>; it compiles under
// CUMES_HAVE_HDF5, which is confined to the adapter library target.
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "cumes/io/legacy_provenance.hpp"
#include "internal_factories.hpp"
#include "io_common.hpp"
#include "cumes/io/writer_helpers.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cumes {
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

herr_t putStrAttr(hid_t loc, const char* name, const std::string& value) {
    hid_t s1 = H5Tcopy(H5T_C_S1);
    if (s1 < 0) return -1;
    herr_t r0 = H5Tset_size(s1, value.size() + 1);
    herr_t r1 = r0 >= 0 ? putAttr(loc, name, s1, value.c_str()) : -1;
    H5Tclose(s1);
    return r1;
}

// Read a fixed-size string attribute back into a std::string.
bool getStrAttr(hid_t loc, const char* name, std::string& out) {
    hid_t aid = H5Aopen(loc, name, H5P_DEFAULT);
    if (aid < 0) return false;
    hid_t ty = H5Aget_type(aid);
    const size_t size = H5Tget_size(ty);
    out.resize(size + 1, '\0');
    const herr_t r = H5Aread(aid, ty, &out[0]);
    H5Tclose(ty);
    H5Aclose(aid);
    if (r < 0) return false;
    out.resize(size);  // drop the stored NUL; keep embedded NULs if any
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return true;
}

// ---------------------------------------------------------------------------
// legacy-v0: byte/layout-exact replica of the deleted device-reading writer
// ---------------------------------------------------------------------------
class Hdf5V0Writer final : public Writer {
 public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report, const OutputSpec& spec,
                        const ValidatedProblem& problem,
                        const LegacyRunScalars& s) override {
        const LegacyInputProvenance pv =
            LegacyInputProvenance::from_validated(problem);
        const std::string tmp = io_detail::tempPathFor(spec.path);
        hid_t fid = H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                              H5P_DEFAULT);
        if (fid < 0) return Status("HDF5 H5Fcreate failed for " + tmp);
        auto fail = [&](const std::string& msg) -> Status {
            H5Fclose(fid);
            remove(tmp.c_str());
            return Status("HDF5: " + msg);
        };
#define H5_CHECK(expr, msg_)                                                   \
    do {                                                                       \
        if ((expr) < 0) return fail(std::string(msg_));                        \
    } while (0)

        // ---- scalar attributes (same order as the legacy writer) ----
        struct IntAttr { const char* name; int value; };
        const IntAttr ints[] = {
            {"mpol", s.mpol},       {"ntor", s.ntor},       {"nfp", s.nfp},
            {"ntheta", s.ntheta},   {"nzeta", s.nzeta},     {"ns", s.ns},
            {"mnmax", s.mnmax},     {"nZnT", s.nZnT},       {"ncurr", s.ncurr},
            {"max_iter", s.max_iter}, {"n_grids", pv.n_grids},
            {"am_n", pv.am_n},      {"ac_n", pv.ac_n},      {"ai_n", pv.ai_n},
            {"aphi_n", pv.aphi_n},  {"raxis_n", pv.raxis_n},
            {"rbc_n", pv.rbc_n},    {"zbs_n", pv.zbs_n},
            {"iterations", s.iterations},
            {"converged", s.converged ? 1 : 0},
        };
        for (const auto& a : ints) {
            H5_CHECK(putAttr(fid, a.name, H5T_NATIVE_INT, &a.value),
                     "int attr");
        }
        struct DblAttr { const char* name; double value; };
        const DblAttr dbls[] = {
            {"delt", s.delt}, {"ftol", s.ftol}, {"lamscale", s.lamscale},
            {"phiedge", pv.phiedge}, {"pres_scale", pv.pres_scale},
            {"adiabatic_index", pv.adiabatic_index}, {"spres_ped", pv.spres_ped},
            {"bloat", pv.bloat}, {"curtor", pv.curtor}, {"tcon0", pv.tcon0},
            {"fsqr", s.fsqr}, {"fsqz", s.fsqz}, {"fsql", s.fsql},
        };
        for (const auto& a : dbls) {
            H5_CHECK(putAttr(fid, a.name, H5T_NATIVE_DOUBLE, &a.value),
                     "double attr");
        }
        H5_CHECK(putStrAttr(fid, "input_file", report.input.source_path),
                 "attr input_file");
        H5_CHECK(putStrAttr(fid, "precision", report.build.scalar_type),
                 "attr precision");

        // ---- provenance arrays (fixed v0 capacities) ----
        auto writeArray = [&](const char* name, hid_t dtype, int nd,
                              const hsize_t* dims, const void* data) -> herr_t {
            hid_t sp = H5Screate_simple(nd, dims, nullptr);
            if (sp < 0) { return -1; }
            hid_t ds = H5Dcreate2(fid, name, dtype, sp, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) { return -1; }
            herr_t r = H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            H5Dclose(ds);
            return r;
        };
        const hsize_t d8[1] = {LegacyInputProvenance::kMaxGrids};
        const hsize_t d16[1] = {LegacyInputProvenance::kMaxCoeff};
        const hsize_t d32[1] = {LegacyInputProvenance::kMaxAxis};
        const hsize_t d16x16[2] = {LegacyInputProvenance::kMaxM,
                                   LegacyInputProvenance::kMaxN};
        H5_CHECK(writeArray("ns_array", H5T_NATIVE_INT, 1, d8, pv.ns_array),
                 "write ns_array");
        H5_CHECK(writeArray("niter_array", H5T_NATIVE_INT, 1, d8,
                            pv.niter_array), "write niter_array");
        H5_CHECK(writeArray("ftol_array", H5T_NATIVE_DOUBLE, 1, d8,
                            pv.ftol_array), "write ftol_array");
        H5_CHECK(writeArray("am", H5T_NATIVE_DOUBLE, 1, d16, pv.am), "write am");
        H5_CHECK(writeArray("ac", H5T_NATIVE_DOUBLE, 1, d16, pv.ac), "write ac");
        H5_CHECK(writeArray("ai", H5T_NATIVE_DOUBLE, 1, d16, pv.ai), "write ai");
        H5_CHECK(writeArray("aphi", H5T_NATIVE_DOUBLE, 1, d16, pv.aphi),
                 "write aphi");
        H5_CHECK(writeArray("raxis_c", H5T_NATIVE_DOUBLE, 1, d32, pv.raxis_c),
                 "write raxis_c");
        H5_CHECK(writeArray("zaxis_s", H5T_NATIVE_DOUBLE, 1, d32, pv.zaxis_s),
                 "write zaxis_s");
        H5_CHECK(writeArray("rbcc", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.rbcc[0][0]),
                 "write rbcc");
        H5_CHECK(writeArray("rbss", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.rbss[0][0]),
                 "write rbss");
        H5_CHECK(writeArray("zbsc", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.zbsc[0][0]),
                 "write zbsc");
        H5_CHECK(writeArray("zbcs", H5T_NATIVE_DOUBLE, 2, d16x16, &pv.zbcs[0][0]),
                 "write zbcs");

        // ---- state datasets (ns, mnmax), per-mode hyperslab writes ----
        if (snapshot.ns != s.ns || snapshot.mnmax != s.mnmax) {
            return fail("snapshot dimensions do not match the scalar pack");
        }
        const hsize_t state_dims[2] = {(hsize_t)s.ns, (hsize_t)s.mnmax};
        auto writeFam = [&](EquilibriumSnapshot::Component comp,
                            const char* name) -> Status {
            const std::vector<double>& dbuf = snapshot.component(comp);
            if (dbuf.size() != snapshot.family_size()) {
                return fail(std::string(name) + ": family size mismatch");
            }
            hid_t sp = H5Screate_simple(2, state_dims, nullptr);
            if (sp < 0) return fail(std::string(name) + ": H5Screate_simple");
            hid_t ds = H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return fail(std::string(name) + ": H5Dcreate2");
            hid_t fs = H5Dget_space(ds);
            if (fs < 0) {
                H5Dclose(ds);
                return fail(std::string(name) + ": H5Dget_space");
            }
            const hsize_t mdim[1] = {(hsize_t)s.ns};
            for (int m = 0; m < s.mnmax; ++m) {
                hsize_t start[2] = {0, (hsize_t)m};
                hsize_t count[2] = {(hsize_t)s.ns, 1};
                if (H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr,
                                        count, nullptr) < 0) {
                    H5Sclose(fs);
                    H5Dclose(ds);
                    return fail(std::string(name) + ": H5Sselect_hyperslab");
                }
                hid_t ms = H5Screate_simple(1, mdim, nullptr);
                herr_t r = H5Dwrite(ds, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT,
                                    dbuf.data() + (size_t)m * s.ns);
                H5Sclose(ms);
                if (r < 0) {
                    H5Sclose(fs);
                    H5Dclose(ds);
                    return fail(std::string(name) + ": H5Dwrite");
                }
            }
            H5Sclose(fs);
            H5Dclose(ds);
            return Status();
        };
        Status st = writeFam(EquilibriumSnapshot::kRmncc, "rmncc");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kZmnsc, "zmnsc");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kLmnsc, "lmnsc");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kRmnss, "rmnss");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kZmncs, "zmncs");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kLmncs, "lmncs");
        if (!st.has_value()) return st;

        if (H5Fclose(fid) < 0) {
            remove(tmp.c_str());
            return Status("HDF5 H5Fclose failed");
        }
#undef H5_CHECK
        // Durable publication (completion-plan follow-up §3): the library
        // owns the descriptor, so publishLibraryFile reopens the completed
        // temp, checks fsync + close, then renames + directory-fsyncs.
        const std::string err = io_detail::publishLibraryFile(tmp, spec.path);
        if (!err.empty()) return Status("HDF5 publish: " + err);
        return Status();
    }
};

// ---------------------------------------------------------------------------
// v1: active dimensions + complete provenance + boundary harmonics
// ---------------------------------------------------------------------------
class Hdf5V1Writer final : public Writer {
 public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report, const OutputSpec& spec,
                        const ValidatedProblem& problem,
                        const LegacyRunScalars& s) override {
        (void)s;  // v1 records the RunReport, not the legacy scalar pack
        const FoldedBoundary& fb = problem.boundary();
        const int mpol = problem.shape().mpol;
        const int ntorp1 = problem.shape().ntor + 1;
        const std::vector<BoundaryHarmonic>& rbc = problem.spec().rbc;
        const std::vector<BoundaryHarmonic>& zbs = problem.spec().zbs;
        const size_t nstages = report.stages.size();
        size_t nrestarts = 0;
        for (const auto& st : report.stages) nrestarts += st.restarts.size();

        const std::string tmp = io_detail::tempPathFor(spec.path);
        hid_t fid = H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                              H5P_DEFAULT);
        if (fid < 0) return Status("HDF5 H5Fcreate failed for " + tmp);
        auto fail = [&](const std::string& msg) -> Status {
            H5Fclose(fid);
            remove(tmp.c_str());
            return Status("HDF5: " + msg);
        };
#define H5_CHECK(expr, msg_)                                                   \
    do {                                                                       \
        if ((expr) < 0) return fail(std::string(msg_));                        \
    } while (0)

        auto writeArray = [&](const char* name, hid_t dtype, int nd,
                              const hsize_t* dims, const void* data) -> herr_t {
            hid_t sp = H5Screate_simple(nd, dims, nullptr);
            if (sp < 0) { return -1; }
            hid_t ds = H5Dcreate2(fid, name, dtype, sp, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) { return -1; }
            herr_t r = H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            H5Dclose(ds);
            return r;
        };

        // ---- state datasets ----
        const hsize_t state_dims[2] = {(hsize_t)snapshot.ns,
                                       (hsize_t)snapshot.mnmax};
        const char* fam_names[6] = {"rmncc", "zmnsc", "lmnsc",
                                    "rmnss", "zmncs", "lmncs"};
        for (int c = 0; c < 6; ++c) {
            const std::vector<double>& dbuf =
                snapshot.component(static_cast<EquilibriumSnapshot::Component>(c));
            if (dbuf.size() != snapshot.family_size()) {
                return fail("family size mismatch");
            }
            hid_t sp = H5Screate_simple(2, state_dims, nullptr);
            if (sp < 0) return fail("H5Screate_simple");
            hid_t ds = H5Dcreate2(fid, fam_names[c], H5T_NATIVE_DOUBLE, sp,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return fail("H5Dcreate2");
            hid_t fs = H5Dget_space(ds);
            const hsize_t mdim[1] = {(hsize_t)snapshot.ns};
            for (int m = 0; m < snapshot.mnmax; ++m) {
                hsize_t start[2] = {0, (hsize_t)m};
                hsize_t count[2] = {(hsize_t)snapshot.ns, 1};
                H5_CHECK(H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr,
                                             count, nullptr),
                         "H5Sselect_hyperslab");
                hid_t ms = H5Screate_simple(1, mdim, nullptr);
                herr_t r = H5Dwrite(ds, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT,
                                    dbuf.data() + (size_t)m * snapshot.ns);
                H5Sclose(ms);
                if (r < 0) {
                    H5Sclose(fs);
                    H5Dclose(ds);
                    return fail("H5Dwrite family");
                }
            }
            H5Sclose(fs);
            H5Dclose(ds);
        }

        // ---- run outcome + provenance attributes ----
        const int precision = (report.build.scalar_type == "double") ? 0 : 1;
        const int status = static_cast<int>(report.status);
        const int dirty = report.build.dirty ? 1 : 0;
        H5_CHECK(putAttr(fid, "precision", H5T_NATIVE_INT, &precision),
                 "attr precision");
        H5_CHECK(putAttr(fid, "status", H5T_NATIVE_INT, &status), "attr status");
        H5_CHECK(putAttr(fid, "total_iterations", H5T_NATIVE_INT,
                         &report.total_effective_iterations),
                 "attr total_iterations");
        H5_CHECK(putAttr(fid, "build_dirty", H5T_NATIVE_INT, &dirty),
                 "attr build_dirty");
        H5_CHECK(putStrAttr(fid, "revision", report.build.revision),
                 "attr revision");
        H5_CHECK(putStrAttr(fid, "build_type", report.build.build_type),
                 "attr build_type");
        H5_CHECK(putStrAttr(fid, "scalar_type", report.build.scalar_type),
                 "attr scalar_type");
        H5_CHECK(putStrAttr(fid, "precision_policy",
                            report.build.precision_policy),
                 "attr precision_policy");
        H5_CHECK(putStrAttr(fid, "compile_flags", report.build.compile_flags),
                 "attr compile_flags");
        H5_CHECK(putStrAttr(fid, "source_path", report.input.source_path),
                 "attr source_path");
        H5_CHECK(putStrAttr(fid, "source_hash", report.input.source_hash),
                 "attr source_hash");
        H5_CHECK(putStrAttr(fid, "gpu_name", report.runtime.gpu_name),
                 "attr gpu_name");
        H5_CHECK(putStrAttr(fid, "driver", report.runtime.driver),
                 "attr driver");
        H5_CHECK(putStrAttr(fid, "runtime", report.runtime.runtime),
                 "attr runtime");
        H5_CHECK(putStrAttr(fid, "toolkit", report.runtime.toolkit),
                 "attr toolkit");

        // ---- stage history + restart arrays (active dims) ----
        std::vector<int> stage_ns(nstages), stage_iter(nstages),
            stage_conv(nstages), rst_off(nstages);
        std::vector<double> st_fsqr(nstages), st_fsqz(nstages), st_fsql(nstages);
        std::vector<int> rst_iter(nrestarts);
        size_t r = 0;
        for (size_t g = 0; g < nstages; ++g) {
            const StageReport& st = report.stages[g];
            stage_ns[g] = st.ns;
            stage_iter[g] = st.effective_iterations;
            stage_conv[g] = st.converged ? 1 : 0;
            st_fsqr[g] = st.final_residual.fsqr;
            st_fsqz[g] = st.final_residual.fsqz;
            st_fsql[g] = st.final_residual.fsql;
            rst_off[g] = static_cast<int>(r);
            for (const auto& ev : st.restarts) rst_iter[r++] = ev.iteration;
        }
        const hsize_t d_stages[1] = {nstages};
        const hsize_t d_restarts[1] = {nrestarts};
        H5_CHECK(writeArray("stage_ns", H5T_NATIVE_INT, 1, d_stages,
                            stage_ns.data()), "write stage_ns");
        H5_CHECK(writeArray("stage_iterations", H5T_NATIVE_INT, 1, d_stages,
                            stage_iter.data()), "write stage_iterations");
        H5_CHECK(writeArray("stage_converged", H5T_NATIVE_INT, 1, d_stages,
                            stage_conv.data()), "write stage_converged");
        H5_CHECK(writeArray("stage_fsqr", H5T_NATIVE_DOUBLE, 1, d_stages,
                            st_fsqr.data()), "write stage_fsqr");
        H5_CHECK(writeArray("stage_fsqz", H5T_NATIVE_DOUBLE, 1, d_stages,
                            st_fsqz.data()), "write stage_fsqz");
        H5_CHECK(writeArray("stage_fsql", H5T_NATIVE_DOUBLE, 1, d_stages,
                            st_fsql.data()), "write stage_fsql");
        H5_CHECK(writeArray("restart_stage_offset", H5T_NATIVE_INT, 1, d_stages,
                            rst_off.data()), "write restart_stage_offset");
        if (nrestarts > 0) {
            H5_CHECK(writeArray("restart_iteration", H5T_NATIVE_INT, 1,
                                d_restarts, rst_iter.data()),
                     "write restart_iteration");
        }

        // ---- raw boundary harmonics ----
        const size_t nrbc = rbc.size(), nzbs = zbs.size();
        std::vector<int> hm(std::max(nrbc, nzbs)), hn(std::max(nrbc, nzbs));
        std::vector<double> hv(std::max(nrbc, nzbs));
        if (nrbc > 0) {
            const hsize_t d_r[1] = {nrbc};
            for (size_t i = 0; i < nrbc; ++i) {
                hm[i] = rbc[i].m; hn[i] = rbc[i].n; hv[i] = rbc[i].value;
            }
            H5_CHECK(writeArray("rbc_m", H5T_NATIVE_INT, 1, d_r, hm.data()),
                     "write rbc_m");
            H5_CHECK(writeArray("rbc_n", H5T_NATIVE_INT, 1, d_r, hn.data()),
                     "write rbc_n");
            H5_CHECK(writeArray("rbc_value", H5T_NATIVE_DOUBLE, 1, d_r,
                                hv.data()), "write rbc_value");
        }
        if (nzbs > 0) {
            const hsize_t d_z[1] = {nzbs};
            for (size_t i = 0; i < nzbs; ++i) {
                hm[i] = zbs[i].m; hn[i] = zbs[i].n; hv[i] = zbs[i].value;
            }
            H5_CHECK(writeArray("zbs_m", H5T_NATIVE_INT, 1, d_z, hm.data()),
                     "write zbs_m");
            H5_CHECK(writeArray("zbs_n", H5T_NATIVE_INT, 1, d_z, hn.data()),
                     "write zbs_n");
            H5_CHECK(writeArray("zbs_value", H5T_NATIVE_DOUBLE, 1, d_z,
                                hv.data()), "write zbs_value");
        }

        // ---- folded boundary matrices (active mpol x (ntor+1)) ----
        const hsize_t bnd_dims[2] = {(hsize_t)mpol, (hsize_t)ntorp1};
        H5_CHECK(writeArray("rbcc", H5T_NATIVE_DOUBLE, 2, bnd_dims,
                            fb.rbcc.data()), "write rbcc");
        H5_CHECK(writeArray("rbss", H5T_NATIVE_DOUBLE, 2, bnd_dims,
                            fb.rbss.data()), "write rbss");
        H5_CHECK(writeArray("zbsc", H5T_NATIVE_DOUBLE, 2, bnd_dims,
                            fb.zbsc.data()), "write zbsc");
        H5_CHECK(writeArray("zbcs", H5T_NATIVE_DOUBLE, 2, bnd_dims,
                            fb.zbcs.data()), "write zbcs");

        if (H5Fclose(fid) < 0) {
            remove(tmp.c_str());
            return Status("HDF5 H5Fclose failed");
        }
#undef H5_CHECK
        // Durable publication (completion-plan follow-up §3): the library
        // owns the descriptor, so publishLibraryFile reopens the completed
        // temp, checks fsync + close, then renames + directory-fsyncs.
        const std::string err = io_detail::publishLibraryFile(tmp, spec.path);
        if (!err.empty()) return Status("HDF5 publish: " + err);
        return Status();
    }
};

// ---------------------------------------------------------------------------
// v1 reader: round-trips the state AND the complete RunReport
// ---------------------------------------------------------------------------
class Hdf5V1Reader final : public Reader {
 public:
    Result<EquilibriumSnapshot> read(const std::string& path,
                                     RunReport* report) override {
        hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid < 0) {
            return Result<EquilibriumSnapshot>("HDF5: cannot open " + path);
        }
        auto fail = [&](const std::string& msg) -> Result<EquilibriumSnapshot> {
            H5Fclose(fid);
            return Result<EquilibriumSnapshot>("HDF5: " + msg);
        };
        auto getDim = [&](const char* name, hsize_t* dims) -> bool {
            hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
            if (ds < 0) return false;
            hid_t sp = H5Dget_space(ds);
            // Rank >= 1: the state datasets are 2-D, the stage/restart arrays
            // 1-D. A scalar (rank 0) is never queried here.
            const bool ok = H5Sget_simple_extent_dims(sp, dims, nullptr) >= 1;
            H5Sclose(sp);
            H5Dclose(ds);
            return ok;
        };
        hsize_t state_dims[2] = {0, 0};
        if (!getDim("rmncc", state_dims)) {
            return fail("missing state dataset");
        }
        const int ns = static_cast<int>(state_dims[0]);
        const int mnmax = static_cast<int>(state_dims[1]);
        if (ns < 1 || mnmax < 1) return fail("bad state dimensions");
        const auto n_opt = checked_mul((size_t)ns, (size_t)mnmax);
        if (!n_opt) return fail("dimension product overflows size_t");

        EquilibriumSnapshot snapshot;
        snapshot.ns = ns;
        snapshot.mnmax = mnmax;
        const char* fam_names[6] = {"rmncc", "zmnsc", "lmnsc",
                                    "rmnss", "zmncs", "lmncs"};
        // The file layout is [surface, mode] (C order over the (ns, mnmax)
        // dataspace) while the snapshot is mode-major (index = m*ns + j). A
        // whole-slab read must therefore TRANSPOSE — the exact
        // declaration/data-order trap the blueprint pins (completion plan
        // step 2.3). The writers use per-mode hyperslabs, which is the
        // transpose-aware mirror image.
        std::vector<double> tmp(*n_opt);
        for (int c = 0; c < 6; ++c) {
            hid_t ds = H5Dopen2(fid, fam_names[c], H5P_DEFAULT);
            if (ds < 0) return fail("missing state dataset");
            hid_t sp = H5Screate_simple(2, state_dims, nullptr);
            const herr_t r = H5Dread(ds, H5T_NATIVE_DOUBLE, sp, H5S_ALL,
                                     H5P_DEFAULT, tmp.data());
            H5Sclose(sp);
            H5Dclose(ds);
            if (r < 0) return fail("state read failed");
            snapshot.families[c].resize(*n_opt);
            for (int m = 0; m < mnmax; ++m) {
                for (int j = 0; j < ns; ++j) {
                    snapshot.families[c][(size_t)m * ns + j] =
                        tmp[(size_t)j * mnmax + m];
                }
            }
        }
        if (report) {
            *report = RunReport{};
            auto getIntAttr = [&](const char* name, int& out) -> bool {
                hid_t aid = H5Aopen(fid, name, H5P_DEFAULT);
                if (aid < 0) return false;
                const herr_t r = H5Aread(aid, H5T_NATIVE_INT, &out);
                H5Aclose(aid);
                return r >= 0;
            };
            auto getIntArr = [&](const char* name, std::vector<int>& out,
                                 size_t expect) -> bool {
                hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
                if (ds < 0) return false;
                hsize_t dims[1] = {0};
                hid_t sp = H5Dget_space(ds);
                H5Sget_simple_extent_dims(sp, dims, nullptr);
                H5Sclose(sp);
                if (dims[0] != expect) {
                    H5Dclose(ds);
                    return false;
                }
                out.resize(expect);
                const herr_t r = H5Dread(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                         H5P_DEFAULT, out.data());
                H5Dclose(ds);
                return r >= 0;
            };
            auto getDblArr = [&](const char* name, std::vector<double>& out,
                                 size_t expect) -> bool {
                hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
                if (ds < 0) return false;
                hsize_t dims[1] = {0};
                hid_t sp = H5Dget_space(ds);
                H5Sget_simple_extent_dims(sp, dims, nullptr);
                H5Sclose(sp);
                if (dims[0] != expect) {
                    H5Dclose(ds);
                    return false;
                }
                out.resize(expect);
                const herr_t r = H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL,
                                         H5S_ALL, H5P_DEFAULT, out.data());
                H5Dclose(ds);
                return r >= 0;
            };
            int precision = 0, status = 0, total = 0, dirty = 0;
            if (!getIntAttr("precision", precision) ||
                !getIntAttr("status", status) ||
                !getIntAttr("total_iterations", total) ||
                !getIntAttr("build_dirty", dirty)) {
                return fail("missing run outcome attributes");
            }
            report->status = static_cast<RunStatus>(status);
            report->total_effective_iterations = total;
            report->build.dirty = (dirty != 0);
            report->build.scalar_type = (precision == 0) ? "double" : "float";
            if (!getStrAttr(fid, "revision", report->build.revision) ||
                !getStrAttr(fid, "build_type", report->build.build_type) ||
                !getStrAttr(fid, "precision_policy",
                            report->build.precision_policy) ||
                !getStrAttr(fid, "compile_flags", report->build.compile_flags) ||
                !getStrAttr(fid, "source_path", report->input.source_path) ||
                !getStrAttr(fid, "source_hash", report->input.source_hash) ||
                !getStrAttr(fid, "gpu_name", report->runtime.gpu_name) ||
                !getStrAttr(fid, "driver", report->runtime.driver) ||
                !getStrAttr(fid, "runtime", report->runtime.runtime) ||
                !getStrAttr(fid, "toolkit", report->runtime.toolkit)) {
                return fail("missing provenance attributes");
            }
            hsize_t d_stages[1] = {0};
            if (!getDim("stage_ns", d_stages)) return fail("missing stage_ns");
            const size_t nstages = (size_t)d_stages[0];
            hsize_t d_restarts[1] = {0};
            const size_t nrestarts =
                getDim("restart_iteration", d_restarts) ? (size_t)d_restarts[0]
                                                        : 0;
            std::vector<int> stage_ns, stage_iter, stage_conv, rst_off;
            std::vector<double> st_fsqr, st_fsqz, st_fsql;
            std::vector<int> rst_iter;
            if (!getIntArr("stage_ns", stage_ns, nstages) ||
                !getIntArr("stage_iterations", stage_iter, nstages) ||
                !getIntArr("stage_converged", stage_conv, nstages) ||
                !getIntArr("restart_stage_offset", rst_off, nstages) ||
                !getDblArr("stage_fsqr", st_fsqr, nstages) ||
                !getDblArr("stage_fsqz", st_fsqz, nstages) ||
                !getDblArr("stage_fsql", st_fsql, nstages) ||
                (nrestarts > 0 &&
                 !getIntArr("restart_iteration", rst_iter, nrestarts))) {
                return fail("stage history read failed");
            }
            // Validate the offsets BEFORE they index rst_iter (completion-plan
            // follow-up §2.2): negative, descending, oversized, or non-zero
            // first offsets must fail cleanly instead of reading out of
            // bounds.
            {
                const std::string off_err =
                    validateRestartOffsets(rst_off, nstages, nrestarts);
                if (!off_err.empty()) {
                    return fail("invalid restart offsets: " + off_err);
                }
            }
            for (size_t g = 0; g < nstages; ++g) {
                StageReport st;
                st.ns = stage_ns[g];
                st.effective_iterations = stage_iter[g];
                st.converged = (stage_conv[g] != 0);
                st.final_residual.fsqr = st_fsqr[g];
                st.final_residual.fsqz = st_fsqz[g];
                st.final_residual.fsql = st_fsql[g];
                const size_t begin = (size_t)rst_off[g];
                const size_t end = (g + 1 < nstages) ? (size_t)rst_off[g + 1]
                                                     : nrestarts;
                for (size_t k = begin; k < end; ++k) {
                    st.restarts.push_back(RestartEvent{rst_iter[k]});
                }
                report->stages.push_back(std::move(st));
            }
        }
        H5Fclose(fid);
        return snapshot;
    }
};

}  // namespace

std::unique_ptr<Writer> make_hdf5_v0_writer() {
    return std::make_unique<Hdf5V0Writer>();
}
std::unique_ptr<Writer> make_hdf5_v1_writer() {
    return std::make_unique<Hdf5V1Writer>();
}
std::unique_ptr<Reader> make_hdf5_v1_reader() {
    return std::make_unique<Hdf5V1Reader>();
}

}  // namespace cumes
