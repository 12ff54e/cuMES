// netcdf_writer.cpp — host-only NetCDF state adapters (completion plan steps
// 2.2/2.3).
//
// Two schemas, one TU, no CUDA:
//   - legacy-v0: byte/layout-exact replica of the pre-overhaul device-reading
//     writer (src/output_netcdf.cpp, deleted): padded fixed capacities,
//     [surface, mode] logical mapping, final-stage scalar pack, legacy
//     provenance. Written purely from the host EquilibriumSnapshot +
//     LegacyInputProvenance + LegacyRunScalars.
//   - v1: active dimensions + complete provenance — precision/status/total
//     iterations, build/input/runtime provenance strings, source hash,
//     per-stage outcomes (ns/iterations/converged/residuals), the full
//     restart history, and the raw boundary harmonics + folded boundary
//     matrices (distinguished, as the blueprint requires). The v1 reader
//     round-trips the state AND the complete RunReport.
//
// This TU (and only this TU) includes <netcdf.h>; it compiles under
// CUMES_HAVE_NETCDF, which is confined to the adapter library target.
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "cumes/io/legacy_provenance.hpp"
#include "internal_factories.hpp"
#include "io_common.hpp"
#include "cumes/io/writer_helpers.hpp"

#include <netcdf.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cumes {
namespace {

bool putStrAttr(int ncid, const char* name, const std::string& value) {
    return nc_put_att_text(ncid, NC_GLOBAL, name, value.size(), value.c_str()) ==
           NC_NOERR;
}

// ---------------------------------------------------------------------------
// legacy-v0: byte/layout-exact replica of the deleted device-reading writer
// ---------------------------------------------------------------------------
class NetcdfV0Writer final : public Writer {
 public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report, const OutputSpec& spec,
                        const ValidatedProblem& problem,
                        const LegacyRunScalars& s) override {
        const LegacyInputProvenance pv =
            LegacyInputProvenance::from_validated(problem);
        const std::string tmp = io_detail::tempPathFor(spec.path);
        int ncid = -1;
        int rc = nc_create(tmp.c_str(), NC_CLOBBER, &ncid);
        if (rc != NC_NOERR) {
            return Status("NetCDF nc_create failed: " +
                          std::string(nc_strerror(rc)));
        }
        auto fail = [&](const std::string& msg) -> Status {
            nc_close(ncid);
            remove(tmp.c_str());
            return Status("NetCDF: " + msg);
        };
#define NC_CHECK(rc_, msg_)                                                    \
    do {                                                                       \
        int _rc = (rc_);                                                       \
        if (_rc != NC_NOERR) {                                                 \
            return fail(std::string(msg_) + ": " + nc_strerror(_rc));          \
        }                                                                      \
    } while (0)

        // ---- dimensions (fixed v0 capacities — the legacy layout) ----
        int dim_ns, dim_mnmax, dim_ngrids, dim_ncoeff, dim_naxis, dim_nbm,
            dim_nbn;
        NC_CHECK(nc_def_dim(ncid, "ns", (size_t)s.ns, &dim_ns), "def dim ns");
        NC_CHECK(nc_def_dim(ncid, "mnmax", (size_t)s.mnmax, &dim_mnmax),
                 "def dim mnmax");
        NC_CHECK(nc_def_dim(ncid, "ngrids", LegacyInputProvenance::kMaxGrids,
                            &dim_ngrids), "def dim ngrids");
        NC_CHECK(nc_def_dim(ncid, "ncoeff", LegacyInputProvenance::kMaxCoeff,
                            &dim_ncoeff), "def dim ncoeff");
        NC_CHECK(nc_def_dim(ncid, "naxis", LegacyInputProvenance::kMaxAxis,
                            &dim_naxis), "def dim naxis");
        NC_CHECK(nc_def_dim(ncid, "nbm", LegacyInputProvenance::kMaxM,
                            &dim_nbm), "def dim nbm");
        NC_CHECK(nc_def_dim(ncid, "nbn", LegacyInputProvenance::kMaxN,
                            &dim_nbn), "def dim nbn");

        // ---- state variables (ns, mnmax) ----
        const int state_dims[2] = {dim_ns, dim_mnmax};
        int v_rmncc, v_zmnsc, v_lmnsc, v_rmnss, v_zmncs, v_lmncs;
        NC_CHECK(nc_def_var(ncid, "rmncc", NC_DOUBLE, 2, state_dims, &v_rmncc),
                 "def rmncc");
        NC_CHECK(nc_def_var(ncid, "zmnsc", NC_DOUBLE, 2, state_dims, &v_zmnsc),
                 "def zmnsc");
        NC_CHECK(nc_def_var(ncid, "lmnsc", NC_DOUBLE, 2, state_dims, &v_lmnsc),
                 "def lmnsc");
        NC_CHECK(nc_def_var(ncid, "rmnss", NC_DOUBLE, 2, state_dims, &v_rmnss),
                 "def rmnss");
        NC_CHECK(nc_def_var(ncid, "zmncs", NC_DOUBLE, 2, state_dims, &v_zmncs),
                 "def zmncs");
        NC_CHECK(nc_def_var(ncid, "lmncs", NC_DOUBLE, 2, state_dims, &v_lmncs),
                 "def lmncs");

        // ---- scalar variables (0 dims) — same order as the legacy writer ----
        struct IntScalar { const char* name; int value; };
        const IntScalar int_scalars[] = {
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
        int v_int_scalar[sizeof(int_scalars) / sizeof(int_scalars[0])];
        for (size_t i = 0; i < sizeof(int_scalars) / sizeof(int_scalars[0]);
             ++i) {
            NC_CHECK(nc_def_var(ncid, int_scalars[i].name, NC_INT, 0, nullptr,
                                &v_int_scalar[i]), "def int scalar");
        }
        struct DblScalar { const char* name; double value; };
        const DblScalar dbl_scalars[] = {
            {"delt", s.delt}, {"ftol", s.ftol}, {"lamscale", s.lamscale},
            {"phiedge", pv.phiedge}, {"pres_scale", pv.pres_scale},
            {"adiabatic_index", pv.adiabatic_index}, {"spres_ped", pv.spres_ped},
            {"bloat", pv.bloat}, {"curtor", pv.curtor}, {"tcon0", pv.tcon0},
            {"fsqr", s.fsqr}, {"fsqz", s.fsqz}, {"fsql", s.fsql},
        };
        int v_dbl_scalar[sizeof(dbl_scalars) / sizeof(dbl_scalars[0])];
        for (size_t i = 0; i < sizeof(dbl_scalars) / sizeof(dbl_scalars[0]);
             ++i) {
            NC_CHECK(nc_def_var(ncid, dbl_scalars[i].name, NC_DOUBLE, 0,
                                nullptr, &v_dbl_scalar[i]), "def double scalar");
        }

        // ---- provenance arrays ----
        int v_ns_array, v_niter_array, v_ftol_array;
        NC_CHECK(nc_def_var(ncid, "ns_array", NC_INT, 1, &dim_ngrids,
                            &v_ns_array), "def ns_array");
        NC_CHECK(nc_def_var(ncid, "niter_array", NC_INT, 1, &dim_ngrids,
                            &v_niter_array), "def niter_array");
        NC_CHECK(nc_def_var(ncid, "ftol_array", NC_DOUBLE, 1, &dim_ngrids,
                            &v_ftol_array), "def ftol_array");
        int v_am, v_ac, v_ai, v_aphi;
        NC_CHECK(nc_def_var(ncid, "am", NC_DOUBLE, 1, &dim_ncoeff, &v_am),
                 "def am");
        NC_CHECK(nc_def_var(ncid, "ac", NC_DOUBLE, 1, &dim_ncoeff, &v_ac),
                 "def ac");
        NC_CHECK(nc_def_var(ncid, "ai", NC_DOUBLE, 1, &dim_ncoeff, &v_ai),
                 "def ai");
        NC_CHECK(nc_def_var(ncid, "aphi", NC_DOUBLE, 1, &dim_ncoeff, &v_aphi),
                 "def aphi");
        int v_raxis_c, v_zaxis_s;
        NC_CHECK(nc_def_var(ncid, "raxis_c", NC_DOUBLE, 1, &dim_naxis,
                            &v_raxis_c), "def raxis_c");
        NC_CHECK(nc_def_var(ncid, "zaxis_s", NC_DOUBLE, 1, &dim_naxis,
                            &v_zaxis_s), "def zaxis_s");
        const int bnd_dims[2] = {dim_nbm, dim_nbn};
        int v_rbcc, v_rbss, v_zbsc, v_zbcs;
        NC_CHECK(nc_def_var(ncid, "rbcc", NC_DOUBLE, 2, bnd_dims, &v_rbcc),
                 "def rbcc");
        NC_CHECK(nc_def_var(ncid, "rbss", NC_DOUBLE, 2, bnd_dims, &v_rbss),
                 "def rbss");
        NC_CHECK(nc_def_var(ncid, "zbsc", NC_DOUBLE, 2, bnd_dims, &v_zbsc),
                 "def zbsc");
        NC_CHECK(nc_def_var(ncid, "zbcs", NC_DOUBLE, 2, bnd_dims, &v_zbcs),
                 "def zbcs");

        // ---- global attributes ----
        NC_CHECK(putStrAttr(ncid, "input_file", report.input.source_path) == true
                     ? NC_NOERR : NC_EATTMETA, "attr input_file");
        NC_CHECK(putStrAttr(ncid, "precision", report.build.scalar_type) == true
                     ? NC_NOERR : NC_EATTMETA, "attr precision");

        NC_CHECK(nc_enddef(ncid), "nc_enddef");

        // ---- state data (per-mode hyperslabs, see layout note) ----
        if (snapshot.ns != s.ns || snapshot.mnmax != s.mnmax) {
            return fail("snapshot dimensions do not match the scalar pack");
        }
        auto writeFam = [&](EquilibriumSnapshot::Component comp, int varid,
                            const char* tag) -> Status {
            const std::vector<double>& dbuf = snapshot.component(comp);
            if (dbuf.size() != snapshot.family_size()) {
                return fail(std::string(tag) + ": family size mismatch");
            }
            for (int m = 0; m < s.mnmax; ++m) {
                const size_t start[2] = {0, (size_t)m};
                const size_t count[2] = {(size_t)s.ns, 1};
                int rc2 = nc_put_vara_double(ncid, varid, start, count,
                                             dbuf.data() + (size_t)m * s.ns);
                if (rc2 != NC_NOERR) {
                    return fail(std::string(tag) + ": " + nc_strerror(rc2));
                }
            }
            return Status();
        };
        Status st = writeFam(EquilibriumSnapshot::kRmncc, v_rmncc, "write rmncc");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kZmnsc, v_zmnsc, "write zmnsc");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kLmnsc, v_lmnsc, "write lmnsc");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kRmnss, v_rmnss, "write rmnss");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kZmncs, v_zmncs, "write zmncs");
        if (!st.has_value()) return st;
        st = writeFam(EquilibriumSnapshot::kLmncs, v_lmncs, "write lmncs");
        if (!st.has_value()) return st;

        // ---- scalar data ----
        for (size_t i = 0; i < sizeof(int_scalars) / sizeof(int_scalars[0]);
             ++i) {
            NC_CHECK(nc_put_var_int(ncid, v_int_scalar[i],
                                    &int_scalars[i].value), "put int scalar");
        }
        for (size_t i = 0; i < sizeof(dbl_scalars) / sizeof(dbl_scalars[0]);
             ++i) {
            NC_CHECK(nc_put_var_double(ncid, v_dbl_scalar[i],
                                       &dbl_scalars[i].value),
                     "put double scalar");
        }

        // ---- provenance data (full fixed-size arrays) ----
        NC_CHECK(nc_put_var_int(ncid, v_ns_array, pv.ns_array), "put ns_array");
        NC_CHECK(nc_put_var_int(ncid, v_niter_array, pv.niter_array),
                 "put niter_array");
        NC_CHECK(nc_put_var_double(ncid, v_ftol_array, pv.ftol_array),
                 "put ftol_array");
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

        {
            int _rc = nc_close(ncid);
            if (_rc != NC_NOERR) {
                remove(tmp.c_str());
                return Status("NetCDF nc_close failed: " +
                              std::string(nc_strerror(_rc)));
            }
        }
#undef NC_CHECK
        const std::string err = io_detail::renamePublish(tmp, spec.path);
        if (!err.empty()) return Status("NetCDF publish: " + err);
        return Status();
    }
};

// ---------------------------------------------------------------------------
// v1: active dimensions + complete provenance + boundary harmonics
// ---------------------------------------------------------------------------
class NetcdfV1Writer final : public Writer {
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

        const std::string tmp = io_detail::tempPathFor(spec.path);
        int ncid = -1;
        int rc = nc_create(tmp.c_str(), NC_CLOBBER, &ncid);
        if (rc != NC_NOERR) {
            return Status("NetCDF nc_create failed: " +
                          std::string(nc_strerror(rc)));
        }
        auto fail = [&](const std::string& msg) -> Status {
            nc_close(ncid);
            remove(tmp.c_str());
            return Status("NetCDF: " + msg);
        };
#define NC_CHECK(rc_, msg_)                                                    \
    do {                                                                       \
        int _rc = (rc_);                                                       \
        if (_rc != NC_NOERR) {                                                 \
            return fail(std::string(msg_) + ": " + nc_strerror(_rc));          \
        }                                                                      \
    } while (0)

        const size_t nstages = report.stages.size();
        size_t nrestarts = 0;
        for (const auto& st : report.stages) nrestarts += st.restarts.size();

        // ---- active dimensions ----
        int dim_ns, dim_mnmax, dim_nstages, dim_nrestarts, dim_nrbc, dim_nzbs,
            dim_mpol, dim_ntorp1;
        NC_CHECK(nc_def_dim(ncid, "ns", (size_t)snapshot.ns, &dim_ns), "def ns");
        NC_CHECK(nc_def_dim(ncid, "mnmax", (size_t)snapshot.mnmax, &dim_mnmax),
                 "def mnmax");
        NC_CHECK(nc_def_dim(ncid, "nstages", nstages, &dim_nstages), "def nstages");
        NC_CHECK(nc_def_dim(ncid, "nrestarts", nrestarts, &dim_nrestarts),
                 "def nrestarts");
        NC_CHECK(nc_def_dim(ncid, "nrbc", rbc.size(), &dim_nrbc), "def nrbc");
        NC_CHECK(nc_def_dim(ncid, "nzbs", zbs.size(), &dim_nzbs), "def nzbs");
        NC_CHECK(nc_def_dim(ncid, "mpol", (size_t)mpol, &dim_mpol), "def mpol");
        NC_CHECK(nc_def_dim(ncid, "ntorp1", (size_t)ntorp1, &dim_ntorp1),
                 "def ntorp1");

        // ---- state variables ----
        const int state_dims[2] = {dim_ns, dim_mnmax};
        const char* fam_names[6] = {"rmncc", "zmnsc", "lmnsc",
                                    "rmnss", "zmncs", "lmncs"};
        int v_fam[6];
        for (int c = 0; c < 6; ++c) {
            NC_CHECK(nc_def_var(ncid, fam_names[c], NC_DOUBLE, 2, state_dims,
                                &v_fam[c]), "def family");
        }

        // ---- run outcome + stage history ----
        int v_precision, v_status, v_total_iter, v_dirty;
        NC_CHECK(nc_def_var(ncid, "precision", NC_INT, 0, nullptr, &v_precision),
                 "def precision");
        NC_CHECK(nc_def_var(ncid, "status", NC_INT, 0, nullptr, &v_status),
                 "def status");
        NC_CHECK(nc_def_var(ncid, "total_iterations", NC_INT, 0, nullptr,
                            &v_total_iter), "def total_iterations");
        NC_CHECK(nc_def_var(ncid, "build_dirty", NC_INT, 0, nullptr, &v_dirty),
                 "def build_dirty");
        int v_ns_array, v_iter_array, v_conv, v_fsqr, v_fsqz, v_fsql, v_rst_off,
            v_rst_iter;
        NC_CHECK(nc_def_var(ncid, "stage_ns", NC_INT, 1, &dim_nstages,
                            &v_ns_array), "def stage_ns");
        NC_CHECK(nc_def_var(ncid, "stage_iterations", NC_INT, 1, &dim_nstages,
                            &v_iter_array), "def stage_iterations");
        NC_CHECK(nc_def_var(ncid, "stage_converged", NC_INT, 1, &dim_nstages,
                            &v_conv), "def stage_converged");
        NC_CHECK(nc_def_var(ncid, "stage_fsqr", NC_DOUBLE, 1, &dim_nstages,
                            &v_fsqr), "def stage_fsqr");
        NC_CHECK(nc_def_var(ncid, "stage_fsqz", NC_DOUBLE, 1, &dim_nstages,
                            &v_fsqz), "def stage_fsqz");
        NC_CHECK(nc_def_var(ncid, "stage_fsql", NC_DOUBLE, 1, &dim_nstages,
                            &v_fsql), "def stage_fsql");
        NC_CHECK(nc_def_var(ncid, "restart_stage_offset", NC_INT, 1,
                            &dim_nstages, &v_rst_off), "def restart_stage_offset");
        NC_CHECK(nc_def_var(ncid, "restart_iteration", NC_INT, 1, &dim_nrestarts,
                            &v_rst_iter), "def restart_iteration");

        // ---- raw boundary harmonics (distinguished from the folded basis) ----
        int v_rbc_m, v_rbc_n, v_rbc_v, v_zbs_m, v_zbs_n, v_zbs_v;
        NC_CHECK(nc_def_var(ncid, "rbc_m", NC_INT, 1, &dim_nrbc, &v_rbc_m),
                 "def rbc_m");
        NC_CHECK(nc_def_var(ncid, "rbc_n", NC_INT, 1, &dim_nrbc, &v_rbc_n),
                 "def rbc_n");
        NC_CHECK(nc_def_var(ncid, "rbc_value", NC_DOUBLE, 1, &dim_nrbc,
                            &v_rbc_v), "def rbc_value");
        NC_CHECK(nc_def_var(ncid, "zbs_m", NC_INT, 1, &dim_nzbs, &v_zbs_m),
                 "def zbs_m");
        NC_CHECK(nc_def_var(ncid, "zbs_n", NC_INT, 1, &dim_nzbs, &v_zbs_n),
                 "def zbs_n");
        NC_CHECK(nc_def_var(ncid, "zbs_value", NC_DOUBLE, 1, &dim_nzbs,
                            &v_zbs_v), "def zbs_value");

        // ---- folded boundary matrices (active mpol x (ntor+1)) ----
        const int bnd_dims[2] = {dim_mpol, dim_ntorp1};
        int v_rbcc, v_rbss, v_zbsc, v_zbcs;
        NC_CHECK(nc_def_var(ncid, "rbcc", NC_DOUBLE, 2, bnd_dims, &v_rbcc),
                 "def rbcc");
        NC_CHECK(nc_def_var(ncid, "rbss", NC_DOUBLE, 2, bnd_dims, &v_rbss),
                 "def rbss");
        NC_CHECK(nc_def_var(ncid, "zbsc", NC_DOUBLE, 2, bnd_dims, &v_zbsc),
                 "def zbsc");
        NC_CHECK(nc_def_var(ncid, "zbcs", NC_DOUBLE, 2, bnd_dims, &v_zbcs),
                 "def zbcs");

        // ---- provenance attributes ----
        NC_CHECK(putStrAttr(ncid, "revision", report.build.revision) == true
                     ? NC_NOERR : NC_EATTMETA, "attr revision");
        NC_CHECK(putStrAttr(ncid, "build_type", report.build.build_type) == true
                     ? NC_NOERR : NC_EATTMETA, "attr build_type");
        NC_CHECK(putStrAttr(ncid, "scalar_type", report.build.scalar_type) == true
                     ? NC_NOERR : NC_EATTMETA, "attr scalar_type");
        NC_CHECK(putStrAttr(ncid, "precision_policy",
                            report.build.precision_policy) == true
                     ? NC_NOERR : NC_EATTMETA, "attr precision_policy");
        NC_CHECK(putStrAttr(ncid, "compile_flags", report.build.compile_flags) == true
                     ? NC_NOERR : NC_EATTMETA, "attr compile_flags");
        NC_CHECK(putStrAttr(ncid, "source_path", report.input.source_path) == true
                     ? NC_NOERR : NC_EATTMETA, "attr source_path");
        NC_CHECK(putStrAttr(ncid, "source_hash", report.input.source_hash) == true
                     ? NC_NOERR : NC_EATTMETA, "attr source_hash");
        NC_CHECK(putStrAttr(ncid, "gpu_name", report.runtime.gpu_name) == true
                     ? NC_NOERR : NC_EATTMETA, "attr gpu_name");
        NC_CHECK(putStrAttr(ncid, "driver", report.runtime.driver) == true
                     ? NC_NOERR : NC_EATTMETA, "attr driver");
        NC_CHECK(putStrAttr(ncid, "runtime", report.runtime.runtime) == true
                     ? NC_NOERR : NC_EATTMETA, "attr runtime");
        NC_CHECK(putStrAttr(ncid, "toolkit", report.runtime.toolkit) == true
                     ? NC_NOERR : NC_EATTMETA, "attr toolkit");

        NC_CHECK(nc_enddef(ncid), "nc_enddef");

        // ---- data ----
        for (int c = 0; c < 6; ++c) {
            const std::vector<double>& dbuf =
                snapshot.component(static_cast<EquilibriumSnapshot::Component>(c));
            if (dbuf.size() != snapshot.family_size()) {
                return fail("family size mismatch");
            }
            for (int m = 0; m < snapshot.mnmax; ++m) {
                const size_t start[2] = {0, (size_t)m};
                const size_t count[2] = {(size_t)snapshot.ns, 1};
                int rc2 = nc_put_vara_double(ncid, v_fam[c], start, count,
                                             dbuf.data() + (size_t)m * snapshot.ns);
                if (rc2 != NC_NOERR) {
                    return fail("write family: " + std::string(nc_strerror(rc2)));
                }
            }
        }
        const int precision = (report.build.scalar_type == "double") ? 0 : 1;
        NC_CHECK(nc_put_var_int(ncid, v_precision, &precision), "put precision");
        const int status = static_cast<int>(report.status);
        NC_CHECK(nc_put_var_int(ncid, v_status, &status), "put status");
        NC_CHECK(nc_put_var_int(ncid, v_total_iter,
                                &report.total_effective_iterations),
                 "put total_iterations");
        const int dirty = report.build.dirty ? 1 : 0;
        NC_CHECK(nc_put_var_int(ncid, v_dirty, &dirty), "put build_dirty");
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
        NC_CHECK(nc_put_var_int(ncid, v_ns_array, stage_ns.data()),
                 "put stage_ns");
        NC_CHECK(nc_put_var_int(ncid, v_iter_array, stage_iter.data()),
                 "put stage_iterations");
        NC_CHECK(nc_put_var_int(ncid, v_conv, stage_conv.data()),
                 "put stage_converged");
        NC_CHECK(nc_put_var_double(ncid, v_fsqr, st_fsqr.data()),
                 "put stage_fsqr");
        NC_CHECK(nc_put_var_double(ncid, v_fsqz, st_fsqz.data()),
                 "put stage_fsqz");
        NC_CHECK(nc_put_var_double(ncid, v_fsql, st_fsql.data()),
                 "put stage_fsql");
        NC_CHECK(nc_put_var_int(ncid, v_rst_off, rst_off.data()),
                 "put restart_stage_offset");
        if (nrestarts > 0) {
            NC_CHECK(nc_put_var_int(ncid, v_rst_iter, rst_iter.data()),
                     "put restart_iteration");
        }
        std::vector<int> hm((size_t)std::max(rbc.size(), zbs.size()));
        std::vector<int> hn((size_t)std::max(rbc.size(), zbs.size()));
        std::vector<double> hv((size_t)std::max(rbc.size(), zbs.size()));
        if (!rbc.empty()) {
            for (size_t i = 0; i < rbc.size(); ++i) {
                hm[i] = rbc[i].m; hn[i] = rbc[i].n; hv[i] = rbc[i].value;
            }
            NC_CHECK(nc_put_var_int(ncid, v_rbc_m, hm.data()), "put rbc_m");
            NC_CHECK(nc_put_var_int(ncid, v_rbc_n, hn.data()), "put rbc_n");
            NC_CHECK(nc_put_var_double(ncid, v_rbc_v, hv.data()), "put rbc_value");
        }
        if (!zbs.empty()) {
            for (size_t i = 0; i < zbs.size(); ++i) {
                hm[i] = zbs[i].m; hn[i] = zbs[i].n; hv[i] = zbs[i].value;
            }
            NC_CHECK(nc_put_var_int(ncid, v_zbs_m, hm.data()), "put zbs_m");
            NC_CHECK(nc_put_var_int(ncid, v_zbs_n, hn.data()), "put zbs_n");
            NC_CHECK(nc_put_var_double(ncid, v_zbs_v, hv.data()), "put zbs_value");
        }
        NC_CHECK(nc_put_var_double(ncid, v_rbcc, fb.rbcc.data()), "put rbcc");
        NC_CHECK(nc_put_var_double(ncid, v_rbss, fb.rbss.data()), "put rbss");
        NC_CHECK(nc_put_var_double(ncid, v_zbsc, fb.zbsc.data()), "put zbsc");
        NC_CHECK(nc_put_var_double(ncid, v_zbcs, fb.zbcs.data()), "put zbcs");

        {
            int _rc = nc_close(ncid);
            if (_rc != NC_NOERR) {
                remove(tmp.c_str());
                return Status("NetCDF nc_close failed: " +
                              std::string(nc_strerror(_rc)));
            }
        }
#undef NC_CHECK
        const std::string err = io_detail::renamePublish(tmp, spec.path);
        if (!err.empty()) return Status("NetCDF publish: " + err);
        return Status();
    }
};

// ---------------------------------------------------------------------------
// v1 reader: round-trips the state AND the complete RunReport
// ---------------------------------------------------------------------------
class NetcdfV1Reader final : public Reader {
 public:
    Result<EquilibriumSnapshot> read(const std::string& path,
                                     RunReport* report) override {
        int ncid = -1;
        int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
        if (rc != NC_NOERR) {
            return Result<EquilibriumSnapshot>("NetCDF: cannot open " + path +
                                               ": " + nc_strerror(rc));
        }
        auto fail = [&](const std::string& msg) -> Result<EquilibriumSnapshot> {
            nc_close(ncid);
            return Result<EquilibriumSnapshot>("NetCDF: " + msg);
        };
        auto getDim = [&](const char* name, size_t& out) -> bool {
            int id = -1;
            return nc_inq_dimid(ncid, name, &id) == NC_NOERR &&
                   nc_inq_dimlen(ncid, id, &out) == NC_NOERR;
        };
        size_t ns = 0, mnmax = 0;
        if (!getDim("ns", ns) || !getDim("mnmax", mnmax)) {
            return fail("missing state dimensions");
        }
        if (ns < 1 || mnmax < 1) {
            return fail("bad dimensions (ns=" + std::to_string(ns) +
                        ", mnmax=" + std::to_string(mnmax) + ")");
        }
        std::size_t n = 0;
        {
            const auto n_opt = checked_mul(ns, mnmax);
            if (!n_opt) return fail("dimension product overflows size_t");
            n = *n_opt;
        }
        EquilibriumSnapshot snapshot;
        snapshot.ns = static_cast<int>(ns);
        snapshot.mnmax = static_cast<int>(mnmax);
        const char* fam_names[6] = {"rmncc", "zmnsc", "lmnsc",
                                    "rmnss", "zmncs", "lmncs"};
        for (int c = 0; c < 6; ++c) {
            int vid = -1;
            if (nc_inq_varid(ncid, fam_names[c], &vid) != NC_NOERR) {
                return fail("missing state variable");
            }
            snapshot.families[c].resize(n);
            for (int m = 0; m < snapshot.mnmax; ++m) {
                const size_t start[2] = {0, (size_t)m};
                const size_t count[2] = {ns, 1};
                if (nc_get_vara_double(ncid, vid, start, count,
                                       snapshot.families[c].data() +
                                           (size_t)m * ns) != NC_NOERR) {
                    return fail("state read failed");
                }
            }
        }
        if (report) {
            *report = RunReport{};
            auto getInt = [&](const char* name, int& out) -> bool {
                int vid = -1;
                return nc_inq_varid(ncid, name, &vid) == NC_NOERR &&
                       nc_get_var_int(ncid, vid, &out) == NC_NOERR;
            };
            auto getStr = [&](const char* name, std::string& out) -> bool {
                size_t len = 0;
                if (nc_inq_attlen(ncid, NC_GLOBAL, name, &len) != NC_NOERR) {
                    return false;
                }
                out.resize(len);
                return nc_get_att_text(ncid, NC_GLOBAL, name, &out[0]) ==
                       NC_NOERR;
            };
            int precision = 0, status = 0, total = 0, dirty = 0;
            if (!getInt("precision", precision) || !getInt("status", status) ||
                !getInt("total_iterations", total) ||
                !getInt("build_dirty", dirty)) {
                return fail("missing run outcome variables");
            }
            report->status = static_cast<RunStatus>(status);
            report->total_effective_iterations = total;
            report->build.dirty = (dirty != 0);
            report->build.scalar_type = (precision == 0) ? "double" : "float";
            if (!getStr("revision", report->build.revision) ||
                !getStr("build_type", report->build.build_type) ||
                !getStr("precision_policy", report->build.precision_policy) ||
                !getStr("compile_flags", report->build.compile_flags) ||
                !getStr("source_path", report->input.source_path) ||
                !getStr("source_hash", report->input.source_hash) ||
                !getStr("gpu_name", report->runtime.gpu_name) ||
                !getStr("driver", report->runtime.driver) ||
                !getStr("runtime", report->runtime.runtime) ||
                !getStr("toolkit", report->runtime.toolkit)) {
                return fail("missing provenance attributes");
            }
            size_t nstages = 0, nrestarts = 0;
            if (!getDim("nstages", nstages) || !getDim("nrestarts", nrestarts)) {
                return fail("missing stage dimensions");
            }
            std::vector<int> stage_ns(nstages), stage_iter(nstages),
                stage_conv(nstages), rst_off(nstages);
            std::vector<double> st_fsqr(nstages), st_fsqz(nstages),
                st_fsql(nstages);
            std::vector<int> rst_iter(nrestarts);
            auto getIntArr = [&](const char* name, int* out) -> bool {
                int vid = -1;
                return nc_inq_varid(ncid, name, &vid) == NC_NOERR &&
                       nc_get_var_int(ncid, vid, out) == NC_NOERR;
            };
            auto getDblArr = [&](const char* name, double* out) -> bool {
                int vid = -1;
                return nc_inq_varid(ncid, name, &vid) == NC_NOERR &&
                       nc_get_var_double(ncid, vid, out) == NC_NOERR;
            };
            if (!getIntArr("stage_ns", stage_ns.data()) ||
                !getIntArr("stage_iterations", stage_iter.data()) ||
                !getIntArr("stage_converged", stage_conv.data()) ||
                !getIntArr("restart_stage_offset", rst_off.data()) ||
                !getDblArr("stage_fsqr", st_fsqr.data()) ||
                !getDblArr("stage_fsqz", st_fsqz.data()) ||
                !getDblArr("stage_fsql", st_fsql.data())) {
                return fail("stage history read failed");
            }
            if (nrestarts > 0 &&
                !getIntArr("restart_iteration", rst_iter.data())) {
                return fail("restart history read failed");
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
        nc_close(ncid);
        return snapshot;
    }
};

}  // namespace

std::unique_ptr<Writer> make_netcdf_v0_writer() {
    return std::make_unique<NetcdfV0Writer>();
}
std::unique_ptr<Writer> make_netcdf_v1_writer() {
    return std::make_unique<NetcdfV1Writer>();
}
std::unique_ptr<Reader> make_netcdf_v1_reader() {
    return std::make_unique<NetcdfV1Reader>();
}

}  // namespace cumes
