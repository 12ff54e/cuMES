// netcdf_writer.cpp — host-only NetCDF state adapters (completion plan steps
// 2.2/2.3).
//
// Schema v1 only, no CUDA: active dimensions + complete provenance —
// precision/status/total iterations, build/input/runtime provenance strings,
// source hash, per-stage outcomes (ns/iterations/converged/residuals), the
// full restart history, and the raw boundary harmonics + folded boundary
// matrices (distinguished, as the blueprint requires). The reader round-trips
// the state AND the complete RunReport.
//
// This TU (and only this TU) includes <netcdf.h>; it compiles under
// CUMES_HAVE_NETCDF, which is confined to the adapter library target.
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "internal_factories.hpp"
#include "io_common.hpp"
#include "cumes/io/writer_helpers.hpp"

#include <netcdf.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace cumes {
namespace {

bool putStrAttr(int ncid, const char* name, const std::string& value) {
    return nc_put_att_text(ncid, NC_GLOBAL, name, value.size(), value.c_str()) ==
           NC_NOERR;
}


// ---------------------------------------------------------------------------
// v1: active dimensions + complete provenance + boundary harmonics
// ---------------------------------------------------------------------------
class NetcdfV1Writer final : public Writer {
 public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report, const OutputSpec& spec,
                        const ValidatedProblem& problem) override {
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
        const InputParams& ip = report.input_params;

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
        // n_-prefixed: NetCDF classic shares ONE namespace for dimensions
        // and variables, so the folded-boundary dims must not collide with
        // the embedded-input scalar variable names (mpol, ntor, ...).
        NC_CHECK(nc_def_dim(ncid, "n_mpol", (size_t)mpol, &dim_mpol),
                 "def n_mpol");
        NC_CHECK(nc_def_dim(ncid, "n_ntorp1", (size_t)ntorp1, &dim_ntorp1),
                 "def n_ntorp1");
        // ---- embedded normalized-input record dimensions ----
        // Empty vectors get NO dimension: classic NetCDF gives a 0-length
        // dimension unlimited semantics, and only ONE unlimited dimension may
        // exist (a second 0-length def fails with NC_UNLIMITED size already
        // in use). The reader treats an absent array variable as an empty
        // vector.
        int dim_nam = -1, dim_nac = -1, dim_nai = -1, dim_naphi = -1,
            dim_nraxis = -1, dim_nzaxis = -1, dim_nstages_in = -1;
        if (!ip.am.empty()) {
            NC_CHECK(nc_def_dim(ncid, "n_am", ip.am.size(), &dim_nam),
                     "def n_am");
        }
        if (!ip.ac.empty()) {
            NC_CHECK(nc_def_dim(ncid, "n_ac", ip.ac.size(), &dim_nac),
                     "def n_ac");
        }
        if (!ip.ai.empty()) {
            NC_CHECK(nc_def_dim(ncid, "n_ai", ip.ai.size(), &dim_nai),
                     "def n_ai");
        }
        if (!ip.aphi.empty()) {
            NC_CHECK(nc_def_dim(ncid, "n_aphi", ip.aphi.size(), &dim_naphi),
                     "def n_aphi");
        }
        if (!ip.raxis_c.empty()) {
            NC_CHECK(nc_def_dim(ncid, "n_raxis", ip.raxis_c.size(),
                                &dim_nraxis), "def n_raxis");
        }
        if (!ip.zaxis_s.empty()) {
            NC_CHECK(nc_def_dim(ncid, "n_zaxis", ip.zaxis_s.size(),
                                &dim_nzaxis), "def n_zaxis");
        }
        if (!ip.stages.empty()) {
            NC_CHECK(nc_def_dim(ncid, "nstages_in", ip.stages.size(),
                                &dim_nstages_in), "def nstages_in");
        }

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

        // ---- embedded normalized-input record (scalars + arrays) ----
        int v_mpol, v_ntor, v_nfp, v_ntheta, v_nzeta, v_ncurr;
        NC_CHECK(nc_def_var(ncid, "mpol", NC_INT, 0, nullptr, &v_mpol),
                 "def mpol");
        NC_CHECK(nc_def_var(ncid, "ntor", NC_INT, 0, nullptr, &v_ntor),
                 "def ntor");
        NC_CHECK(nc_def_var(ncid, "nfp", NC_INT, 0, nullptr, &v_nfp),
                 "def nfp");
        NC_CHECK(nc_def_var(ncid, "ntheta", NC_INT, 0, nullptr, &v_ntheta),
                 "def ntheta");
        NC_CHECK(nc_def_var(ncid, "nzeta", NC_INT, 0, nullptr, &v_nzeta),
                 "def nzeta");
        NC_CHECK(nc_def_var(ncid, "ncurr", NC_INT, 0, nullptr, &v_ncurr),
                 "def ncurr");
        int v_delt, v_phiedge, v_pres_scale, v_adiabatic_index, v_spres_ped,
            v_bloat, v_curtor, v_tcon0;
        NC_CHECK(nc_def_var(ncid, "delt", NC_DOUBLE, 0, nullptr, &v_delt),
                 "def delt");
        NC_CHECK(nc_def_var(ncid, "phiedge", NC_DOUBLE, 0, nullptr, &v_phiedge),
                 "def phiedge");
        NC_CHECK(nc_def_var(ncid, "pres_scale", NC_DOUBLE, 0, nullptr,
                            &v_pres_scale), "def pres_scale");
        NC_CHECK(nc_def_var(ncid, "adiabatic_index", NC_DOUBLE, 0, nullptr,
                            &v_adiabatic_index), "def adiabatic_index");
        NC_CHECK(nc_def_var(ncid, "spres_ped", NC_DOUBLE, 0, nullptr,
                            &v_spres_ped), "def spres_ped");
        NC_CHECK(nc_def_var(ncid, "bloat", NC_DOUBLE, 0, nullptr, &v_bloat),
                 "def bloat");
        NC_CHECK(nc_def_var(ncid, "curtor", NC_DOUBLE, 0, nullptr, &v_curtor),
                 "def curtor");
        NC_CHECK(nc_def_var(ncid, "tcon0", NC_DOUBLE, 0, nullptr, &v_tcon0),
                 "def tcon0");
        int v_am = -1, v_ac = -1, v_ai = -1, v_aphi = -1, v_raxis = -1,
            v_zaxis = -1, v_stg_in_ns = -1, v_stg_max_iter = -1,
            v_stg_ftol = -1;
        if (!ip.am.empty()) {
            NC_CHECK(nc_def_var(ncid, "am", NC_DOUBLE, 1, &dim_nam, &v_am),
                     "def am");
        }
        if (!ip.ac.empty()) {
            NC_CHECK(nc_def_var(ncid, "ac", NC_DOUBLE, 1, &dim_nac, &v_ac),
                     "def ac");
        }
        if (!ip.ai.empty()) {
            NC_CHECK(nc_def_var(ncid, "ai", NC_DOUBLE, 1, &dim_nai, &v_ai),
                     "def ai");
        }
        if (!ip.aphi.empty()) {
            NC_CHECK(nc_def_var(ncid, "aphi", NC_DOUBLE, 1, &dim_naphi,
                                &v_aphi), "def aphi");
        }
        if (!ip.raxis_c.empty()) {
            NC_CHECK(nc_def_var(ncid, "raxis_c", NC_DOUBLE, 1, &dim_nraxis,
                                &v_raxis), "def raxis_c");
        }
        if (!ip.zaxis_s.empty()) {
            NC_CHECK(nc_def_var(ncid, "zaxis_s", NC_DOUBLE, 1, &dim_nzaxis,
                                &v_zaxis), "def zaxis_s");
        }
        if (!ip.stages.empty()) {
            NC_CHECK(nc_def_var(ncid, "stage_in_ns", NC_INT, 1, &dim_nstages_in,
                                &v_stg_in_ns), "def stage_in_ns");
            NC_CHECK(nc_def_var(ncid, "stage_max_iter", NC_INT, 1,
                                &dim_nstages_in, &v_stg_max_iter),
                     "def stage_max_iter");
            NC_CHECK(nc_def_var(ncid, "stage_ftol", NC_DOUBLE, 1,
                                &dim_nstages_in, &v_stg_ftol),
                     "def stage_ftol");
        }

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
        NC_CHECK(putStrAttr(ncid, "schema", ip.schema) == true
                     ? NC_NOERR : NC_EATTMETA, "attr schema");

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
        // ---- embedded normalized-input record data ----
        NC_CHECK(nc_put_var_int(ncid, v_mpol, &ip.mpol), "put mpol");
        NC_CHECK(nc_put_var_int(ncid, v_ntor, &ip.ntor), "put ntor");
        NC_CHECK(nc_put_var_int(ncid, v_nfp, &ip.nfp), "put nfp");
        NC_CHECK(nc_put_var_int(ncid, v_ntheta, &ip.ntheta), "put ntheta");
        NC_CHECK(nc_put_var_int(ncid, v_nzeta, &ip.nzeta), "put nzeta");
        NC_CHECK(nc_put_var_int(ncid, v_ncurr, &ip.ncurr), "put ncurr");
        NC_CHECK(nc_put_var_double(ncid, v_delt, &ip.delt), "put delt");
        NC_CHECK(nc_put_var_double(ncid, v_phiedge, &ip.phiedge), "put phiedge");
        NC_CHECK(nc_put_var_double(ncid, v_pres_scale, &ip.pres_scale),
                 "put pres_scale");
        NC_CHECK(nc_put_var_double(ncid, v_adiabatic_index, &ip.adiabatic_index),
                 "put adiabatic_index");
        NC_CHECK(nc_put_var_double(ncid, v_spres_ped, &ip.spres_ped),
                 "put spres_ped");
        NC_CHECK(nc_put_var_double(ncid, v_bloat, &ip.bloat), "put bloat");
        NC_CHECK(nc_put_var_double(ncid, v_curtor, &ip.curtor), "put curtor");
        NC_CHECK(nc_put_var_double(ncid, v_tcon0, &ip.tcon0), "put tcon0");
        if (!ip.am.empty()) {
            NC_CHECK(nc_put_var_double(ncid, v_am, ip.am.data()), "put am");
        }
        if (!ip.ac.empty()) {
            NC_CHECK(nc_put_var_double(ncid, v_ac, ip.ac.data()), "put ac");
        }
        if (!ip.ai.empty()) {
            NC_CHECK(nc_put_var_double(ncid, v_ai, ip.ai.data()), "put ai");
        }
        if (!ip.aphi.empty()) {
            NC_CHECK(nc_put_var_double(ncid, v_aphi, ip.aphi.data()), "put aphi");
        }
        if (!ip.raxis_c.empty()) {
            NC_CHECK(nc_put_var_double(ncid, v_raxis, ip.raxis_c.data()),
                     "put raxis_c");
        }
        if (!ip.zaxis_s.empty()) {
            NC_CHECK(nc_put_var_double(ncid, v_zaxis, ip.zaxis_s.data()),
                     "put zaxis_s");
        }
        if (!ip.stages.empty()) {
            const size_t nstages_in = ip.stages.size();
            std::vector<int> stg_in_ns(nstages_in), stg_max_iter(nstages_in);
            std::vector<double> stg_ftol(nstages_in);
            for (size_t g = 0; g < nstages_in; ++g) {
                stg_in_ns[g] = ip.stages[g].ns;
                stg_max_iter[g] = ip.stages[g].max_iter;
                stg_ftol[g] = ip.stages[g].ftol;
            }
            NC_CHECK(nc_put_var_int(ncid, v_stg_in_ns, stg_in_ns.data()),
                     "put stage_in_ns");
            NC_CHECK(nc_put_var_int(ncid, v_stg_max_iter, stg_max_iter.data()),
                     "put stage_max_iter");
            NC_CHECK(nc_put_var_double(ncid, v_stg_ftol, stg_ftol.data()),
                     "put stage_ftol");
        }
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
        // Durable publication (completion-plan follow-up §3): the library
        // owns the descriptor, so publishLibraryFile reopens the completed
        // temp, checks fsync + close, then renames + directory-fsyncs.
        const std::string err = io_detail::publishLibraryFile(tmp, spec.path);
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
        // Allocation failures in a malformed-container read must become typed
        // errors, never exceptions across the Reader interface
        // (reader-rank-hardening handoff §4).
        try {
        auto getDim = [&](const char* name, int& id_out, size_t& len_out) -> bool {
            int id = -1;
            return nc_inq_dimid(ncid, name, &id) == NC_NOERR &&
                   nc_inq_dimlen(ncid, id, &len_out) == NC_NOERR &&
                   (id_out = id, true);
        };
        // ---- exact-rank/type/extent helpers (reader-rank-hardening §2) -----
        // A reader of an untrusted container must never assume the rank or
        // shape implied by a variable NAME: every read into a fixed or sized
        // host buffer is preceded by an exact rank, exact datatype, and
        // exact extent/dimension-ID check.
        auto varHasType = [&](int vid, nc_type want) -> bool {
            nc_type t = NC_NAT;
            return nc_inq_vartype(ncid, vid, &t) == NC_NOERR && t == want;
        };
        // Scalar int: rank exactly 0 + exact NC_INT, then one value.
        auto readScalarInt = [&](const char* name, int& out) -> bool {
            int vid = -1, ndims = 0;
            if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
            if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR) return false;
            if (ndims != 0) return false;
            if (!varHasType(vid, NC_INT)) return false;
            return nc_get_var_int(ncid, vid, &out) == NC_NOERR;
        };
        // 1-D vector: rank exactly 1 (BEFORE asking for the dimension-ID
        // list — nc_inq_vardimid writes one ID per dimension), the variable
        // must live on the EXPECTED dimension id with the exact extent, and
        // the datatype must be exact. The read itself uses an explicit
        // start/count selection so it can never consume more than the
        // checked destination size.
        auto readVectorInt = [&](const char* name, int expect_dim,
                                 size_t expect, std::vector<int>& out) -> bool {
            int vid = -1, ndims = 0, dimid = -1;
            if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
            if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR) return false;
            if (ndims != 1) return false;
            if (nc_inq_vardimid(ncid, vid, &dimid) != NC_NOERR) return false;
            if (dimid != expect_dim) return false;
            size_t len = 0;
            if (nc_inq_dimlen(ncid, dimid, &len) != NC_NOERR) return false;
            if (len != expect) return false;
            if (!varHasType(vid, NC_INT)) return false;
            const auto bytes = checked_mul(expect, sizeof(int));
            if (!bytes) return false;
            out.resize(expect);
            const size_t start[1] = {0}, count[1] = {expect};
            return nc_get_vara_int(ncid, vid, start, count, out.data()) ==
                   NC_NOERR;
        };
        // Scalar double: rank exactly 0 + exact NC_DOUBLE, then one value.
        auto readScalarDouble = [&](const char* name, double& out) -> bool {
            int vid = -1, ndims = 0;
            if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
            if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR) return false;
            if (ndims != 0) return false;
            if (!varHasType(vid, NC_DOUBLE)) return false;
            return nc_get_var_double(ncid, vid, &out) == NC_NOERR;
        };
        // 2-D double matrix: rank exactly 2, dimension IDs exactly
        // [dim0, dim1], exact NC_DOUBLE, then one bounded whole-matrix read
        // in the writer's C order (row stride = dim1 extent).
        auto readMatrixDouble = [&](const char* name, int dim0, int dim1,
                                    size_t n0, size_t n1,
                                    std::vector<double>& out) -> bool {
            int vid = -1, ndims = 0, dimids[2] = {-1, -1};
            if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
            if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR) return false;
            if (ndims != 2) return false;
            if (nc_inq_vardimid(ncid, vid, dimids) != NC_NOERR) return false;
            if (dimids[0] != dim0 || dimids[1] != dim1) return false;
            if (!varHasType(vid, NC_DOUBLE)) return false;
            const auto el = checked_mul(n0, n1);
            if (!el) return false;
            const auto bytes = checked_mul(*el, sizeof(double));
            if (!bytes) return false;
            out.resize(*el);
            const size_t start[2] = {0, 0}, count[2] = {n0, n1};
            return nc_get_vara_double(ncid, vid, start, count, out.data()) ==
                   NC_NOERR;
        };
        auto readVectorDouble = [&](const char* name, int expect_dim,
                                    size_t expect,
                                    std::vector<double>& out) -> bool {
            int vid = -1, ndims = 0, dimid = -1;
            if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
            if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR) return false;
            if (ndims != 1) return false;
            if (nc_inq_vardimid(ncid, vid, &dimid) != NC_NOERR) return false;
            if (dimid != expect_dim) return false;
            size_t len = 0;
            if (nc_inq_dimlen(ncid, dimid, &len) != NC_NOERR) return false;
            if (len != expect) return false;
            if (!varHasType(vid, NC_DOUBLE)) return false;
            const auto bytes = checked_mul(expect, sizeof(double));
            if (!bytes) return false;
            out.resize(expect);
            const size_t start[1] = {0}, count[1] = {expect};
            return nc_get_vara_double(ncid, vid, start, count, out.data()) ==
                   NC_NOERR;
        };
        // 2-D state family: rank exactly 2, dimension IDs exactly
        // [ns_dim, mnmax_dim] (so the extents are the named dimensions by
        // construction), exact NC_DOUBLE, then bounded per-surface reads.
        auto readStateFamily = [&](const char* name, int ns_dim, int mnmax_dim,
                                   size_t ns, int mnmax,
                                   std::vector<double>& fam) -> bool {
            int vid = -1, ndims = 0, dimids[2] = {-1, -1};
            if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
            if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR) return false;
            if (ndims != 2) return false;
            if (nc_inq_vardimid(ncid, vid, dimids) != NC_NOERR) return false;
            if (dimids[0] != ns_dim || dimids[1] != mnmax_dim) return false;
            if (!varHasType(vid, NC_DOUBLE)) return false;
            for (int m = 0; m < mnmax; ++m) {
                const size_t start[2] = {0, (size_t)m};
                const size_t count[2] = {ns, 1};
                if (nc_get_vara_double(ncid, vid, start, count,
                                       fam.data() + (size_t)m * ns) !=
                    NC_NOERR) {
                    return false;
                }
            }
            return true;
        };

        int ns_dim = -1, mnmax_dim = -1;
        size_t ns = 0, mnmax = 0;
        if (!getDim("ns", ns_dim, ns) || !getDim("mnmax", mnmax_dim, mnmax)) {
            return fail("missing state dimensions");
        }
        // Bounds BEFORE narrowing and allocation (handoff §4): dimensions
        // must fit the EquilibriumSnapshot int fields, and the product must
        // pass the checked multiplication.
        if (ns < 1 || mnmax < 1 || ns > INT_MAX || mnmax > INT_MAX) {
            return fail("bad dimensions (ns=" + std::to_string(ns) +
                        ", mnmax=" + std::to_string(mnmax) + ")");
        }
        std::size_t n = 0;
        {
            const auto n_opt = checked_mul(ns, mnmax);
            if (!n_opt) return fail("dimension product overflows size_t");
            if (*n_opt > cumes::io_detail::kMaxStateElementsPerFamily) {
                return fail("state dimensions exceed the resource cap");
            }
            const auto bytes = checked_mul(*n_opt, sizeof(double));
            if (!bytes) return fail("state byte count overflows size_t");
            n = *n_opt;
        }
        EquilibriumSnapshot snapshot;
        snapshot.ns = static_cast<int>(ns);
        snapshot.mnmax = static_cast<int>(mnmax);
        const char* fam_names[6] = {"rmncc", "zmnsc", "lmnsc",
                                    "rmnss", "zmncs", "lmncs"};
        for (int c = 0; c < 6; ++c) {
            snapshot.families[c].resize(n);
            if (!readStateFamily(fam_names[c], ns_dim, mnmax_dim, ns,
                                 snapshot.mnmax, snapshot.families[c])) {
                return fail("state read failed (missing/malformed " +
                            std::string(fam_names[c]) + ")");
            }
        }
        if (report) {
            // Parse transactionally: a malformed late field must not leave a
            // partially populated report visible to the caller.
            RunReport parsed_report;
            auto getStr = [&](const char* name, std::string& out) -> bool {
                size_t len = 0;
                if (nc_inq_attlen(ncid, NC_GLOBAL, name, &len) != NC_NOERR) {
                    return false;
                }
                // Documented resource cap (handoff §4): a hostile length must
                // fail before the host allocates it.
                if (len > cumes::io_detail::kMaxProvenanceStringBytes) {
                    return false;
                }
                out.resize(len);
                // Zero-length attributes (e.g. an empty compile_flags) are
                // legal; skip the read instead of forming &out[0] on an
                // empty string (completion-plan follow-up §2.2).
                if (len == 0) return true;
                return nc_get_att_text(ncid, NC_GLOBAL, name, &out[0]) ==
                       NC_NOERR;
            };
            int precision = 0, status = 0, total = 0, dirty = 0;
            if (!readScalarInt("precision", precision) ||
                !readScalarInt("status", status) ||
                !readScalarInt("total_iterations", total) ||
                !readScalarInt("build_dirty", dirty)) {
                return fail("missing run outcome variables");
            }
            // Closed-range validation of the serialized scalars (handoff §4).
            if (precision < 0 || precision > 1) {
                return fail("invalid precision scalar");
            }
            if (status < 0 || status > 4) {
                return fail("invalid run status scalar");
            }
            if (total < 0) {
                return fail("negative total iteration count");
            }
            if (dirty != 0 && dirty != 1) {
                return fail("invalid build_dirty scalar");
            }
            parsed_report.status = static_cast<RunStatus>(status);
            parsed_report.total_effective_iterations = total;
            parsed_report.build.dirty = (dirty != 0);
            parsed_report.build.scalar_type =
                (precision == 0) ? "double" : "float";
            if (!getStr("revision", parsed_report.build.revision) ||
                !getStr("build_type", parsed_report.build.build_type) ||
                !getStr("precision_policy",
                        parsed_report.build.precision_policy) ||
                !getStr("compile_flags", parsed_report.build.compile_flags) ||
                !getStr("source_path", parsed_report.input.source_path) ||
                !getStr("source_hash", parsed_report.input.source_hash) ||
                !getStr("gpu_name", parsed_report.runtime.gpu_name) ||
                !getStr("driver", parsed_report.runtime.driver) ||
                !getStr("runtime", parsed_report.runtime.runtime) ||
                !getStr("toolkit", parsed_report.runtime.toolkit)) {
                return fail("missing provenance attributes");
            }
            int nstages_dim = -1, nrestarts_dim = -1;
            size_t nstages = 0, nrestarts = 0;
            if (!getDim("nstages", nstages_dim, nstages) ||
                !getDim("nrestarts", nrestarts_dim, nrestarts)) {
                return fail("missing stage dimensions");
            }
            // Documented stage/restart resource caps (handoff §4): a hostile
            // dimension must fail before the host allocates it.
            if (nstages > cumes::io_detail::kMaxStageCount ||
                nrestarts > cumes::io_detail::kMaxStageCount) {
                return fail("stage/restart dimensions exceed the resource cap");
            }
            std::vector<int> stage_ns(nstages), stage_iter(nstages),
                stage_conv(nstages), rst_off(nstages);
            std::vector<double> st_fsqr(nstages), st_fsqz(nstages),
                st_fsql(nstages);
            std::vector<int> rst_iter(nrestarts);
            if (!readVectorInt("stage_ns", nstages_dim, nstages, stage_ns) ||
                !readVectorInt("stage_iterations", nstages_dim, nstages,
                               stage_iter) ||
                !readVectorInt("stage_converged", nstages_dim, nstages,
                               stage_conv) ||
                !readVectorInt("restart_stage_offset", nstages_dim, nstages,
                               rst_off) ||
                !readVectorDouble("stage_fsqr", nstages_dim, nstages,
                                  st_fsqr) ||
                !readVectorDouble("stage_fsqz", nstages_dim, nstages,
                                  st_fsqz) ||
                !readVectorDouble("stage_fsql", nstages_dim, nstages,
                                  st_fsql)) {
                return fail("stage history read failed");
            }
            // restart_iteration must exist whenever nrestarts > 0, and must
            // pass the same rank/type/extent checks whenever it exists.
            {
                int vid = -1;
                if (nc_inq_varid(ncid, "restart_iteration", &vid) == NC_NOERR) {
                    if (!readVectorInt("restart_iteration", nrestarts_dim,
                                       nrestarts, rst_iter)) {
                        return fail("restart history read failed");
                    }
                } else if (nrestarts > 0) {
                    return fail("restart history read failed");
                }
            }
            {
                const std::string off_err =
                    validateRestartOffsets(rst_off, nstages, nrestarts);
                if (!off_err.empty()) {
                    return fail("invalid restart offsets: " + off_err);
                }
            }
            for (size_t g = 0; g < nstages; ++g) {
                if (stage_ns[g] <= 0) {
                    return fail("nonpositive stage surface count");
                }
                if (stage_iter[g] < 0) {
                    return fail("negative stage iteration count");
                }
                if (stage_conv[g] != 0 && stage_conv[g] != 1) {
                    return fail("invalid stage_converged value");
                }
            }
            for (int iteration : rst_iter) {
                if (iteration < 0) {
                    return fail("negative restart iteration");
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
                parsed_report.stages.push_back(std::move(st));
            }
            // ---- embedded normalized-input record ----
            // All-or-nothing: containers written before the record carry none
            // of its fields (the record stays default-empty); when the first
            // scalar exists, the FULL record is required and every read gets
            // the exact-rank/type/extent hardening above.
            {
                int probe = -1;
                if (nc_inq_varid(ncid, "mpol", &probe) == NC_NOERR) {
                    InputParams ip;
                    if (!readScalarInt("mpol", ip.mpol) ||
                        !readScalarInt("ntor", ip.ntor) ||
                        !readScalarInt("nfp", ip.nfp) ||
                        !readScalarInt("ntheta", ip.ntheta) ||
                        !readScalarInt("nzeta", ip.nzeta) ||
                        !readScalarInt("ncurr", ip.ncurr) ||
                        !readScalarDouble("delt", ip.delt) ||
                        !readScalarDouble("phiedge", ip.phiedge) ||
                        !readScalarDouble("pres_scale", ip.pres_scale) ||
                        !readScalarDouble("adiabatic_index",
                                          ip.adiabatic_index) ||
                        !readScalarDouble("spres_ped", ip.spres_ped) ||
                        !readScalarDouble("bloat", ip.bloat) ||
                        !readScalarDouble("curtor", ip.curtor) ||
                        !readScalarDouble("tcon0", ip.tcon0)) {
                        return fail("malformed embedded input record");
                    }
                    // Arrays: an ABSENT variable means an empty vector (the
                    // writer omits empty arrays to avoid 0-length dims); a
                    // present one must pass the strict read.
                    {
                        auto readOptional = [&](const char* dname,
                                                const char* vname,
                                                std::vector<double>& out)
                            -> bool {
                            int vid = -1;
                            if (nc_inq_varid(ncid, vname, &vid) != NC_NOERR) {
                                return true;  // absent -> empty
                            }
                            int dim = -1;
                            size_t n = 0;
                            if (!getDim(dname, dim, n)) return false;
                            return readVectorDouble(vname, dim, n, out);
                        };
                        if (!readOptional("n_am", "am", ip.am) ||
                            !readOptional("n_ac", "ac", ip.ac) ||
                            !readOptional("n_ai", "ai", ip.ai) ||
                            !readOptional("n_aphi", "aphi", ip.aphi) ||
                            !readOptional("n_raxis", "raxis_c", ip.raxis_c) ||
                            !readOptional("n_zaxis", "zaxis_s", ip.zaxis_s)) {
                            return fail("malformed embedded input record");
                        }
                    }
                    // The input stage arrays are optional as a whole (the
                    // writer omits them for an empty stage list).
                    {
                        int vid = -1;
                        if (nc_inq_varid(ncid, "stage_in_ns", &vid) ==
                            NC_NOERR) {
                            int dim_nstages_in = -1;
                            size_t nstages_in = 0;
                            if (!getDim("nstages_in", dim_nstages_in,
                                        nstages_in)) {
                                return fail("malformed embedded input record");
                            }
                            if (nstages_in >
                                cumes::io_detail::kMaxStageCount) {
                                return fail("embedded input stage count "
                                            "exceeds the resource cap");
                            }
                            std::vector<int> stg_in_ns(nstages_in),
                                stg_max_iter(nstages_in);
                            std::vector<double> stg_ftol(nstages_in);
                            if (!readVectorInt("stage_in_ns", dim_nstages_in,
                                               nstages_in, stg_in_ns) ||
                                !readVectorInt("stage_max_iter",
                                               dim_nstages_in, nstages_in,
                                               stg_max_iter) ||
                                !readVectorDouble("stage_ftol",
                                                  dim_nstages_in, nstages_in,
                                                  stg_ftol)) {
                                return fail("malformed embedded input record");
                            }
                            for (size_t g = 0; g < nstages_in; ++g) {
                                InputStage st;
                                st.ns = stg_in_ns[g];
                                st.max_iter = stg_max_iter[g];
                                st.ftol = stg_ftol[g];
                                ip.stages.push_back(st);
                            }
                        }
                    }
                    // Boundary + folded (always part of a present record).
                    int dim_nrbc = -1, dim_nzbs = -1, dim_mpol = -1,
                        dim_ntorp1 = -1;
                    size_t nrbc = 0, nzbs = 0, mpol = 0, ntorp1 = 0;
                    if (!getDim("nrbc", dim_nrbc, nrbc) ||
                        !getDim("nzbs", dim_nzbs, nzbs) ||
                        !getDim("n_mpol", dim_mpol, mpol) ||
                        !getDim("n_ntorp1", dim_ntorp1, ntorp1)) {
                        return fail("malformed embedded input record");
                    }
                    if (!readVectorInt("rbc_m", dim_nrbc, nrbc, ip.rbc_m) ||
                        !readVectorInt("rbc_n", dim_nrbc, nrbc, ip.rbc_n) ||
                        !readVectorDouble("rbc_value", dim_nrbc, nrbc,
                                          ip.rbc_value) ||
                        !readVectorInt("zbs_m", dim_nzbs, nzbs, ip.zbs_m) ||
                        !readVectorInt("zbs_n", dim_nzbs, nzbs, ip.zbs_n) ||
                        !readVectorDouble("zbs_value", dim_nzbs, nzbs,
                                          ip.zbs_value) ||
                        !readMatrixDouble("rbcc", dim_mpol, dim_ntorp1, mpol,
                                          ntorp1, ip.rbcc) ||
                        !readMatrixDouble("rbss", dim_mpol, dim_ntorp1, mpol,
                                          ntorp1, ip.rbss) ||
                        !readMatrixDouble("zbsc", dim_mpol, dim_ntorp1, mpol,
                                          ntorp1, ip.zbsc) ||
                        !readMatrixDouble("zbcs", dim_mpol, dim_ntorp1, mpol,
                                          ntorp1, ip.zbcs)) {
                        return fail("malformed embedded input record");
                    }
                    // The schema tag is informational; an absent or unreadable
                    // one keeps the default.
                    std::string schema_tag;
                    if (getStr("schema", schema_tag)) ip.schema = schema_tag;
                    parsed_report.input_params = std::move(ip);
                }
            }
            *report = std::move(parsed_report);
        }
        nc_close(ncid);
        return snapshot;
        } catch (const std::bad_alloc&) {
            nc_close(ncid);
            return Result<EquilibriumSnapshot>(
                "NetCDF: allocation failed (dimensions implausible or corrupt)");
        } catch (const std::length_error&) {
            nc_close(ncid);
            return Result<EquilibriumSnapshot>(
                "NetCDF: allocation failed (dimensions implausible or corrupt)");
        }
    }
};

}  // namespace

std::unique_ptr<Writer> make_netcdf_v1_writer() {
    return std::make_unique<NetcdfV1Writer>();
}
std::unique_ptr<Reader> make_netcdf_v1_reader() {
    return std::make_unique<NetcdfV1Reader>();
}

}  // namespace cumes
