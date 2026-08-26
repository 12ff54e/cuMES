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
#include "cumes/io/writer_helpers.hpp"
#include "internal_factories.hpp"
#include "io_common.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <netcdf.h>

namespace cumes {
namespace {

bool put_str_attr(int ncid, const char* name, const std::string& value) {
    return nc_put_att_text(ncid, NC_GLOBAL, name, value.size(),
                           value.c_str()) == NC_NOERR;
}

// ---------------------------------------------------------------------------
// v1: active dimensions + complete provenance + boundary harmonics
// ---------------------------------------------------------------------------
class NetcdfV1Writer final : public Writer {
   public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report,
                        const OutputSpec& spec,
                        const ValidatedProblem& problem) override {
        const FoldedBoundary& fb = problem.boundary();
        const int mpol = problem.shape().mpol;
        const int ntorp1 = problem.shape().ntor + 1;
        const std::vector<BoundaryHarmonic>& rbc = problem.spec().rbc;
        const std::vector<BoundaryHarmonic>& zbs = problem.spec().zbs;

        const std::string tmp = io_detail::temp_path_for(spec.path);
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
#define NC_CHECK(rc_, msg_)                                           \
    do {                                                              \
        int _rc = (rc_);                                              \
        if (_rc != NC_NOERR) {                                        \
            return fail(std::string(msg_) + ": " + nc_strerror(_rc)); \
        }                                                             \
    } while (0)

        const size_t nstages = report.stages.size();
        size_t nrestarts = 0;
        for (const auto& st : report.stages) nrestarts += st.restarts.size();
        const InputParams& ip = report.input_params;

        // ---- active dimensions ----
        int dim_ns, dim_mnmax, dim_nstages, dim_nrestarts, dim_nrbc, dim_nzbs,
            dim_mpol, dim_ntorp1;
        NC_CHECK(nc_def_dim(ncid, "ns", (size_t)snapshot.ns, &dim_ns),
                 "def ns");
        NC_CHECK(nc_def_dim(ncid, "mnmax", (size_t)snapshot.mnmax, &dim_mnmax),
                 "def mnmax");
        NC_CHECK(nc_def_dim(ncid, "nstages", nstages, &dim_nstages),
                 "def nstages");
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
            dim_nraxis = -1, dim_nzaxis = -1, dim_nstages_in = -1,
            dim_nextcur = -1;
        if (!ip.am.empty()) {
            NC_CHECK(nc_def_dim(ncid, "n_am", ip.am.size(), &dim_nam),
                     "def n_am");
        }
        if (!ip.extcur.empty()) {
            NC_CHECK(
                nc_def_dim(ncid, "n_extcur", ip.extcur.size(), &dim_nextcur),
                "def n_extcur");
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
            NC_CHECK(
                nc_def_dim(ncid, "n_raxis", ip.raxis_c.size(), &dim_nraxis),
                "def n_raxis");
        }
        if (!ip.zaxis_s.empty()) {
            NC_CHECK(
                nc_def_dim(ncid, "n_zaxis", ip.zaxis_s.size(), &dim_nzaxis),
                "def n_zaxis");
        }
        if (!ip.stages.empty()) {
            NC_CHECK(nc_def_dim(ncid, "nstages_in", ip.stages.size(),
                                &dim_nstages_in),
                     "def nstages_in");
        }

        // ---- state variables ----
        const int state_dims[2] = {dim_ns, dim_mnmax};
        const char* fam_names[6] = {"rmncc", "zmnsc", "lmnsc",
                                    "rmnss", "zmncs", "lmncs"};
        int v_fam[6];
        for (int c = 0; c < 6; ++c) {
            NC_CHECK(nc_def_var(ncid, fam_names[c], NC_DOUBLE, 2, state_dims,
                                &v_fam[c]),
                     "def family");
        }

        // ---- run outcome + stage history ----
        int v_precision, v_status, v_total_iter, v_dirty;
        NC_CHECK(
            nc_def_var(ncid, "precision", NC_INT, 0, nullptr, &v_precision),
            "def precision");
        NC_CHECK(nc_def_var(ncid, "status", NC_INT, 0, nullptr, &v_status),
                 "def status");
        NC_CHECK(nc_def_var(ncid, "total_iterations", NC_INT, 0, nullptr,
                            &v_total_iter),
                 "def total_iterations");
        NC_CHECK(nc_def_var(ncid, "build_dirty", NC_INT, 0, nullptr, &v_dirty),
                 "def build_dirty");
        int v_ns_array, v_iter_array, v_conv, v_fsqr, v_fsqz, v_fsql, v_rst_off,
            v_rst_iter;
        NC_CHECK(
            nc_def_var(ncid, "stage_ns", NC_INT, 1, &dim_nstages, &v_ns_array),
            "def stage_ns");
        NC_CHECK(nc_def_var(ncid, "stage_iterations", NC_INT, 1, &dim_nstages,
                            &v_iter_array),
                 "def stage_iterations");
        NC_CHECK(nc_def_var(ncid, "stage_converged", NC_INT, 1, &dim_nstages,
                            &v_conv),
                 "def stage_converged");
        NC_CHECK(
            nc_def_var(ncid, "stage_fsqr", NC_DOUBLE, 1, &dim_nstages, &v_fsqr),
            "def stage_fsqr");
        NC_CHECK(
            nc_def_var(ncid, "stage_fsqz", NC_DOUBLE, 1, &dim_nstages, &v_fsqz),
            "def stage_fsqz");
        NC_CHECK(
            nc_def_var(ncid, "stage_fsql", NC_DOUBLE, 1, &dim_nstages, &v_fsql),
            "def stage_fsql");
        NC_CHECK(nc_def_var(ncid, "restart_stage_offset", NC_INT, 1,
                            &dim_nstages, &v_rst_off),
                 "def restart_stage_offset");
        NC_CHECK(nc_def_var(ncid, "restart_iteration", NC_INT, 1,
                            &dim_nrestarts, &v_rst_iter),
                 "def restart_iteration");

        // ---- raw boundary harmonics (distinguished from the folded basis)
        // ----
        int v_rbc_m, v_rbc_n, v_rbc_v, v_zbs_m, v_zbs_n, v_zbs_v;
        NC_CHECK(nc_def_var(ncid, "rbc_m", NC_INT, 1, &dim_nrbc, &v_rbc_m),
                 "def rbc_m");
        NC_CHECK(nc_def_var(ncid, "rbc_n", NC_INT, 1, &dim_nrbc, &v_rbc_n),
                 "def rbc_n");
        NC_CHECK(
            nc_def_var(ncid, "rbc_value", NC_DOUBLE, 1, &dim_nrbc, &v_rbc_v),
            "def rbc_value");
        NC_CHECK(nc_def_var(ncid, "zbs_m", NC_INT, 1, &dim_nzbs, &v_zbs_m),
                 "def zbs_m");
        NC_CHECK(nc_def_var(ncid, "zbs_n", NC_INT, 1, &dim_nzbs, &v_zbs_n),
                 "def zbs_n");
        NC_CHECK(
            nc_def_var(ncid, "zbs_value", NC_DOUBLE, 1, &dim_nzbs, &v_zbs_v),
            "def zbs_value");

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
                            &v_pres_scale),
                 "def pres_scale");
        NC_CHECK(nc_def_var(ncid, "adiabatic_index", NC_DOUBLE, 0, nullptr,
                            &v_adiabatic_index),
                 "def adiabatic_index");
        NC_CHECK(
            nc_def_var(ncid, "spres_ped", NC_DOUBLE, 0, nullptr, &v_spres_ped),
            "def spres_ped");
        NC_CHECK(nc_def_var(ncid, "bloat", NC_DOUBLE, 0, nullptr, &v_bloat),
                 "def bloat");
        NC_CHECK(nc_def_var(ncid, "curtor", NC_DOUBLE, 0, nullptr, &v_curtor),
                 "def curtor");
        NC_CHECK(nc_def_var(ncid, "tcon0", NC_DOUBLE, 0, nullptr, &v_tcon0),
                 "def tcon0");
        int v_lfreeb, v_nvacskip, v_makegrid_parameters_present;
        NC_CHECK(nc_def_var(ncid, "lfreeb", NC_INT, 0, nullptr, &v_lfreeb),
                 "def lfreeb");
        NC_CHECK(nc_def_var(ncid, "nvacskip", NC_INT, 0, nullptr, &v_nvacskip),
                 "def nvacskip");
        NC_CHECK(nc_def_var(ncid, "makegrid_parameters_present", NC_INT, 0,
                            nullptr, &v_makegrid_parameters_present),
                 "def makegrid_parameters_present");
        int v_mg_normalize, v_mg_symmetry, v_mg_nfp, v_mg_num_r, v_mg_num_z,
            v_mg_num_phi, v_mg_min_r, v_mg_max_r, v_mg_min_z, v_mg_max_z;
        NC_CHECK(nc_def_var(ncid, "makegrid_normalize_by_currents", NC_INT, 0,
                            nullptr, &v_mg_normalize),
                 "def makegrid_normalize_by_currents");
        NC_CHECK(nc_def_var(ncid, "makegrid_assume_stellarator_symmetry",
                            NC_INT, 0, nullptr, &v_mg_symmetry),
                 "def makegrid_assume_stellarator_symmetry");
        NC_CHECK(nc_def_var(ncid, "makegrid_number_of_field_periods", NC_INT, 0,
                            nullptr, &v_mg_nfp),
                 "def makegrid_number_of_field_periods");
        NC_CHECK(nc_def_var(ncid, "makegrid_r_grid_minimum", NC_DOUBLE, 0,
                            nullptr, &v_mg_min_r),
                 "def makegrid_r_grid_minimum");
        NC_CHECK(nc_def_var(ncid, "makegrid_r_grid_maximum", NC_DOUBLE, 0,
                            nullptr, &v_mg_max_r),
                 "def makegrid_r_grid_maximum");
        NC_CHECK(nc_def_var(ncid, "makegrid_number_of_r_grid_points", NC_INT, 0,
                            nullptr, &v_mg_num_r),
                 "def makegrid_number_of_r_grid_points");
        NC_CHECK(nc_def_var(ncid, "makegrid_z_grid_minimum", NC_DOUBLE, 0,
                            nullptr, &v_mg_min_z),
                 "def makegrid_z_grid_minimum");
        NC_CHECK(nc_def_var(ncid, "makegrid_z_grid_maximum", NC_DOUBLE, 0,
                            nullptr, &v_mg_max_z),
                 "def makegrid_z_grid_maximum");
        NC_CHECK(nc_def_var(ncid, "makegrid_number_of_z_grid_points", NC_INT, 0,
                            nullptr, &v_mg_num_z),
                 "def makegrid_number_of_z_grid_points");
        NC_CHECK(nc_def_var(ncid, "makegrid_number_of_phi_grid_points", NC_INT,
                            0, nullptr, &v_mg_num_phi),
                 "def makegrid_number_of_phi_grid_points");
        int v_am = -1, v_ac = -1, v_ai = -1, v_aphi = -1, v_raxis = -1,
            v_zaxis = -1, v_stg_in_ns = -1, v_stg_max_iter = -1,
            v_stg_ftol = -1, v_extcur = -1;
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
            NC_CHECK(
                nc_def_var(ncid, "aphi", NC_DOUBLE, 1, &dim_naphi, &v_aphi),
                "def aphi");
        }
        if (!ip.raxis_c.empty()) {
            NC_CHECK(nc_def_var(ncid, "raxis_c", NC_DOUBLE, 1, &dim_nraxis,
                                &v_raxis),
                     "def raxis_c");
        }
        if (!ip.zaxis_s.empty()) {
            NC_CHECK(nc_def_var(ncid, "zaxis_s", NC_DOUBLE, 1, &dim_nzaxis,
                                &v_zaxis),
                     "def zaxis_s");
        }
        if (!ip.extcur.empty()) {
            NC_CHECK(nc_def_var(ncid, "extcur", NC_DOUBLE, 1, &dim_nextcur,
                                &v_extcur),
                     "def extcur");
        }
        if (!ip.stages.empty()) {
            NC_CHECK(nc_def_var(ncid, "stage_in_ns", NC_INT, 1, &dim_nstages_in,
                                &v_stg_in_ns),
                     "def stage_in_ns");
            NC_CHECK(nc_def_var(ncid, "stage_max_iter", NC_INT, 1,
                                &dim_nstages_in, &v_stg_max_iter),
                     "def stage_max_iter");
            NC_CHECK(nc_def_var(ncid, "stage_ftol", NC_DOUBLE, 1,
                                &dim_nstages_in, &v_stg_ftol),
                     "def stage_ftol");
        }

        // ---- provenance attributes ----
        NC_CHECK(put_str_attr(ncid, "revision", report.build.revision) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr revision");
        NC_CHECK(
            put_str_attr(ncid, "build_type", report.build.build_type) == true
                ? NC_NOERR
                : NC_EATTMETA,
            "attr build_type");
        NC_CHECK(
            put_str_attr(ncid, "scalar_type", report.build.scalar_type) == true
                ? NC_NOERR
                : NC_EATTMETA,
            "attr scalar_type");
        NC_CHECK(put_str_attr(ncid, "precision_policy",
                              report.build.precision_policy) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr precision_policy");
        NC_CHECK(put_str_attr(ncid, "compile_flags",
                              report.build.compile_flags) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr compile_flags");
        NC_CHECK(put_str_attr(ncid, "mgrid_file", ip.mgrid_file) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr mgrid_file");
        NC_CHECK(put_str_attr(ncid, "coils_file", ip.coils_file) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr coils_file");
        NC_CHECK(put_str_attr(ncid, "makegrid_parameters_file",
                              ip.makegrid_parameters_file) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr makegrid_parameters_file");
        NC_CHECK(
            put_str_attr(ncid, "source_path", report.input.source_path) == true
                ? NC_NOERR
                : NC_EATTMETA,
            "attr source_path");
        NC_CHECK(
            put_str_attr(ncid, "source_hash", report.input.source_hash) == true
                ? NC_NOERR
                : NC_EATTMETA,
            "attr source_hash");
        NC_CHECK(put_str_attr(ncid, "gpu_name", report.runtime.gpu_name) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr gpu_name");
        NC_CHECK(put_str_attr(ncid, "driver", report.runtime.driver) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr driver");
        NC_CHECK(put_str_attr(ncid, "runtime", report.runtime.runtime) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr runtime");
        NC_CHECK(put_str_attr(ncid, "toolkit", report.runtime.toolkit) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr toolkit");
        NC_CHECK(put_str_attr(ncid, "schema", ip.schema) == true ? NC_NOERR
                                                                 : NC_EATTMETA,
                 "attr schema");
        NC_CHECK(put_str_attr(ncid, "pmass_type", ip.pmass_type) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr pmass_type");
        NC_CHECK(put_str_attr(ncid, "piota_type", ip.piota_type) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr piota_type");
        NC_CHECK(put_str_attr(ncid, "pcurr_type", ip.pcurr_type) == true
                     ? NC_NOERR
                     : NC_EATTMETA,
                 "attr pcurr_type");

        NC_CHECK(nc_enddef(ncid), "nc_enddef");

        // ---- data ----
        for (int c = 0; c < 6; ++c) {
            const std::vector<double>& dbuf = snapshot.component(
                static_cast<EquilibriumSnapshot::Component>(c));
            if (dbuf.size() != snapshot.family_size()) {
                return fail("family size mismatch");
            }
            for (int m = 0; m < snapshot.mnmax; ++m) {
                const size_t start[2] = {0, (size_t)m};
                const size_t count[2] = {(size_t)snapshot.ns, 1};
                int rc2 =
                    nc_put_vara_double(ncid, v_fam[c], start, count,
                                       dbuf.data() + (size_t)m * snapshot.ns);
                if (rc2 != NC_NOERR) {
                    return fail("write family: " +
                                std::string(nc_strerror(rc2)));
                }
            }
        }
        const int precision = (report.build.scalar_type == "double") ? 0 : 1;
        NC_CHECK(nc_put_var_int(ncid, v_precision, &precision),
                 "put precision");
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
        NC_CHECK(nc_put_var_double(ncid, v_phiedge, &ip.phiedge),
                 "put phiedge");
        NC_CHECK(nc_put_var_double(ncid, v_pres_scale, &ip.pres_scale),
                 "put pres_scale");
        NC_CHECK(
            nc_put_var_double(ncid, v_adiabatic_index, &ip.adiabatic_index),
            "put adiabatic_index");
        NC_CHECK(nc_put_var_double(ncid, v_spres_ped, &ip.spres_ped),
                 "put spres_ped");
        NC_CHECK(nc_put_var_double(ncid, v_bloat, &ip.bloat), "put bloat");
        NC_CHECK(nc_put_var_double(ncid, v_curtor, &ip.curtor), "put curtor");
        NC_CHECK(nc_put_var_double(ncid, v_tcon0, &ip.tcon0), "put tcon0");
        const int lfreeb = ip.lfreeb ? 1 : 0;
        NC_CHECK(nc_put_var_int(ncid, v_lfreeb, &lfreeb), "put lfreeb");
        NC_CHECK(nc_put_var_int(ncid, v_nvacskip, &ip.nvacskip),
                 "put nvacskip");
        const int makegrid_present =
            ip.embedded_makegrid_parameters.has_value() ? 1 : 0;
        NC_CHECK(nc_put_var_int(ncid, v_makegrid_parameters_present,
                                &makegrid_present),
                 "put makegrid_parameters_present");
        const MakegridParametersSpec makegrid =
            ip.embedded_makegrid_parameters.value_or(MakegridParametersSpec{});
        const int makegrid_normalize = makegrid.normalize_by_currents ? 1 : 0;
        const int makegrid_symmetry =
            makegrid.assume_stellarator_symmetry ? 1 : 0;
        NC_CHECK(nc_put_var_int(ncid, v_mg_normalize, &makegrid_normalize),
                 "put makegrid_normalize_by_currents");
        NC_CHECK(nc_put_var_int(ncid, v_mg_symmetry, &makegrid_symmetry),
                 "put makegrid_assume_stellarator_symmetry");
        NC_CHECK(
            nc_put_var_int(ncid, v_mg_nfp, &makegrid.number_of_field_periods),
            "put makegrid_number_of_field_periods");
        NC_CHECK(nc_put_var_double(ncid, v_mg_min_r, &makegrid.r_grid_minimum),
                 "put makegrid_r_grid_minimum");
        NC_CHECK(nc_put_var_double(ncid, v_mg_max_r, &makegrid.r_grid_maximum),
                 "put makegrid_r_grid_maximum");
        NC_CHECK(
            nc_put_var_int(ncid, v_mg_num_r, &makegrid.number_of_r_grid_points),
            "put makegrid_number_of_r_grid_points");
        NC_CHECK(nc_put_var_double(ncid, v_mg_min_z, &makegrid.z_grid_minimum),
                 "put makegrid_z_grid_minimum");
        NC_CHECK(nc_put_var_double(ncid, v_mg_max_z, &makegrid.z_grid_maximum),
                 "put makegrid_z_grid_maximum");
        NC_CHECK(
            nc_put_var_int(ncid, v_mg_num_z, &makegrid.number_of_z_grid_points),
            "put makegrid_number_of_z_grid_points");
        NC_CHECK(nc_put_var_int(ncid, v_mg_num_phi,
                                &makegrid.number_of_phi_grid_points),
                 "put makegrid_number_of_phi_grid_points");
        if (!ip.extcur.empty()) {
            NC_CHECK(nc_put_var_double(ncid, v_extcur, ip.extcur.data()),
                     "put extcur");
        }
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
            NC_CHECK(nc_put_var_double(ncid, v_aphi, ip.aphi.data()),
                     "put aphi");
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
        std::vector<double> st_fsqr(nstages), st_fsqz(nstages),
            st_fsql(nstages);
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
                hm[i] = rbc[i].m;
                hn[i] = rbc[i].n;
                hv[i] = rbc[i].value;
            }
            NC_CHECK(nc_put_var_int(ncid, v_rbc_m, hm.data()), "put rbc_m");
            NC_CHECK(nc_put_var_int(ncid, v_rbc_n, hn.data()), "put rbc_n");
            NC_CHECK(nc_put_var_double(ncid, v_rbc_v, hv.data()),
                     "put rbc_value");
        }
        if (!zbs.empty()) {
            for (size_t i = 0; i < zbs.size(); ++i) {
                hm[i] = zbs[i].m;
                hn[i] = zbs[i].n;
                hv[i] = zbs[i].value;
            }
            NC_CHECK(nc_put_var_int(ncid, v_zbs_m, hm.data()), "put zbs_m");
            NC_CHECK(nc_put_var_int(ncid, v_zbs_n, hn.data()), "put zbs_n");
            NC_CHECK(nc_put_var_double(ncid, v_zbs_v, hv.data()),
                     "put zbs_value");
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
        // owns the descriptor, so publish_library_file reopens the completed
        // temp, checks fsync + close, then renames + directory-fsyncs.
        const std::string err = io_detail::publish_library_file(tmp, spec.path);
        if (!err.empty()) return Status("NetCDF publish: " + err);
        return Status();
    }
};

// ---------------------------------------------------------------------------
// v1 reader: round-trips the state AND the complete RunReport
// ---------------------------------------------------------------------------
class NetcdfV1Reader final : public Reader {
   public:
    Result<EquilibriumSnapshot> read(
        const std::string& path,
        std::optional<std::reference_wrapper<RunReport>> report) override {
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
            auto get_dim = [&](const char* name, int& id_out,
                               size_t& len_out) -> bool {
                int id = -1;
                return nc_inq_dimid(ncid, name, &id) == NC_NOERR &&
                       nc_inq_dimlen(ncid, id, &len_out) == NC_NOERR &&
                       (id_out = id, true);
            };
            // ---- exact-rank/type/extent helpers (reader-rank-hardening §2)
            // ----- A reader of an untrusted container must never assume the
            // rank or shape implied by a variable NAME: every read into a fixed
            // or sized host buffer is preceded by an exact rank, exact
            // datatype, and exact extent/dimension-ID check.
            auto var_has_type = [&](int vid, nc_type want) -> bool {
                nc_type t = NC_NAT;
                return nc_inq_vartype(ncid, vid, &t) == NC_NOERR && t == want;
            };
            // Scalar int: rank exactly 0 + exact NC_INT, then one value.
            auto read_scalar_int = [&](const char* name, int& out) -> bool {
                int vid = -1, ndims = 0;
                if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
                if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR)
                    return false;
                if (ndims != 0) return false;
                if (!var_has_type(vid, NC_INT)) return false;
                return nc_get_var_int(ncid, vid, &out) == NC_NOERR;
            };
            // 1-D vector: rank exactly 1 (BEFORE asking for the dimension-ID
            // list — nc_inq_vardimid writes one ID per dimension), the variable
            // must live on the EXPECTED dimension id with the exact extent, and
            // the datatype must be exact. The read itself uses an explicit
            // start/count selection so it can never consume more than the
            // checked destination size.
            auto read_vector_int = [&](const char* name, int expect_dim,
                                       size_t expect,
                                       std::vector<int>& out) -> bool {
                int vid = -1, ndims = 0, dimid = -1;
                if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
                if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR)
                    return false;
                if (ndims != 1) return false;
                if (nc_inq_vardimid(ncid, vid, &dimid) != NC_NOERR)
                    return false;
                if (dimid != expect_dim) return false;
                size_t len = 0;
                if (nc_inq_dimlen(ncid, dimid, &len) != NC_NOERR) return false;
                if (len != expect) return false;
                if (!var_has_type(vid, NC_INT)) return false;
                const auto bytes = checked_mul(expect, sizeof(int));
                if (!bytes) return false;
                out.resize(expect);
                const size_t start[1] = {0}, count[1] = {expect};
                return nc_get_vara_int(ncid, vid, start, count, out.data()) ==
                       NC_NOERR;
            };
            // Scalar double: rank exactly 0 + exact NC_DOUBLE, then one value.
            auto read_scalar_double = [&](const char* name,
                                          double& out) -> bool {
                int vid = -1, ndims = 0;
                if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
                if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR)
                    return false;
                if (ndims != 0) return false;
                if (!var_has_type(vid, NC_DOUBLE)) return false;
                return nc_get_var_double(ncid, vid, &out) == NC_NOERR;
            };
            // 2-D double matrix: rank exactly 2, dimension IDs exactly
            // [dim0, dim1], exact NC_DOUBLE, then one bounded whole-matrix read
            // in the writer's C order (row stride = dim1 extent).
            auto read_matrix_double = [&](const char* name, int dim0, int dim1,
                                          size_t n0, size_t n1,
                                          std::vector<double>& out) -> bool {
                int vid = -1, ndims = 0, dimids[2] = {-1, -1};
                if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
                if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR)
                    return false;
                if (ndims != 2) return false;
                if (nc_inq_vardimid(ncid, vid, dimids) != NC_NOERR)
                    return false;
                if (dimids[0] != dim0 || dimids[1] != dim1) return false;
                if (!var_has_type(vid, NC_DOUBLE)) return false;
                const auto el = checked_mul(n0, n1);
                if (!el) return false;
                const auto bytes = checked_mul(*el, sizeof(double));
                if (!bytes) return false;
                out.resize(*el);
                const size_t start[2] = {0, 0}, count[2] = {n0, n1};
                return nc_get_vara_double(ncid, vid, start, count,
                                          out.data()) == NC_NOERR;
            };
            auto read_vector_double = [&](const char* name, int expect_dim,
                                          size_t expect,
                                          std::vector<double>& out) -> bool {
                int vid = -1, ndims = 0, dimid = -1;
                if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
                if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR)
                    return false;
                if (ndims != 1) return false;
                if (nc_inq_vardimid(ncid, vid, &dimid) != NC_NOERR)
                    return false;
                if (dimid != expect_dim) return false;
                size_t len = 0;
                if (nc_inq_dimlen(ncid, dimid, &len) != NC_NOERR) return false;
                if (len != expect) return false;
                if (!var_has_type(vid, NC_DOUBLE)) return false;
                const auto bytes = checked_mul(expect, sizeof(double));
                if (!bytes) return false;
                out.resize(expect);
                const size_t start[1] = {0}, count[1] = {expect};
                return nc_get_vara_double(ncid, vid, start, count,
                                          out.data()) == NC_NOERR;
            };
            // 2-D state family: rank exactly 2, dimension IDs exactly
            // [ns_dim, mnmax_dim] (so the extents are the named dimensions by
            // construction), exact NC_DOUBLE, then bounded per-surface reads.
            auto read_state_family = [&](const char* name, int ns_dim,
                                         int mnmax_dim, size_t ns, int mnmax,
                                         std::vector<double>& fam) -> bool {
                int vid = -1, ndims = 0, dimids[2] = {-1, -1};
                if (nc_inq_varid(ncid, name, &vid) != NC_NOERR) return false;
                if (nc_inq_varndims(ncid, vid, &ndims) != NC_NOERR)
                    return false;
                if (ndims != 2) return false;
                if (nc_inq_vardimid(ncid, vid, dimids) != NC_NOERR)
                    return false;
                if (dimids[0] != ns_dim || dimids[1] != mnmax_dim) return false;
                if (!var_has_type(vid, NC_DOUBLE)) return false;
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
            if (!get_dim("ns", ns_dim, ns) ||
                !get_dim("mnmax", mnmax_dim, mnmax)) {
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
                if (*n_opt > cumes::io_detail::MAX_STATE_ELEMENTS_PER_FAMILY) {
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
                if (!read_state_family(fam_names[c], ns_dim, mnmax_dim, ns,
                                       snapshot.mnmax, snapshot.families[c])) {
                    return fail("state read failed (missing/malformed " +
                                std::string(fam_names[c]) + ")");
                }
            }
            if (report) {
                // Parse transactionally: a malformed late field must not leave
                // a partially populated report visible to the caller.
                RunReport parsed_report;
                auto get_str = [&](const char* name, std::string& out) -> bool {
                    size_t len = 0;
                    if (nc_inq_attlen(ncid, NC_GLOBAL, name, &len) !=
                        NC_NOERR) {
                        return false;
                    }
                    // Documented resource cap (handoff §4): a hostile length
                    // must fail before the host allocates it.
                    if (len > cumes::io_detail::MAX_PROVENANCE_STRING_BYTES) {
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
                if (!read_scalar_int("precision", precision) ||
                    !read_scalar_int("status", status) ||
                    !read_scalar_int("total_iterations", total) ||
                    !read_scalar_int("build_dirty", dirty)) {
                    return fail("missing run outcome variables");
                }
                // Closed-range validation of the serialized scalars (handoff
                // §4).
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
                if (!get_str("revision", parsed_report.build.revision) ||
                    !get_str("build_type", parsed_report.build.build_type) ||
                    !get_str("precision_policy",
                             parsed_report.build.precision_policy) ||
                    !get_str("compile_flags",
                             parsed_report.build.compile_flags) ||
                    !get_str("source_path", parsed_report.input.source_path) ||
                    !get_str("source_hash", parsed_report.input.source_hash) ||
                    !get_str("gpu_name", parsed_report.runtime.gpu_name) ||
                    !get_str("driver", parsed_report.runtime.driver) ||
                    !get_str("runtime", parsed_report.runtime.runtime) ||
                    !get_str("toolkit", parsed_report.runtime.toolkit)) {
                    return fail("missing provenance attributes");
                }
                int nstages_dim = -1, nrestarts_dim = -1;
                size_t nstages = 0, nrestarts = 0;
                if (!get_dim("nstages", nstages_dim, nstages) ||
                    !get_dim("nrestarts", nrestarts_dim, nrestarts)) {
                    return fail("missing stage dimensions");
                }
                // Documented stage/restart resource caps (handoff §4): a
                // hostile dimension must fail before the host allocates it.
                if (nstages > cumes::io_detail::MAX_STAGE_COUNT ||
                    nrestarts > cumes::io_detail::MAX_STAGE_COUNT) {
                    return fail(
                        "stage/restart dimensions exceed the resource cap");
                }
                std::vector<int> stage_ns(nstages), stage_iter(nstages),
                    stage_conv(nstages), rst_off(nstages);
                std::vector<double> st_fsqr(nstages), st_fsqz(nstages),
                    st_fsql(nstages);
                std::vector<int> rst_iter(nrestarts);
                if (!read_vector_int("stage_ns", nstages_dim, nstages,
                                     stage_ns) ||
                    !read_vector_int("stage_iterations", nstages_dim, nstages,
                                     stage_iter) ||
                    !read_vector_int("stage_converged", nstages_dim, nstages,
                                     stage_conv) ||
                    !read_vector_int("restart_stage_offset", nstages_dim,
                                     nstages, rst_off) ||
                    !read_vector_double("stage_fsqr", nstages_dim, nstages,
                                        st_fsqr) ||
                    !read_vector_double("stage_fsqz", nstages_dim, nstages,
                                        st_fsqz) ||
                    !read_vector_double("stage_fsql", nstages_dim, nstages,
                                        st_fsql)) {
                    return fail("stage history read failed");
                }
                // restart_iteration must exist whenever nrestarts > 0, and must
                // pass the same rank/type/extent checks whenever it exists.
                {
                    int vid = -1;
                    if (nc_inq_varid(ncid, "restart_iteration", &vid) ==
                        NC_NOERR) {
                        if (!read_vector_int("restart_iteration", nrestarts_dim,
                                             nrestarts, rst_iter)) {
                            return fail("restart history read failed");
                        }
                    } else if (nrestarts > 0) {
                        return fail("restart history read failed");
                    }
                }
                {
                    const std::string off_err =
                        validate_restart_offsets(rst_off, nstages, nrestarts);
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
                    const size_t end =
                        (g + 1 < nstages) ? (size_t)rst_off[g + 1] : nrestarts;
                    for (size_t k = begin; k < end; ++k) {
                        st.restarts.push_back(RestartEvent{rst_iter[k]});
                    }
                    parsed_report.stages.push_back(std::move(st));
                }
                // ---- embedded normalized-input record ----
                // All-or-nothing: containers written before the record carry
                // none of its fields (the record stays default-empty); when the
                // first scalar exists, the FULL record is required and every
                // read gets the exact-rank/type/extent hardening above.
                {
                    int probe = -1;
                    if (nc_inq_varid(ncid, "mpol", &probe) == NC_NOERR) {
                        InputParams ip;
                        if (!read_scalar_int("mpol", ip.mpol) ||
                            !read_scalar_int("ntor", ip.ntor) ||
                            !read_scalar_int("nfp", ip.nfp) ||
                            !read_scalar_int("ntheta", ip.ntheta) ||
                            !read_scalar_int("nzeta", ip.nzeta) ||
                            !read_scalar_int("ncurr", ip.ncurr) ||
                            !read_scalar_double("delt", ip.delt) ||
                            !read_scalar_double("phiedge", ip.phiedge) ||
                            !read_scalar_double("pres_scale", ip.pres_scale) ||
                            !read_scalar_double("adiabatic_index",
                                                ip.adiabatic_index) ||
                            !read_scalar_double("spres_ped", ip.spres_ped) ||
                            !read_scalar_double("bloat", ip.bloat) ||
                            !read_scalar_double("curtor", ip.curtor) ||
                            !read_scalar_double("tcon0", ip.tcon0)) {
                            return fail("malformed embedded input record");
                        }
                        {
                            int variable_id = -1;
                            if (nc_inq_varid(ncid, "lfreeb", &variable_id) ==
                                NC_NOERR) {
                                int lfreeb = 0;
                                if (!read_scalar_int("lfreeb", lfreeb)) {
                                    return fail(
                                        "malformed embedded input record");
                                }
                                ip.lfreeb = (lfreeb != 0);
                            }
                            if (nc_inq_varid(ncid, "nvacskip", &variable_id) ==
                                    NC_NOERR &&
                                !read_scalar_int("nvacskip", ip.nvacskip)) {
                                return fail("malformed embedded input record");
                            }
                            get_str("mgrid_file", ip.mgrid_file);
                            get_str("coils_file", ip.coils_file);
                            get_str("makegrid_parameters_file",
                                    ip.makegrid_parameters_file);
                            if (nc_inq_varid(ncid,
                                             "makegrid_parameters_present",
                                             &variable_id) == NC_NOERR) {
                                int present = 0;
                                if (!read_scalar_int(
                                        "makegrid_parameters_present",
                                        present)) {
                                    return fail(
                                        "malformed embedded input record");
                                }
                                if (present != 0) {
                                    MakegridParametersSpec parameters;
                                    int normalize = 0;
                                    int symmetry = 0;
                                    if (!read_scalar_int(
                                            "makegrid_normalize_by_currents",
                                            normalize) ||
                                        !read_scalar_int(
                                            "makegrid_assume_stellarator_"
                                            "symmetry",
                                            symmetry) ||
                                        !read_scalar_int(
                                            "makegrid_number_of_field_periods",
                                            parameters
                                                .number_of_field_periods) ||
                                        !read_scalar_double(
                                            "makegrid_r_grid_minimum",
                                            parameters.r_grid_minimum) ||
                                        !read_scalar_double(
                                            "makegrid_r_grid_maximum",
                                            parameters.r_grid_maximum) ||
                                        !read_scalar_int(
                                            "makegrid_number_of_r_grid_points",
                                            parameters
                                                .number_of_r_grid_points) ||
                                        !read_scalar_double(
                                            "makegrid_z_grid_minimum",
                                            parameters.z_grid_minimum) ||
                                        !read_scalar_double(
                                            "makegrid_z_grid_maximum",
                                            parameters.z_grid_maximum) ||
                                        !read_scalar_int(
                                            "makegrid_number_of_z_grid_points",
                                            parameters
                                                .number_of_z_grid_points) ||
                                        !read_scalar_int(
                                            "makegrid_number_of_phi_grid_"
                                            "points",
                                            parameters
                                                .number_of_phi_grid_points)) {
                                        return fail(
                                            "malformed embedded input record");
                                    }
                                    parameters.normalize_by_currents =
                                        normalize != 0;
                                    parameters.assume_stellarator_symmetry =
                                        symmetry != 0;
                                    ip.embedded_makegrid_parameters =
                                        parameters;
                                }
                            }
                        }
                        // Arrays: an ABSENT variable means an empty vector (the
                        // writer omits empty arrays to avoid 0-length dims); a
                        // present one must pass the strict read.
                        {
                            auto read_optional =
                                [&](const char* dname, const char* vname,
                                    std::vector<double>& out) -> bool {
                                int vid = -1;
                                if (nc_inq_varid(ncid, vname, &vid) !=
                                    NC_NOERR) {
                                    return true;  // absent -> empty
                                }
                                int dim = -1;
                                size_t n = 0;
                                if (!get_dim(dname, dim, n)) return false;
                                return read_vector_double(vname, dim, n, out);
                            };
                            if (!read_optional("n_am", "am", ip.am) ||
                                !read_optional("n_ac", "ac", ip.ac) ||
                                !read_optional("n_ai", "ai", ip.ai) ||
                                !read_optional("n_aphi", "aphi", ip.aphi) ||
                                !read_optional("n_raxis", "raxis_c",
                                               ip.raxis_c) ||
                                !read_optional("n_zaxis", "zaxis_s",
                                               ip.zaxis_s) ||
                                !read_optional("n_extcur", "extcur",
                                               ip.extcur)) {
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
                                if (!get_dim("nstages_in", dim_nstages_in,
                                             nstages_in)) {
                                    return fail(
                                        "malformed embedded input record");
                                }
                                if (nstages_in >
                                    cumes::io_detail::MAX_STAGE_COUNT) {
                                    return fail(
                                        "embedded input stage count "
                                        "exceeds the resource cap");
                                }
                                std::vector<int> stg_in_ns(nstages_in),
                                    stg_max_iter(nstages_in);
                                std::vector<double> stg_ftol(nstages_in);
                                if (!read_vector_int("stage_in_ns",
                                                     dim_nstages_in, nstages_in,
                                                     stg_in_ns) ||
                                    !read_vector_int("stage_max_iter",
                                                     dim_nstages_in, nstages_in,
                                                     stg_max_iter) ||
                                    !read_vector_double("stage_ftol",
                                                        dim_nstages_in,
                                                        nstages_in, stg_ftol)) {
                                    return fail(
                                        "malformed embedded input record");
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
                        if (!get_dim("nrbc", dim_nrbc, nrbc) ||
                            !get_dim("nzbs", dim_nzbs, nzbs) ||
                            !get_dim("n_mpol", dim_mpol, mpol) ||
                            !get_dim("n_ntorp1", dim_ntorp1, ntorp1)) {
                            return fail("malformed embedded input record");
                        }
                        if (!read_vector_int("rbc_m", dim_nrbc, nrbc,
                                             ip.rbc_m) ||
                            !read_vector_int("rbc_n", dim_nrbc, nrbc,
                                             ip.rbc_n) ||
                            !read_vector_double("rbc_value", dim_nrbc, nrbc,
                                                ip.rbc_value) ||
                            !read_vector_int("zbs_m", dim_nzbs, nzbs,
                                             ip.zbs_m) ||
                            !read_vector_int("zbs_n", dim_nzbs, nzbs,
                                             ip.zbs_n) ||
                            !read_vector_double("zbs_value", dim_nzbs, nzbs,
                                                ip.zbs_value) ||
                            !read_matrix_double("rbcc", dim_mpol, dim_ntorp1,
                                                mpol, ntorp1, ip.rbcc) ||
                            !read_matrix_double("rbss", dim_mpol, dim_ntorp1,
                                                mpol, ntorp1, ip.rbss) ||
                            !read_matrix_double("zbsc", dim_mpol, dim_ntorp1,
                                                mpol, ntorp1, ip.zbsc) ||
                            !read_matrix_double("zbcs", dim_mpol, dim_ntorp1,
                                                mpol, ntorp1, ip.zbcs)) {
                            return fail("malformed embedded input record");
                        }
                        // The schema tag is informational; an absent or
                        // unreadable one keeps the default.
                        // The schema/profile-type tags are informational; an
                        // absent one keeps the default (an older container
                        // lacks the profile types -> "power_series").
                        std::string schema_tag;
                        if (get_str("schema", schema_tag))
                            ip.schema = schema_tag;
                        std::string pmass_tag, piota_tag, pcurr_tag;
                        if (get_str("pmass_type", pmass_tag))
                            ip.pmass_type = pmass_tag;
                        if (get_str("piota_type", piota_tag))
                            ip.piota_type = piota_tag;
                        if (get_str("pcurr_type", pcurr_tag))
                            ip.pcurr_type = pcurr_tag;
                        parsed_report.input_params = std::move(ip);
                    }
                }
                report->get() = std::move(parsed_report);
            }
            nc_close(ncid);
            return snapshot;
        } catch (const std::bad_alloc&) {
            nc_close(ncid);
            return Result<EquilibriumSnapshot>(
                "NetCDF: allocation failed (dimensions implausible or "
                "corrupt)");
        } catch (const std::length_error&) {
            nc_close(ncid);
            return Result<EquilibriumSnapshot>(
                "NetCDF: allocation failed (dimensions implausible or "
                "corrupt)");
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
