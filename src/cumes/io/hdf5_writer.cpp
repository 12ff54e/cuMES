// hdf5_writer.cpp — host-only HDF5 state adapters (completion plan steps
// 2.2/2.3). Mirrors src/cumes/io/netcdf_writer.cpp content-for-content:
//
// Schema v1 only: active dimensions + complete provenance (precision/status/
// total iterations, build/input/runtime strings, source hash, per-stage
// outcomes, full restart history, raw + folded boundary harmonics) with a
// round-tripping reader.
//
// This TU (and only this TU) includes <hdf5.h>; it compiles under
// CUMES_HAVE_HDF5, which is confined to the adapter library target.
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

#include <hdf5.h>

namespace cumes {
namespace {

constexpr const char* HALF_FIELD_NAMES[EquilibriumSnapshot::HALF_FIELD_COUNT] =
    {"sqrtg", "bsups", "bsupu", "bsupv", "bsubs", "bsubu", "bsubv"};
constexpr const char* FULL_FIELD_NAMES[EquilibriumSnapshot::FULL_FIELD_COUNT] =
    {"jsups", "jsupu", "jsupv", "jsubs", "jsubu", "jsubv"};

// Write one scalar (or fixed-size string) attribute on the root group.
herr_t put_attr(hid_t loc, const char* name, hid_t dtype, const void* val) {
    hid_t sid = H5Screate(H5S_SCALAR);
    if (sid < 0) { return -1; }
    hid_t aid = H5Acreate2(loc, name, dtype, sid, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(sid);
    if (aid < 0) { return -1; }
    herr_t r = H5Awrite(aid, dtype, val);
    H5Aclose(aid);
    return r;
}

herr_t put_str_attr(hid_t loc, const char* name, const std::string& value) {
    hid_t s1 = H5Tcopy(H5T_C_S1);
    if (s1 < 0) return -1;
    herr_t r0 = H5Tset_size(s1, value.size() + 1);
    herr_t r1 = r0 >= 0 ? put_attr(loc, name, s1, value.c_str()) : -1;
    H5Tclose(s1);
    return r1;
}

// RAII handle closer (reader-rank-hardening handoff §3): every opened
// dataset/dataspace/datatype/attribute is closed on every failure path.
// H5Idec_ref closes any object class when its reference count reaches zero.
class H5Closer {
   public:
    explicit H5Closer(hid_t id) : id_(id) {}
    ~H5Closer() {
        if (id_ >= 0) H5Idec_ref(id_);
    }
    H5Closer(const H5Closer&) = delete;
    H5Closer& operator=(const H5Closer&) = delete;
    hid_t get() const { return id_; }

   private:
    hid_t id_;
};

// Read a fixed-size string attribute back into a std::string.
// Hardened (reader-rank-hardening handoff §3): the dataspace must be scalar
// or exactly one element BEFORE the single-width destination is sized — a
// multi-element attribute would otherwise overflow it — and the datatype
// must be a bounded fixed-size string.
bool get_str_attr(hid_t loc, const char* name, std::string& out) {
    H5Closer aid(H5Aopen(loc, name, H5P_DEFAULT));
    if (aid.get() < 0) return false;
    // Exact dataspace element count: scalar, or a 1-D space of one point.
    {
        H5Closer sp(H5Aget_space(aid.get()));
        if (sp.get() < 0) return false;
        const int ndims = H5Sget_simple_extent_ndims(sp.get());
        hssize_t npts = 1;
        if (ndims == 1) {
            hsize_t ext[1] = {0};
            if (H5Sget_simple_extent_dims(sp.get(), ext, nullptr) < 0) {
                return false;
            }
            npts = (hssize_t)ext[0];
        } else if (ndims != 0) {
            return false;
        }
        if (npts != 1) return false;
    }
    H5Closer ty(H5Aget_type(aid.get()));
    if (ty.get() < 0) return false;
    if (H5Tget_class(ty.get()) != H5T_STRING) return false;
    // Schema v1 writes bounded fixed-width strings. H5Tget_size() returns
    // sizeof(char*) for a variable-length string, not its payload length, and
    // H5Aread would require pointer storage plus explicit reclamation. Reject
    // that representation before allocating or reading.
    const htri_t is_variable = H5Tis_variable_str(ty.get());
    if (is_variable != 0) return false;  // positive = vlen; negative = error
    const size_t width = H5Tget_size(ty.get());
    // Bounded width: the declared type size must fit the documented resource
    // cap before the destination is allocated.
    if (width == 0 || width > cumes::io_detail::MAX_PROVENANCE_STRING_BYTES) {
        return false;
    }
    out.resize(width + 1, '\0');
    const herr_t r = H5Aread(aid.get(), ty.get(), &out[0]);
    if (r < 0) return false;
    out.resize(width);  // drop the stored NUL; keep embedded NULs if any
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return true;
}

// ---------------------------------------------------------------------------
// v1: active dimensions + complete provenance + boundary harmonics
// ---------------------------------------------------------------------------
class Hdf5V1Writer final : public Writer {
   public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report,
                        const OutputSpec& spec,
                        const ValidatedProblem& problem) override {
        if (snapshot.has_any_derived_fields() &&
            !snapshot.has_derived_fields()) {
            return Status("HDF5: incomplete derived-field snapshot");
        }
        const FoldedBoundary& fb = problem.boundary();
        const int mpol = problem.shape().mpol;
        const int ntorp1 = problem.shape().ntor + 1;
        const std::vector<BoundaryHarmonic>& rbc = problem.spec().rbc;
        const std::vector<BoundaryHarmonic>& zbs = problem.spec().zbs;
        const size_t nstages = report.stages.size();
        size_t nrestarts = 0;
        for (const auto& st : report.stages) nrestarts += st.restarts.size();
        const InputParams& ip = report.input_params;

        const std::string tmp = io_detail::temp_path_for(spec.path);
        hid_t fid =
            H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (fid < 0) return Status("HDF5 H5Fcreate failed for " + tmp);
        auto fail = [&](const std::string& msg) -> Status {
            H5Fclose(fid);
            remove(tmp.c_str());
            return Status("HDF5: " + msg);
        };
#define H5_CHECK(expr, msg_)                            \
    do {                                                \
        if ((expr) < 0) return fail(std::string(msg_)); \
    } while (0)

        auto write_array = [&](const char* name, hid_t dtype, int nd,
                               const hsize_t* dims,
                               const void* data) -> herr_t {
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
            const std::vector<double>& dbuf = snapshot.component(
                static_cast<EquilibriumSnapshot::Component>(c));
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
        if (snapshot.has_derived_fields()) {
            const hsize_t half_dims[3] = {
                static_cast<hsize_t>(snapshot.ns - 1),
                static_cast<hsize_t>(snapshot.nzeta),
                static_cast<hsize_t>(snapshot.ntheta)};
            const hsize_t full_dims[3] = {
                static_cast<hsize_t>(snapshot.ns),
                static_cast<hsize_t>(snapshot.nzeta),
                static_cast<hsize_t>(snapshot.ntheta)};
            for (int c = 0; c < EquilibriumSnapshot::HALF_FIELD_COUNT; ++c) {
                H5_CHECK(write_array(HALF_FIELD_NAMES[c], H5T_NATIVE_DOUBLE, 3,
                                     half_dims, snapshot.half_fields[c].data()),
                         "write half-grid field");
            }
            for (int c = 0; c < EquilibriumSnapshot::FULL_FIELD_COUNT; ++c) {
                H5_CHECK(write_array(FULL_FIELD_NAMES[c], H5T_NATIVE_DOUBLE, 3,
                                     full_dims, snapshot.full_fields[c].data()),
                         "write full-grid field");
            }
        }

        // ---- run outcome + provenance attributes ----
        const int precision = (report.build.scalar_type == "double") ? 0 : 1;
        const int status = static_cast<int>(report.status);
        const int dirty = report.build.dirty ? 1 : 0;
        H5_CHECK(put_attr(fid, "precision", H5T_NATIVE_INT, &precision),
                 "attr precision");
        H5_CHECK(put_attr(fid, "status", H5T_NATIVE_INT, &status),
                 "attr status");
        H5_CHECK(put_attr(fid, "total_iterations", H5T_NATIVE_INT,
                          &report.total_effective_iterations),
                 "attr total_iterations");
        H5_CHECK(put_attr(fid, "build_dirty", H5T_NATIVE_INT, &dirty),
                 "attr build_dirty");
        H5_CHECK(put_str_attr(fid, "revision", report.build.revision),
                 "attr revision");
        H5_CHECK(put_str_attr(fid, "build_type", report.build.build_type),
                 "attr build_type");
        H5_CHECK(put_str_attr(fid, "scalar_type", report.build.scalar_type),
                 "attr scalar_type");
        H5_CHECK(put_str_attr(fid, "precision_policy",
                              report.build.precision_policy),
                 "attr precision_policy");
        H5_CHECK(put_str_attr(fid, "compile_flags", report.build.compile_flags),
                 "attr compile_flags");
        H5_CHECK(put_str_attr(fid, "source_path", report.input.source_path),
                 "attr source_path");
        H5_CHECK(put_str_attr(fid, "source_hash", report.input.source_hash),
                 "attr source_hash");
        H5_CHECK(put_str_attr(fid, "gpu_name", report.runtime.gpu_name),
                 "attr gpu_name");
        H5_CHECK(put_str_attr(fid, "driver", report.runtime.driver),
                 "attr driver");
        H5_CHECK(put_str_attr(fid, "runtime", report.runtime.runtime),
                 "attr runtime");
        H5_CHECK(put_str_attr(fid, "toolkit", report.runtime.toolkit),
                 "attr toolkit");

        // ---- embedded normalized-input record attributes + arrays ----
        H5_CHECK(put_attr(fid, "mpol", H5T_NATIVE_INT, &ip.mpol), "attr mpol");
        H5_CHECK(put_attr(fid, "ntor", H5T_NATIVE_INT, &ip.ntor), "attr ntor");
        H5_CHECK(put_attr(fid, "nfp", H5T_NATIVE_INT, &ip.nfp), "attr nfp");
        H5_CHECK(put_attr(fid, "ntheta", H5T_NATIVE_INT, &ip.ntheta),
                 "attr ntheta");
        H5_CHECK(put_attr(fid, "nzeta", H5T_NATIVE_INT, &ip.nzeta),
                 "attr nzeta");
        H5_CHECK(put_attr(fid, "ncurr", H5T_NATIVE_INT, &ip.ncurr),
                 "attr ncurr");
        H5_CHECK(put_attr(fid, "delt", H5T_NATIVE_DOUBLE, &ip.delt),
                 "attr delt");
        H5_CHECK(put_attr(fid, "phiedge", H5T_NATIVE_DOUBLE, &ip.phiedge),
                 "attr phiedge");
        H5_CHECK(put_attr(fid, "pres_scale", H5T_NATIVE_DOUBLE, &ip.pres_scale),
                 "attr pres_scale");
        H5_CHECK(put_attr(fid, "adiabatic_index", H5T_NATIVE_DOUBLE,
                          &ip.adiabatic_index),
                 "attr adiabatic_index");
        H5_CHECK(put_attr(fid, "spres_ped", H5T_NATIVE_DOUBLE, &ip.spres_ped),
                 "attr spres_ped");
        H5_CHECK(put_attr(fid, "bloat", H5T_NATIVE_DOUBLE, &ip.bloat),
                 "attr bloat");
        H5_CHECK(put_attr(fid, "curtor", H5T_NATIVE_DOUBLE, &ip.curtor),
                 "attr curtor");
        H5_CHECK(put_attr(fid, "tcon0", H5T_NATIVE_DOUBLE, &ip.tcon0),
                 "attr tcon0");
        const int lfreeb = ip.lfreeb ? 1 : 0;
        H5_CHECK(put_attr(fid, "lfreeb", H5T_NATIVE_INT, &lfreeb),
                 "attr lfreeb");
        H5_CHECK(put_attr(fid, "nvacskip", H5T_NATIVE_INT, &ip.nvacskip),
                 "attr nvacskip");
        H5_CHECK(put_str_attr(fid, "mgrid_file", ip.mgrid_file),
                 "attr mgrid_file");
        H5_CHECK(put_str_attr(fid, "coils_file", ip.coils_file),
                 "attr coils_file");
        H5_CHECK(put_str_attr(fid, "makegrid_parameters_file",
                              ip.makegrid_parameters_file),
                 "attr makegrid_parameters_file");
        const int makegrid_present =
            ip.embedded_makegrid_parameters.has_value() ? 1 : 0;
        H5_CHECK(put_attr(fid, "makegrid_parameters_present", H5T_NATIVE_INT,
                          &makegrid_present),
                 "attr makegrid_parameters_present");
        const MakegridParametersSpec makegrid =
            ip.embedded_makegrid_parameters.value_or(MakegridParametersSpec{});
        const int makegrid_normalize = makegrid.normalize_by_currents ? 1 : 0;
        const int makegrid_symmetry =
            makegrid.assume_stellarator_symmetry ? 1 : 0;
        H5_CHECK(put_attr(fid, "makegrid_normalize_by_currents", H5T_NATIVE_INT,
                          &makegrid_normalize),
                 "attr makegrid_normalize_by_currents");
        H5_CHECK(put_attr(fid, "makegrid_assume_stellarator_symmetry",
                          H5T_NATIVE_INT, &makegrid_symmetry),
                 "attr makegrid_assume_stellarator_symmetry");
        H5_CHECK(put_attr(fid, "makegrid_number_of_field_periods",
                          H5T_NATIVE_INT, &makegrid.number_of_field_periods),
                 "attr makegrid_number_of_field_periods");
        H5_CHECK(put_attr(fid, "makegrid_r_grid_minimum", H5T_NATIVE_DOUBLE,
                          &makegrid.r_grid_minimum),
                 "attr makegrid_r_grid_minimum");
        H5_CHECK(put_attr(fid, "makegrid_r_grid_maximum", H5T_NATIVE_DOUBLE,
                          &makegrid.r_grid_maximum),
                 "attr makegrid_r_grid_maximum");
        H5_CHECK(put_attr(fid, "makegrid_number_of_r_grid_points",
                          H5T_NATIVE_INT, &makegrid.number_of_r_grid_points),
                 "attr makegrid_number_of_r_grid_points");
        H5_CHECK(put_attr(fid, "makegrid_z_grid_minimum", H5T_NATIVE_DOUBLE,
                          &makegrid.z_grid_minimum),
                 "attr makegrid_z_grid_minimum");
        H5_CHECK(put_attr(fid, "makegrid_z_grid_maximum", H5T_NATIVE_DOUBLE,
                          &makegrid.z_grid_maximum),
                 "attr makegrid_z_grid_maximum");
        H5_CHECK(put_attr(fid, "makegrid_number_of_z_grid_points",
                          H5T_NATIVE_INT, &makegrid.number_of_z_grid_points),
                 "attr makegrid_number_of_z_grid_points");
        H5_CHECK(put_attr(fid, "makegrid_number_of_phi_grid_points",
                          H5T_NATIVE_INT, &makegrid.number_of_phi_grid_points),
                 "attr makegrid_number_of_phi_grid_points");
        H5_CHECK(put_str_attr(fid, "schema", ip.schema), "attr schema");
        H5_CHECK(put_str_attr(fid, "pmass_type", ip.pmass_type),
                 "attr pmass_type");
        H5_CHECK(put_str_attr(fid, "piota_type", ip.piota_type),
                 "attr piota_type");
        H5_CHECK(put_str_attr(fid, "pcurr_type", ip.pcurr_type),
                 "attr pcurr_type");
        {
            auto write_vec = [&](const char* name,
                                 const std::vector<double>& v) -> herr_t {
                const hsize_t d[1] = {v.size()};
                return write_array(name, H5T_NATIVE_DOUBLE, 1, d,
                                   v.empty() ? nullptr : v.data());
            };
            H5_CHECK(write_vec("am", ip.am), "write am");
            H5_CHECK(write_vec("extcur", ip.extcur), "write extcur");
            H5_CHECK(write_vec("ac", ip.ac), "write ac");
            H5_CHECK(write_vec("ai", ip.ai), "write ai");
            H5_CHECK(write_vec("aphi", ip.aphi), "write aphi");
            H5_CHECK(write_vec("raxis_c", ip.raxis_c), "write raxis_c");
            H5_CHECK(write_vec("zaxis_s", ip.zaxis_s), "write zaxis_s");
        }
        {
            const size_t nstages_in = ip.stages.size();
            const hsize_t d_in[1] = {nstages_in};
            std::vector<int> stg_in_ns(nstages_in), stg_max_iter(nstages_in);
            std::vector<double> stg_ftol(nstages_in);
            for (size_t g = 0; g < nstages_in; ++g) {
                stg_in_ns[g] = ip.stages[g].ns;
                stg_max_iter[g] = ip.stages[g].max_iter;
                stg_ftol[g] = ip.stages[g].ftol;
            }
            H5_CHECK(write_array("stage_in_ns", H5T_NATIVE_INT, 1, d_in,
                                 stg_in_ns.data()),
                     "write stage_in_ns");
            H5_CHECK(write_array("stage_max_iter", H5T_NATIVE_INT, 1, d_in,
                                 stg_max_iter.data()),
                     "write stage_max_iter");
            H5_CHECK(write_array("stage_ftol", H5T_NATIVE_DOUBLE, 1, d_in,
                                 stg_ftol.data()),
                     "write stage_ftol");
        }

        // ---- stage history + restart arrays (active dims) ----
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
        const hsize_t d_stages[1] = {nstages};
        const hsize_t d_restarts[1] = {nrestarts};
        H5_CHECK(write_array("stage_ns", H5T_NATIVE_INT, 1, d_stages,
                             stage_ns.data()),
                 "write stage_ns");
        H5_CHECK(write_array("stage_iterations", H5T_NATIVE_INT, 1, d_stages,
                             stage_iter.data()),
                 "write stage_iterations");
        H5_CHECK(write_array("stage_converged", H5T_NATIVE_INT, 1, d_stages,
                             stage_conv.data()),
                 "write stage_converged");
        H5_CHECK(write_array("stage_fsqr", H5T_NATIVE_DOUBLE, 1, d_stages,
                             st_fsqr.data()),
                 "write stage_fsqr");
        H5_CHECK(write_array("stage_fsqz", H5T_NATIVE_DOUBLE, 1, d_stages,
                             st_fsqz.data()),
                 "write stage_fsqz");
        H5_CHECK(write_array("stage_fsql", H5T_NATIVE_DOUBLE, 1, d_stages,
                             st_fsql.data()),
                 "write stage_fsql");
        H5_CHECK(write_array("restart_stage_offset", H5T_NATIVE_INT, 1,
                             d_stages, rst_off.data()),
                 "write restart_stage_offset");
        if (nrestarts > 0) {
            H5_CHECK(write_array("restart_iteration", H5T_NATIVE_INT, 1,
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
                hm[i] = rbc[i].m;
                hn[i] = rbc[i].n;
                hv[i] = rbc[i].value;
            }
            H5_CHECK(write_array("rbc_m", H5T_NATIVE_INT, 1, d_r, hm.data()),
                     "write rbc_m");
            H5_CHECK(write_array("rbc_n", H5T_NATIVE_INT, 1, d_r, hn.data()),
                     "write rbc_n");
            H5_CHECK(
                write_array("rbc_value", H5T_NATIVE_DOUBLE, 1, d_r, hv.data()),
                "write rbc_value");
        }
        if (nzbs > 0) {
            const hsize_t d_z[1] = {nzbs};
            for (size_t i = 0; i < nzbs; ++i) {
                hm[i] = zbs[i].m;
                hn[i] = zbs[i].n;
                hv[i] = zbs[i].value;
            }
            H5_CHECK(write_array("zbs_m", H5T_NATIVE_INT, 1, d_z, hm.data()),
                     "write zbs_m");
            H5_CHECK(write_array("zbs_n", H5T_NATIVE_INT, 1, d_z, hn.data()),
                     "write zbs_n");
            H5_CHECK(
                write_array("zbs_value", H5T_NATIVE_DOUBLE, 1, d_z, hv.data()),
                "write zbs_value");
        }

        // ---- folded boundary matrices (active mpol x (ntor+1)) ----
        const hsize_t bnd_dims[2] = {(hsize_t)mpol, (hsize_t)ntorp1};
        H5_CHECK(
            write_array("rbcc", H5T_NATIVE_DOUBLE, 2, bnd_dims, fb.rbcc.data()),
            "write rbcc");
        H5_CHECK(
            write_array("rbss", H5T_NATIVE_DOUBLE, 2, bnd_dims, fb.rbss.data()),
            "write rbss");
        H5_CHECK(
            write_array("zbsc", H5T_NATIVE_DOUBLE, 2, bnd_dims, fb.zbsc.data()),
            "write zbsc");
        H5_CHECK(
            write_array("zbcs", H5T_NATIVE_DOUBLE, 2, bnd_dims, fb.zbcs.data()),
            "write zbcs");

        if (H5Fclose(fid) < 0) {
            remove(tmp.c_str());
            return Status("HDF5 H5Fclose failed");
        }
#undef H5_CHECK
        // Durable publication (completion-plan follow-up §3): the library
        // owns the descriptor, so publish_library_file reopens the completed
        // temp, checks fsync + close, then renames + directory-fsyncs.
        const std::string err = io_detail::publish_library_file(tmp, spec.path);
        if (!err.empty()) return Status("HDF5 publish: " + err);
        return Status();
    }
};

// ---------------------------------------------------------------------------
// v1 reader: round-trips the state AND the complete RunReport
// ---------------------------------------------------------------------------
class Hdf5V1Reader final : public Reader {
   public:
    Result<EquilibriumSnapshot> read(
        const std::string& path,
        std::optional<std::reference_wrapper<RunReport>> report) override {
        hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid < 0) {
            return Result<EquilibriumSnapshot>("HDF5: cannot open " + path);
        }
        auto fail = [&](const std::string& msg) -> Result<EquilibriumSnapshot> {
            H5Fclose(fid);
            return Result<EquilibriumSnapshot>("HDF5: " + msg);
        };
        // Allocation failures in a malformed-container read must become typed
        // errors, never exceptions across the Reader interface
        // (reader-rank-hardening handoff §4).
        try {
            // ---- exact-rank dataspace helpers (reader-rank-hardening §3)
            // -------- H5Sget_simple_extent_ndims is queried BEFORE
            // H5Sget_simple_extent_ dims: the latter writes `rank` dimension
            // values into the caller's buffer, so a higher-rank dataspace would
            // overflow it.
            auto get_dim_exact = [&](const char* name, int rank,
                                     hsize_t* dims) -> bool {
                H5Closer ds(H5Dopen2(fid, name, H5P_DEFAULT));
                if (ds.get() < 0) return false;
                H5Closer sp(H5Dget_space(ds.get()));
                if (sp.get() < 0) return false;
                const int ndims = H5Sget_simple_extent_ndims(sp.get());
                if (ndims != rank) return false;
                return H5Sget_simple_extent_dims(sp.get(), dims, nullptr) ==
                       rank;
            };
            // Portable schema integer: signed, native-int width (32 bits on all
            // supported builds). Endian conversion remains intentionally
            // allowed.
            auto type_is_schema_integer = [&](hid_t ty) -> bool {
                return ty >= 0 && H5Tget_class(ty) == H5T_INTEGER &&
                       H5Tget_size(ty) == sizeof(int) &&
                       H5Tget_sign(ty) == H5T_SGN_2;
            };
            auto dataset_is_integer = [&](hid_t ds) -> bool {
                H5Closer ty(H5Dget_type(ds));
                return type_is_schema_integer(ty.get());
            };
            auto dataset_is_double = [&](hid_t ds) -> bool {
                H5Closer ty(H5Dget_type(ds));
                return ty.get() >= 0 && H5Tget_class(ty.get()) == H5T_FLOAT &&
                       H5Tget_size(ty.get()) == sizeof(double);
            };
            // Scalar-or-one-element attribute, integer type, exact 4-byte size,
            // and a checked element count before the single-int read.
            auto get_int_attr = [&](const char* name, int& out) -> bool {
                H5Closer aid(H5Aopen(fid, name, H5P_DEFAULT));
                if (aid.get() < 0) return false;
                {
                    H5Closer sp(H5Aget_space(aid.get()));
                    if (sp.get() < 0) return false;
                    const int ndims = H5Sget_simple_extent_ndims(sp.get());
                    hssize_t npts = 1;
                    if (ndims == 1) {
                        hsize_t ext[1] = {0};
                        if (H5Sget_simple_extent_dims(sp.get(), ext, nullptr) <
                            0) {
                            return false;
                        }
                        npts = (hssize_t)ext[0];
                    } else if (ndims != 0) {
                        return false;
                    }
                    if (npts != 1) return false;
                }
                H5Closer ty(H5Aget_type(aid.get()));
                if (!type_is_schema_integer(ty.get())) return false;
                return H5Aread(aid.get(), H5T_NATIVE_INT, &out) >= 0;
            };
            // 1-D integer/double vectors with exact rank, extent, and datatype.
            auto get_int_arr = [&](const char* name, std::vector<int>& out,
                                   size_t expect) -> bool {
                H5Closer ds(H5Dopen2(fid, name, H5P_DEFAULT));
                if (ds.get() < 0) return false;
                hsize_t dims[1] = {0};
                if (!get_dim_exact(name, 1, dims)) return false;
                if (dims[0] != expect) return false;
                if (!dataset_is_integer(ds.get())) return false;
                const auto bytes = checked_mul(expect, sizeof(int));
                if (!bytes) return false;
                out.resize(expect);
                return H5Dread(ds.get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, out.data()) >= 0;
            };
            auto get_dbl_arr = [&](const char* name, std::vector<double>& out,
                                   size_t expect) -> bool {
                H5Closer ds(H5Dopen2(fid, name, H5P_DEFAULT));
                if (ds.get() < 0) return false;
                hsize_t dims[1] = {0};
                if (!get_dim_exact(name, 1, dims)) return false;
                if (dims[0] != expect) return false;
                if (!dataset_is_double(ds.get())) return false;
                const auto bytes = checked_mul(expect, sizeof(double));
                if (!bytes) return false;
                out.resize(expect);
                return H5Dread(ds.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, out.data()) >= 0;
            };
            // Scalar-or-one-element double attribute with the same dataspace
            // hardening as get_int_attr and an exact 8-byte float type.
            auto get_dbl_attr = [&](const char* name, double& out) -> bool {
                H5Closer aid(H5Aopen(fid, name, H5P_DEFAULT));
                if (aid.get() < 0) return false;
                {
                    H5Closer sp(H5Aget_space(aid.get()));
                    if (sp.get() < 0) return false;
                    const int ndims = H5Sget_simple_extent_ndims(sp.get());
                    hssize_t npts = 1;
                    if (ndims == 1) {
                        hsize_t ext[1] = {0};
                        if (H5Sget_simple_extent_dims(sp.get(), ext, nullptr) <
                            0) {
                            return false;
                        }
                        npts = (hssize_t)ext[0];
                    } else if (ndims != 0) {
                        return false;
                    }
                    if (npts != 1) return false;
                }
                H5Closer ty(H5Aget_type(aid.get()));
                if (ty.get() < 0) return false;
                if (!(H5Tget_class(ty.get()) == H5T_FLOAT &&
                      H5Tget_size(ty.get()) == sizeof(double))) {
                    return false;
                }
                return H5Aread(aid.get(), H5T_NATIVE_DOUBLE, &out) >= 0;
            };
            // 2-D double matrix with exact rank and extents (the folded
            // boundary datasets, stored in the writer's C order — row stride =
            // expect1).
            auto get_dbl_mat = [&](const char* name, std::vector<double>& out,
                                   size_t expect0, size_t expect1) -> bool {
                H5Closer ds(H5Dopen2(fid, name, H5P_DEFAULT));
                if (ds.get() < 0) return false;
                hsize_t dims[2] = {0, 0};
                if (!get_dim_exact(name, 2, dims)) return false;
                if (dims[0] != expect0 || dims[1] != expect1) return false;
                if (!dataset_is_double(ds.get())) return false;
                const auto el = checked_mul(expect0, expect1);
                if (!el) return false;
                const auto bytes = checked_mul(*el, sizeof(double));
                if (!bytes) return false;
                out.resize(*el);
                return H5Dread(ds.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, out.data()) >= 0;
            };
            auto get_dbl_field = [&](const char* name, const hsize_t* expected,
                                     std::vector<double>& out) -> bool {
                H5Closer ds(H5Dopen2(fid, name, H5P_DEFAULT));
                if (ds.get() < 0) return false;
                hsize_t dims[3] = {0, 0, 0};
                if (!get_dim_exact(name, 3, dims)) return false;
                for (int d = 0; d < 3; ++d)
                    if (dims[d] != expected[d]) return false;
                if (!dataset_is_double(ds.get())) return false;
                const auto plane = checked_mul(static_cast<size_t>(dims[1]),
                                               static_cast<size_t>(dims[2]));
                const auto count =
                    plane ? checked_mul(static_cast<size_t>(dims[0]), *plane)
                          : std::nullopt;
                if (!count || *count > io_detail::MAX_REAL_FIELD_ELEMENTS)
                    return false;
                out.resize(*count);
                return H5Dread(ds.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, out.data()) >= 0;
            };

            hsize_t state_dims[2] = {0, 0};
            if (!get_dim_exact("rmncc", 2, state_dims)) {
                return fail("missing/malformed state dataset (rmncc)");
            }
            // Bounds BEFORE narrowing and allocation (handoff §4): dimensions
            // must fit the EquilibriumSnapshot int fields.
            if (state_dims[0] < 1 || state_dims[1] < 1 ||
                state_dims[0] > INT_MAX || state_dims[1] > INT_MAX) {
                return fail("bad state dimensions");
            }
            const int ns = static_cast<int>(state_dims[0]);
            const int mnmax = static_cast<int>(state_dims[1]);
            const auto n_opt = checked_mul((size_t)ns, (size_t)mnmax);
            if (!n_opt) return fail("dimension product overflows size_t");
            if (*n_opt > cumes::io_detail::MAX_STATE_ELEMENTS_PER_FAMILY) {
                return fail("state dimensions exceed the resource cap");
            }
            {
                const auto bytes = checked_mul(*n_opt, sizeof(double));
                if (!bytes) return fail("state byte count overflows size_t");
            }

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
                // EVERY family must independently satisfy rank 2 + the exact
                // [ns, mnmax] extents + a double-compatible type before the
                // read (reader-rank-hardening §3): rmncc alone no longer
                // vouches for the other five.
                H5Closer ds(H5Dopen2(fid, fam_names[c], H5P_DEFAULT));
                if (ds.get() < 0) {
                    return fail("missing state dataset (" +
                                std::string(fam_names[c]) + ")");
                }
                hsize_t fam_dims[2] = {0, 0};
                if (!get_dim_exact(fam_names[c], 2, fam_dims) ||
                    fam_dims[0] != state_dims[0] ||
                    fam_dims[1] != state_dims[1]) {
                    return fail("malformed state dataset (" +
                                std::string(fam_names[c]) + ")");
                }
                if (!dataset_is_double(ds.get())) {
                    return fail("malformed state dataset type (" +
                                std::string(fam_names[c]) + ")");
                }
                H5Closer sp(H5Screate_simple(2, state_dims, nullptr));
                if (sp.get() < 0) return fail("state read failed");
                const herr_t r = H5Dread(ds.get(), H5T_NATIVE_DOUBLE, sp.get(),
                                         H5S_ALL, H5P_DEFAULT, tmp.data());
                if (r < 0) return fail("state read failed");
                snapshot.families[c].resize(*n_opt);
                for (int m = 0; m < mnmax; ++m) {
                    for (int j = 0; j < ns; ++j) {
                        snapshot.families[c][(size_t)m * ns + j] =
                            tmp[(size_t)j * mnmax + m];
                    }
                }
            }
            int present_fields = 0;
            for (const char* name : HALF_FIELD_NAMES) {
                const htri_t present = H5Lexists(fid, name, H5P_DEFAULT);
                if (present < 0) return fail("cannot inspect field datasets");
                present_fields += present > 0 ? 1 : 0;
            }
            for (const char* name : FULL_FIELD_NAMES) {
                const htri_t present = H5Lexists(fid, name, H5P_DEFAULT);
                if (present < 0) return fail("cannot inspect field datasets");
                present_fields += present > 0 ? 1 : 0;
            }
            if (present_fields != 0) {
                constexpr int EXPECTED_FIELDS =
                    static_cast<int>(EquilibriumSnapshot::HALF_FIELD_COUNT) +
                    static_cast<int>(EquilibriumSnapshot::FULL_FIELD_COUNT);
                if (present_fields != EXPECTED_FIELDS) {
                    return fail("incomplete derived-field datasets");
                }
                hsize_t half_dims[3] = {0, 0, 0};
                hsize_t full_dims[3] = {0, 0, 0};
                if (!get_dim_exact(HALF_FIELD_NAMES[0], 3, half_dims) ||
                    !get_dim_exact(FULL_FIELD_NAMES[0], 3, full_dims) ||
                    half_dims[0] != static_cast<hsize_t>(ns - 1) ||
                    full_dims[0] != static_cast<hsize_t>(ns) ||
                    half_dims[1] != full_dims[1] ||
                    half_dims[2] != full_dims[2] || half_dims[1] < 1 ||
                    half_dims[2] < 1 || half_dims[1] > INT_MAX ||
                    half_dims[2] > INT_MAX) {
                    return fail("malformed derived-field dimensions");
                }
                snapshot.nzeta = static_cast<int>(half_dims[1]);
                snapshot.ntheta = static_cast<int>(half_dims[2]);
                for (int c = 0; c < EquilibriumSnapshot::HALF_FIELD_COUNT;
                     ++c) {
                    if (!get_dbl_field(HALF_FIELD_NAMES[c], half_dims,
                                       snapshot.half_fields[c])) {
                        return fail("malformed half-grid field dataset (" +
                                    std::string(HALF_FIELD_NAMES[c]) + ")");
                    }
                }
                for (int c = 0; c < EquilibriumSnapshot::FULL_FIELD_COUNT;
                     ++c) {
                    if (!get_dbl_field(FULL_FIELD_NAMES[c], full_dims,
                                       snapshot.full_fields[c])) {
                        return fail("malformed full-grid field dataset (" +
                                    std::string(FULL_FIELD_NAMES[c]) + ")");
                    }
                }
            }
            if (report) {
                // Parse transactionally: a malformed late field must not leave
                // a partially populated report visible to the caller.
                RunReport parsed_report;
                int precision = 0, status = 0, total = 0, dirty = 0;
                if (!get_int_attr("precision", precision) ||
                    !get_int_attr("status", status) ||
                    !get_int_attr("total_iterations", total) ||
                    !get_int_attr("build_dirty", dirty)) {
                    return fail("missing run outcome attributes");
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
                if (!get_str_attr(fid, "revision",
                                  parsed_report.build.revision) ||
                    !get_str_attr(fid, "build_type",
                                  parsed_report.build.build_type) ||
                    !get_str_attr(fid, "precision_policy",
                                  parsed_report.build.precision_policy) ||
                    !get_str_attr(fid, "compile_flags",
                                  parsed_report.build.compile_flags) ||
                    !get_str_attr(fid, "source_path",
                                  parsed_report.input.source_path) ||
                    !get_str_attr(fid, "source_hash",
                                  parsed_report.input.source_hash) ||
                    !get_str_attr(fid, "gpu_name",
                                  parsed_report.runtime.gpu_name) ||
                    !get_str_attr(fid, "driver",
                                  parsed_report.runtime.driver) ||
                    !get_str_attr(fid, "runtime",
                                  parsed_report.runtime.runtime) ||
                    !get_str_attr(fid, "toolkit",
                                  parsed_report.runtime.toolkit)) {
                    return fail("missing provenance attributes");
                }
                hsize_t d_stages[1] = {0};
                if (!get_dim_exact("stage_ns", 1, d_stages)) {
                    return fail("missing/malformed stage_ns");
                }
                const size_t nstages = (size_t)d_stages[0];
                // restart_iteration is OPTIONAL only in the sense of being
                // absent: when the dataset exists it must be rank 1 (a rank-0
                // or rank-2 dataset is malformed and must fail, not be
                // skipped).
                hsize_t d_restarts[1] = {0};
                size_t nrestarts = 0;
                {
                    H5Closer probe(
                        H5Dopen2(fid, "restart_iteration", H5P_DEFAULT));
                    if (probe.get() >= 0 &&
                        !get_dim_exact("restart_iteration", 1, d_restarts)) {
                        return fail("malformed restart_iteration dataset");
                    }
                    if (probe.get() >= 0) nrestarts = (size_t)d_restarts[0];
                }
                // Documented stage/restart resource caps (handoff §4).
                if (nstages > cumes::io_detail::MAX_STAGE_COUNT ||
                    nrestarts > cumes::io_detail::MAX_STAGE_COUNT) {
                    return fail(
                        "stage/restart dimensions exceed the resource cap");
                }
                std::vector<int> stage_ns, stage_iter, stage_conv, rst_off;
                std::vector<double> st_fsqr, st_fsqz, st_fsql;
                std::vector<int> rst_iter;
                if (!get_int_arr("stage_ns", stage_ns, nstages) ||
                    !get_int_arr("stage_iterations", stage_iter, nstages) ||
                    !get_int_arr("stage_converged", stage_conv, nstages) ||
                    !get_int_arr("restart_stage_offset", rst_off, nstages) ||
                    !get_dbl_arr("stage_fsqr", st_fsqr, nstages) ||
                    !get_dbl_arr("stage_fsqz", st_fsqz, nstages) ||
                    !get_dbl_arr("stage_fsql", st_fsql, nstages) ||
                    (nrestarts > 0 &&
                     !get_int_arr("restart_iteration", rst_iter, nrestarts))) {
                    return fail("stage history read failed");
                }
                // Validate the offsets BEFORE they index rst_iter
                // (completion-plan follow-up §2.2): negative, descending,
                // oversized, or non-zero first offsets must fail cleanly
                // instead of reading out of bounds.
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
                // All-or-nothing: when the mpol attribute exists the full
                // record is required (pre-record containers keep the
                // default-empty InputParams); every read gets the
                // exact-rank/type/extent hardening above.
                {
                    H5Closer aid(H5Aopen(fid, "mpol", H5P_DEFAULT));
                    if (aid.get() >= 0) {
                        InputParams ip;
                        if (!get_int_attr("mpol", ip.mpol) ||
                            !get_int_attr("ntor", ip.ntor) ||
                            !get_int_attr("nfp", ip.nfp) ||
                            !get_int_attr("ntheta", ip.ntheta) ||
                            !get_int_attr("nzeta", ip.nzeta) ||
                            !get_int_attr("ncurr", ip.ncurr) ||
                            !get_dbl_attr("delt", ip.delt) ||
                            !get_dbl_attr("phiedge", ip.phiedge) ||
                            !get_dbl_attr("pres_scale", ip.pres_scale) ||
                            !get_dbl_attr("adiabatic_index",
                                          ip.adiabatic_index) ||
                            !get_dbl_attr("spres_ped", ip.spres_ped) ||
                            !get_dbl_attr("bloat", ip.bloat) ||
                            !get_dbl_attr("curtor", ip.curtor) ||
                            !get_dbl_attr("tcon0", ip.tcon0)) {
                            return fail("malformed embedded input record");
                        }
                        {
                            H5Closer lfreeb_attr(
                                H5Aopen(fid, "lfreeb", H5P_DEFAULT));
                            if (lfreeb_attr.get() >= 0) {
                                int lfreeb = 0;
                                if (!get_int_attr("lfreeb", lfreeb)) {
                                    return fail(
                                        "malformed embedded input record");
                                }
                                ip.lfreeb = lfreeb != 0;
                            }
                            H5Closer nvacskip_attr(
                                H5Aopen(fid, "nvacskip", H5P_DEFAULT));
                            if (nvacskip_attr.get() >= 0 &&
                                !get_int_attr("nvacskip", ip.nvacskip)) {
                                return fail("malformed embedded input record");
                            }
                            get_str_attr(fid, "mgrid_file", ip.mgrid_file);
                            get_str_attr(fid, "coils_file", ip.coils_file);
                            get_str_attr(fid, "makegrid_parameters_file",
                                         ip.makegrid_parameters_file);
                            H5Closer present_attr(
                                H5Aopen(fid, "makegrid_parameters_present",
                                        H5P_DEFAULT));
                            if (present_attr.get() >= 0) {
                                int present = 0;
                                if (!get_int_attr("makegrid_parameters_present",
                                                  present)) {
                                    return fail(
                                        "malformed embedded input record");
                                }
                                if (present != 0) {
                                    MakegridParametersSpec parameters;
                                    int normalize = 0;
                                    int symmetry = 0;
                                    if (!get_int_attr(
                                            "makegrid_normalize_by_currents",
                                            normalize) ||
                                        !get_int_attr(
                                            "makegrid_assume_stellarator_"
                                            "symmetry",
                                            symmetry) ||
                                        !get_int_attr(
                                            "makegrid_number_of_field_periods",
                                            parameters
                                                .number_of_field_periods) ||
                                        !get_dbl_attr(
                                            "makegrid_r_grid_minimum",
                                            parameters.r_grid_minimum) ||
                                        !get_dbl_attr(
                                            "makegrid_r_grid_maximum",
                                            parameters.r_grid_maximum) ||
                                        !get_int_attr(
                                            "makegrid_number_of_r_grid_points",
                                            parameters
                                                .number_of_r_grid_points) ||
                                        !get_dbl_attr(
                                            "makegrid_z_grid_minimum",
                                            parameters.z_grid_minimum) ||
                                        !get_dbl_attr(
                                            "makegrid_z_grid_maximum",
                                            parameters.z_grid_maximum) ||
                                        !get_int_attr(
                                            "makegrid_number_of_z_grid_points",
                                            parameters
                                                .number_of_z_grid_points) ||
                                        !get_int_attr(
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
                        {
                            auto get_vec =
                                [&](const char* name,
                                    std::vector<double>& out) -> bool {
                                hsize_t dims[1] = {0};
                                if (!get_dim_exact(name, 1, dims)) return false;
                                return get_dbl_arr(name, out, (size_t)dims[0]);
                            };
                            if (!get_vec("am", ip.am) ||
                                !get_vec("ac", ip.ac) ||
                                !get_vec("ai", ip.ai) ||
                                !get_vec("aphi", ip.aphi) ||
                                !get_vec("raxis_c", ip.raxis_c) ||
                                !get_vec("zaxis_s", ip.zaxis_s)) {
                                return fail("malformed embedded input record");
                            }
                            if (H5Lexists(fid, "extcur", H5P_DEFAULT) > 0 &&
                                !get_vec("extcur", ip.extcur)) {
                                return fail("malformed embedded input record");
                            }
                        }
                        {
                            hsize_t d_in[1] = {0};
                            if (!get_dim_exact("stage_in_ns", 1, d_in)) {
                                return fail("malformed embedded input record");
                            }
                            const size_t nstages_in = (size_t)d_in[0];
                            if (nstages_in >
                                cumes::io_detail::MAX_STAGE_COUNT) {
                                return fail(
                                    "embedded input stage count exceeds "
                                    "the resource cap");
                            }
                            std::vector<int> stg_in_ns, stg_max_iter;
                            std::vector<double> stg_ftol;
                            if (!get_int_arr("stage_in_ns", stg_in_ns,
                                             nstages_in) ||
                                !get_int_arr("stage_max_iter", stg_max_iter,
                                             nstages_in) ||
                                !get_dbl_arr("stage_ftol", stg_ftol,
                                             nstages_in)) {
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
                        {
                            hsize_t d_r[1] = {0}, d_z[1] = {0}, d_b[2] = {0, 0};
                            if (!get_dim_exact("rbc_m", 1, d_r) ||
                                !get_dim_exact("zbs_m", 1, d_z) ||
                                !get_dim_exact("rbcc", 2, d_b)) {
                                return fail("malformed embedded input record");
                            }
                            const size_t nrbc = (size_t)d_r[0],
                                         nzbs = (size_t)d_z[0];
                            if (!get_int_arr("rbc_m", ip.rbc_m, nrbc) ||
                                !get_int_arr("rbc_n", ip.rbc_n, nrbc) ||
                                !get_dbl_arr("rbc_value", ip.rbc_value, nrbc) ||
                                !get_int_arr("zbs_m", ip.zbs_m, nzbs) ||
                                !get_int_arr("zbs_n", ip.zbs_n, nzbs) ||
                                !get_dbl_arr("zbs_value", ip.zbs_value, nzbs) ||
                                !get_dbl_mat("rbcc", ip.rbcc, (size_t)d_b[0],
                                             (size_t)d_b[1]) ||
                                !get_dbl_mat("rbss", ip.rbss, (size_t)d_b[0],
                                             (size_t)d_b[1]) ||
                                !get_dbl_mat("zbsc", ip.zbsc, (size_t)d_b[0],
                                             (size_t)d_b[1]) ||
                                !get_dbl_mat("zbcs", ip.zbcs, (size_t)d_b[0],
                                             (size_t)d_b[1])) {
                                return fail("malformed embedded input record");
                            }
                        }
                        // The schema/profile-type tags are informational; an
                        // absent or unreadable one keeps the default (an
                        // older container lacks the profile types ->
                        // "power_series").
                        std::string schema_tag;
                        if (get_str_attr(fid, "schema", schema_tag)) {
                            ip.schema = schema_tag;
                        }
                        std::string pmass_tag, piota_tag, pcurr_tag;
                        if (get_str_attr(fid, "pmass_type", pmass_tag)) {
                            ip.pmass_type = pmass_tag;
                        }
                        if (get_str_attr(fid, "piota_type", piota_tag)) {
                            ip.piota_type = piota_tag;
                        }
                        if (get_str_attr(fid, "pcurr_type", pcurr_tag)) {
                            ip.pcurr_type = pcurr_tag;
                        }
                        parsed_report.input_params = std::move(ip);
                    }
                }
                report->get() = std::move(parsed_report);
            }
            H5Fclose(fid);
            return snapshot;
        } catch (const std::bad_alloc&) {
            H5Fclose(fid);
            return Result<EquilibriumSnapshot>(
                "HDF5: allocation failed (dimensions implausible or corrupt)");
        } catch (const std::length_error&) {
            H5Fclose(fid);
            return Result<EquilibriumSnapshot>(
                "HDF5: allocation failed (dimensions implausible or corrupt)");
        }
    }
};

}  // namespace

std::unique_ptr<Writer> make_hdf5_v1_writer() {
    return std::make_unique<Hdf5V1Writer>();
}
std::unique_ptr<Reader> make_hdf5_v1_reader() {
    return std::make_unique<Hdf5V1Reader>();
}

}  // namespace cumes
