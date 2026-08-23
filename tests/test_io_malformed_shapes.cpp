// test_io_malformed_shapes.cpp — v1 container rank/type/extent hardening
// (overhaul-history.md, reader-rank entry §5).
//
// The v1 NetCDF/HDF5 readers must prove the EXACT rank, datatype, and extent
// of every object before reading into a fixed-size or sized host buffer. A
// malicious container that lies about shape (rank 1/3 state families, swapped
// dimensions, scalar variables encoded as arrays, rank 0/2 stage datasets,
// multi-element attributes, wrong datatypes, or dimensions beyond the
// documented bounds) must fail with a typed Result error — no crash, no
// out-of-bounds access, no partial RunReport, no excessive allocation.
//
// Host-only on purpose: the ASan/UBSan twin (asan_test_io_malformed_shapes)
// runs this exact source under the host sanitizers.
#include "cumes/io/reader.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/io/writer.hpp"
#include "cumes/io/writer_helpers.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>
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

using cumes::OutputFormat;
using cumes::Reader;
using cumes::RunReport;

// Per-test temp directory with RAII cleanup.
class TempDir {
   public:
    TempDir() {
        char tmpl[] = "/tmp/cumes_io_shapes_XXXXXX";
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

// ---------------------------------------------------------------------------
// NetCDF fixture: a minimal-but-schema-complete v1 container (ns=3, mnmax=2,
// nstages=2, nrestarts=1) with ONE shape mutation applied.
// ---------------------------------------------------------------------------
#ifdef CUMES_HAVE_NETCDF
enum class NcMutation {
    NONE,
    FAMILY_RANK_1,         // rmncc declared with rank 1
    FAMILY_RANK_3,         // rmncc declared with rank 3
    FAMILY_SWAPPED_DIMS,   // rmncc declared [mnmax, ns] instead of [ns, mnmax]
    SCALAR_AS_ARRAY,       // "precision" declared as a rank-1 array of 3 ints
    STAGE_RANK_0,          // stage_ns declared scalar
    STAGE_RANK_2,          // stage_ns declared rank 2
    WRONG_FAMILY_TYPE,     // rmncc declared NC_INT
    RESOURCE_CAP,          // state product exceeds the practical reader cap
    HUGE_NS,               // ns dimension = 2^40 (beyond the INT_MAX bound)
    DIRTY_OUT_OF_RANGE,    // build_dirty = 2
    STAGE_NS_NONPOSITIVE,  // stage_ns contains 0
    STAGE_CONV_OUT_OF_RANGE,  // stage_converged contains 2
    RESTART_NEGATIVE,         // restart_iteration contains -1
};

static bool write_netcdf_fixture(const std::string& path, NcMutation mut) {
    int ncid = -1;
    // CDF-5 (NC_64BIT_DATA) is required for the HUGE_NS case: the classic
    // and 64-bit-offset formats cap dimension lengths at 32 bits, which can
    // never exceed INT_MAX. The reader (nc_open) accepts the format
    // transparently.
    if (nc_create(path.c_str(), NC_CLOBBER | NC_64BIT_DATA, &ncid) !=
        NC_NOERR) {
        return false;
    }
#define NC_F(expr)                            \
    do {                                      \
        if ((expr) != NC_NOERR) return false; \
    } while (0)
    const bool ok = [&]() -> bool {
        const size_t huge = (size_t)1 << 40;
        const bool resource_case = mut == NcMutation::RESOURCE_CAP;
        const bool dimension_only = resource_case || mut == NcMutation::HUGE_NS;
        const size_t capped_ns =
            cumes::io_detail::MAX_STATE_ELEMENTS_PER_FAMILY / 2 + 1;
        const size_t ns_len = (mut == NcMutation::HUGE_NS)
                                  ? huge
                                  : (resource_case ? capped_ns : 3);
        int d_ns, d_mn, d_st, d_rs;
        NC_F(nc_def_dim(ncid, "ns", ns_len, &d_ns));
        NC_F(nc_def_dim(ncid, "mnmax", 2, &d_mn));
        NC_F(nc_def_dim(ncid, "nstages", 2, &d_st));
        NC_F(nc_def_dim(ncid, "nrestarts", 1, &d_rs));
        const int fam_dims[2] = {d_ns, d_mn};
        const char* fams[6] = {"rmncc", "zmnsc", "lmnsc",
                               "rmnss", "zmncs", "lmncs"};
        int v_fam[6];
        // Dimension-only resource fixtures omit family variables so CDF-5
        // does not reserve their prospective multi-gigabyte file size. The
        // reader must reject from dimensions before looking up a family.
        for (int c = 0; c < 6 && !dimension_only; ++c) {
            if (c == 0 && mut == NcMutation::FAMILY_RANK_1) {
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 1, &d_ns, &v_fam[c]));
            } else if (c == 0 && mut == NcMutation::FAMILY_RANK_3) {
                const int three[3] = {d_ns, d_mn, d_st};
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 3, three, &v_fam[c]));
            } else if (c == 0 && mut == NcMutation::FAMILY_SWAPPED_DIMS) {
                const int swapped[2] = {d_mn, d_ns};
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 2, swapped,
                                &v_fam[c]));
            } else if (c == 0 && mut == NcMutation::WRONG_FAMILY_TYPE) {
                NC_F(nc_def_var(ncid, fams[c], NC_INT, 2, fam_dims, &v_fam[c]));
            } else {
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 2, fam_dims,
                                &v_fam[c]));
            }
        }
        int v_prec, v_status, v_total, v_dirty;
        if (mut == NcMutation::SCALAR_AS_ARRAY) {
            NC_F(nc_def_var(ncid, "precision", NC_INT, 1, &d_st, &v_prec));
        } else {
            NC_F(nc_def_var(ncid, "precision", NC_INT, 0, nullptr, &v_prec));
        }
        NC_F(nc_def_var(ncid, "status", NC_INT, 0, nullptr, &v_status));
        NC_F(
            nc_def_var(ncid, "total_iterations", NC_INT, 0, nullptr, &v_total));
        NC_F(nc_def_var(ncid, "build_dirty", NC_INT, 0, nullptr, &v_dirty));
        int v_ns, v_iter, v_conv, v_fsqr, v_fsqz, v_fsql, v_off, v_riter;
        if (mut == NcMutation::STAGE_RANK_0) {
            NC_F(nc_def_var(ncid, "stage_ns", NC_INT, 0, nullptr, &v_ns));
        } else if (mut == NcMutation::STAGE_RANK_2) {
            const int two2[2] = {d_st, d_st};
            NC_F(nc_def_var(ncid, "stage_ns", NC_INT, 2, two2, &v_ns));
        } else {
            NC_F(nc_def_var(ncid, "stage_ns", NC_INT, 1, &d_st, &v_ns));
        }
        NC_F(nc_def_var(ncid, "stage_iterations", NC_INT, 1, &d_st, &v_iter));
        NC_F(nc_def_var(ncid, "stage_converged", NC_INT, 1, &d_st, &v_conv));
        NC_F(nc_def_var(ncid, "stage_fsqr", NC_DOUBLE, 1, &d_st, &v_fsqr));
        NC_F(nc_def_var(ncid, "stage_fsqz", NC_DOUBLE, 1, &d_st, &v_fsqz));
        NC_F(nc_def_var(ncid, "stage_fsql", NC_DOUBLE, 1, &d_st, &v_fsql));
        NC_F(
            nc_def_var(ncid, "restart_stage_offset", NC_INT, 1, &d_st, &v_off));
        NC_F(nc_def_var(ncid, "restart_iteration", NC_INT, 1, &d_rs, &v_riter));
        const char* str_attrs[][2] = {{"revision", "r1"},
                                      {"build_type", "Release"},
                                      {"precision_policy", "verify-double"},
                                      {"compile_flags", ""},
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
        // Dimension-only fixtures are rejected before family reads.
        if (!dimension_only) {
            for (int c = 0; c < 6; ++c) {
                NC_F(nc_put_var_double(ncid, v_fam[c], fambuf));
            }
        }
        const int zero = 0;
        if (mut == NcMutation::SCALAR_AS_ARRAY) {
            const int three[3] = {0, 0, 0};
            NC_F(nc_put_var_int(ncid, v_prec, three));
        } else {
            NC_F(nc_put_var_int(ncid, v_prec, &zero));
        }
        NC_F(nc_put_var_int(ncid, v_status, &zero));
        NC_F(nc_put_var_int(ncid, v_total, &zero));
        const int dirty = (mut == NcMutation::DIRTY_OUT_OF_RANGE) ? 2 : 0;
        NC_F(nc_put_var_int(ncid, v_dirty, &dirty));
        if (mut != NcMutation::STAGE_RANK_0 &&
            mut != NcMutation::STAGE_RANK_2) {
            std::vector<int> stage_ns(2, 3), stage_iter(2, 1), stage_conv(2, 1);
            if (mut == NcMutation::STAGE_NS_NONPOSITIVE) stage_ns[0] = 0;
            if (mut == NcMutation::STAGE_CONV_OUT_OF_RANGE) stage_conv[0] = 2;
            std::vector<double> stage_res(2, 0.0);
            const int offs[2] = {0, 1};  // valid offsets for the control case
            NC_F(nc_put_var_int(ncid, v_ns, stage_ns.data()));
            NC_F(nc_put_var_int(ncid, v_iter, stage_iter.data()));
            NC_F(nc_put_var_int(ncid, v_conv, stage_conv.data()));
            NC_F(nc_put_var_double(ncid, v_fsqr, stage_res.data()));
            NC_F(nc_put_var_double(ncid, v_fsqz, stage_res.data()));
            NC_F(nc_put_var_double(ncid, v_fsql, stage_res.data()));
            NC_F(nc_put_var_int(ncid, v_off, offs));
        }
        const int restart_iteration =
            (mut == NcMutation::RESTART_NEGATIVE) ? -1 : 7;
        NC_F(nc_put_var_int(ncid, v_riter, &restart_iteration));
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

// ---------------------------------------------------------------------------
// HDF5 fixture: the same minimal v1 container with ONE shape mutation.
// ---------------------------------------------------------------------------
#ifdef CUMES_HAVE_HDF5
enum class H5Mutation {
    NONE,
    RMNCC_RANK_1,             // rmncc with rank 1
    RMNCC_RANK_3,             // rmncc with rank 3
    FAMILY_EXTENT_DIFFERS,    // zmnsc with extent [ns, mnmax+1]
    STAGE_RANK_0,             // stage_ns scalar
    STAGE_RANK_2,             // stage_ns rank 2
    INT_ATTR_ARRAY,           // "precision" attribute with 3 elements
    STR_ATTR_ARRAY,           // "revision" attribute with 2 elements
    VARIABLE_STR_ATTR,        // "revision" is a scalar variable-length string
    WRONG_FAMILY_TYPE,        // rmncc stored as H5T_NATIVE_INT
    RESOURCE_CAP,             // state product exceeds the practical reader cap
    HUGE_NS,                  // family extent ns = 2^40
    UNSIGNED_INT_ATTR,        // scalar precision attribute is unsigned
    STAGE_UNSIGNED,           // stage_ns stored as unsigned 32-bit
    STAGE_U8,                 // stage_ns stored as unsigned 8-bit
    STAGE_I64,                // stage_ns stored as signed 64-bit
    DIRTY_OUT_OF_RANGE,       // build_dirty = 2
    STAGE_NS_NONPOSITIVE,     // stage_ns contains 0
    STAGE_CONV_OUT_OF_RANGE,  // stage_converged contains 2
    RESTART_NEGATIVE,         // restart_iteration contains -1
};

static bool write_hdf5_fixture(const std::string& path, H5Mutation mut) {
    hid_t fid =
        H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) return false;
#define H5_F(expr)                    \
    do {                              \
        if ((expr) < 0) return false; \
    } while (0)
    const bool ok = [&]() -> bool {
        auto put_int_attr = [&](const char* name, int value) -> herr_t {
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
        auto put_str_attr = [&](const char* name,
                                const std::string& value) -> herr_t {
            hid_t s1 = H5Tcopy(H5T_C_S1);
            if (s1 < 0) return -1;
            herr_t r0 = H5Tset_size(s1, value.size() + 1);
            if (r0 < 0) {
                H5Tclose(s1);
                return -1;
            }
            hid_t sid = H5Screate(H5S_SCALAR);
            if (sid < 0) {
                H5Tclose(s1);
                return -1;
            }
            hid_t aid =
                H5Acreate2(fid, name, s1, sid, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sid);
            if (aid < 0) {
                H5Tclose(s1);
                return -1;
            }
            const herr_t r = H5Awrite(aid, s1, value.c_str());
            H5Aclose(aid);
            H5Tclose(s1);
            return r;
        };
        auto put_array = [&](const char* name, hid_t dtype, hsize_t len,
                             const void* data) -> herr_t {
            hid_t sp = H5Screate_simple(1, &len, nullptr);
            if (sp < 0) return -1;
            hid_t ds = H5Dcreate2(fid, name, dtype, sp, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return -1;
            const herr_t r =
                H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            H5Dclose(ds);
            return r;
        };

        const hsize_t huge = (hsize_t)1 << 40;
        const hsize_t capped_ns =
            (hsize_t)cumes::io_detail::MAX_STATE_ELEMENTS_PER_FAMILY / 2 + 1;
        const hsize_t ns_ext =
            (mut == H5Mutation::HUGE_NS)
                ? huge
                : ((mut == H5Mutation::RESOURCE_CAP) ? capped_ns : 3);
        const hsize_t state_dims[2] = {ns_ext, 2};
        const char* fams[6] = {"rmncc", "zmnsc", "lmnsc",
                               "rmnss", "zmncs", "lmncs"};
        const double fambuf[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int c = 0; c < 6; ++c) {
            hid_t sp = -1;
            if (c == 0 && mut == H5Mutation::RMNCC_RANK_1) {
                sp = H5Screate_simple(1, &ns_ext, nullptr);
            } else if (c == 0 && mut == H5Mutation::RMNCC_RANK_3) {
                const hsize_t three[3] = {3, 2, 2};
                sp = H5Screate_simple(3, three, nullptr);
            } else if (c == 1 && mut == H5Mutation::FAMILY_EXTENT_DIFFERS) {
                const hsize_t differs[2] = {ns_ext, 3};
                sp = H5Screate_simple(2, differs, nullptr);
            } else {
                sp = H5Screate_simple(2, state_dims, nullptr);
            }
            if (sp < 0) return false;
            // Write the 6-value buffer only when the dataspace holds exactly
            // 6 points: mutated shapes (rank 1/3, differing extents, huge
            // dimensions) are left unwritten ON PURPOSE — the reader must
            // reject at the shape/bound checks before reading any value, and
            // a whole-slab write of a mutated extent would overread fambuf.
            const hssize_t npts = H5Sget_simple_extent_npoints(sp);
            const hid_t dtype = (c == 0 && mut == H5Mutation::WRONG_FAMILY_TYPE)
                                    ? H5T_NATIVE_INT
                                    : H5T_NATIVE_DOUBLE;
            hid_t ds = H5Dcreate2(fid, fams[c], dtype, sp, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return false;
            if (npts == 6 && H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                      fambuf) < 0) {
                H5Dclose(ds);
                return false;
            }
            H5Dclose(ds);
        }
        if (mut == H5Mutation::INT_ATTR_ARRAY) {
            // "precision" as a 3-element integer array attribute.
            const hsize_t ext[1] = {3};
            hid_t sp = H5Screate_simple(1, ext, nullptr);
            if (sp < 0) return false;
            hid_t aid = H5Acreate2(fid, "precision", H5T_NATIVE_INT, sp,
                                   H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (aid < 0) return false;
            const int three[3] = {0, 0, 0};
            const herr_t r = H5Awrite(aid, H5T_NATIVE_INT, three);
            H5Aclose(aid);
            H5_F(r);
        } else if (mut == H5Mutation::UNSIGNED_INT_ATTR) {
            hid_t sp = H5Screate(H5S_SCALAR);
            if (sp < 0) return false;
            hid_t aid = H5Acreate2(fid, "precision", H5T_NATIVE_UINT, sp,
                                   H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (aid < 0) return false;
            const unsigned int zero = 0;
            const herr_t r = H5Awrite(aid, H5T_NATIVE_UINT, &zero);
            H5Aclose(aid);
            H5_F(r);
        } else {
            H5_F(put_int_attr("precision", 0));
        }
        H5_F(put_int_attr("status", 0));
        H5_F(put_int_attr("total_iterations", 0));
        H5_F(put_int_attr("build_dirty",
                          mut == H5Mutation::DIRTY_OUT_OF_RANGE ? 2 : 0));
        if (mut == H5Mutation::STR_ATTR_ARRAY) {
            // "revision" as a 2-element string array attribute.
            const hsize_t ext[1] = {2};
            hid_t sp = H5Screate_simple(1, ext, nullptr);
            if (sp < 0) return false;
            hid_t s1 = H5Tcopy(H5T_C_S1);
            if (s1 < 0) {
                H5Sclose(sp);
                return false;
            }
            if (H5Tset_size(s1, 3) < 0) {
                H5Tclose(s1);
                H5Sclose(sp);
                return false;
            }
            hid_t aid =
                H5Acreate2(fid, "revision", s1, sp, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (aid < 0) {
                H5Tclose(s1);
                return false;
            }
            const char* two[2] = {"ab", "cd"};
            const herr_t r = H5Awrite(aid, s1, two);
            H5Aclose(aid);
            H5Tclose(s1);
            H5_F(r);
        } else if (mut == H5Mutation::VARIABLE_STR_ATTR) {
            hid_t s1 = H5Tcopy(H5T_C_S1);
            if (s1 < 0) return false;
            if (H5Tset_size(s1, H5T_VARIABLE) < 0) {
                H5Tclose(s1);
                return false;
            }
            hid_t sid = H5Screate(H5S_SCALAR);
            if (sid < 0) {
                H5Tclose(s1);
                return false;
            }
            hid_t aid =
                H5Acreate2(fid, "revision", s1, sid, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sid);
            if (aid < 0) {
                H5Tclose(s1);
                return false;
            }
            const char* value = "r1";
            const herr_t r = H5Awrite(aid, s1, &value);
            H5Aclose(aid);
            H5Tclose(s1);
            H5_F(r);
        } else {
            H5_F(put_str_attr("revision", "r1"));
        }
        H5_F(put_str_attr("build_type", "Release"));
        H5_F(put_str_attr("precision_policy", "verify-double"));
        H5_F(put_str_attr("compile_flags", ""));
        H5_F(put_str_attr("source_path", "in.json"));
        H5_F(put_str_attr("source_hash", "h"));
        H5_F(put_str_attr("gpu_name", "g"));
        H5_F(put_str_attr("driver", "d"));
        H5_F(put_str_attr("runtime", "rt"));
        H5_F(put_str_attr("toolkit", "t"));
        std::vector<int> stage_ns(2, 3), stage_iter(2, 1), stage_conv(2, 1);
        std::vector<double> stage_res(2, 0.0);
        std::vector<int> rst_iter(1, 7);
        if (mut == H5Mutation::STAGE_NS_NONPOSITIVE) stage_ns[0] = 0;
        if (mut == H5Mutation::STAGE_CONV_OUT_OF_RANGE) stage_conv[0] = 2;
        if (mut == H5Mutation::RESTART_NEGATIVE) rst_iter[0] = -1;
        if (mut == H5Mutation::STAGE_RANK_0) {
            hid_t sp = H5Screate(H5S_SCALAR);
            if (sp < 0) return false;
            hid_t ds = H5Dcreate2(fid, "stage_ns", H5T_NATIVE_INT, sp,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return false;
            const int zero = 0;
            const herr_t r = H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                      H5P_DEFAULT, &zero);
            H5Dclose(ds);
            H5_F(r);
        } else if (mut == H5Mutation::STAGE_RANK_2) {
            const hsize_t two2[2] = {2, 2};
            hid_t sp = H5Screate_simple(2, two2, nullptr);
            if (sp < 0) return false;
            hid_t ds = H5Dcreate2(fid, "stage_ns", H5T_NATIVE_INT, sp,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return false;
            const int four[4] = {0, 0, 0, 0};
            const herr_t r = H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                      H5P_DEFAULT, four);
            H5Dclose(ds);
            H5_F(r);
        } else if (mut == H5Mutation::STAGE_UNSIGNED) {
            const unsigned int values[2] = {3, 3};
            H5_F(put_array("stage_ns", H5T_NATIVE_UINT, 2, values));
        } else if (mut == H5Mutation::STAGE_U8) {
            const unsigned char values[2] = {3, 3};
            H5_F(put_array("stage_ns", H5T_NATIVE_UCHAR, 2, values));
        } else if (mut == H5Mutation::STAGE_I64) {
            const long long values[2] = {3, 3};
            H5_F(put_array("stage_ns", H5T_NATIVE_LLONG, 2, values));
        } else {
            H5_F(put_array("stage_ns", H5T_NATIVE_INT, 2, stage_ns.data()));
        }
        H5_F(put_array("stage_iterations", H5T_NATIVE_INT, 2,
                       stage_iter.data()));
        H5_F(
            put_array("stage_converged", H5T_NATIVE_INT, 2, stage_conv.data()));
        H5_F(put_array("stage_fsqr", H5T_NATIVE_DOUBLE, 2, stage_res.data()));
        H5_F(put_array("stage_fsqz", H5T_NATIVE_DOUBLE, 2, stage_res.data()));
        H5_F(put_array("stage_fsql", H5T_NATIVE_DOUBLE, 2, stage_res.data()));
        {
            const int offs[2] = {0, 1};  // valid offsets for the control case
            H5_F(put_array("restart_stage_offset", H5T_NATIVE_INT, 2, offs));
        }
        H5_F(
            put_array("restart_iteration", H5T_NATIVE_INT, 1, rst_iter.data()));
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
// One backend: the unmutated container must round-trip; every mutation must
// fail with a typed Result error (no crash, no partial report).
// ---------------------------------------------------------------------------
#ifdef CUMES_HAVE_NETCDF
static void test_netcdf(const TempDir& dir) {
    struct Case {
        NcMutation mut;
        const char* label;
        const char* error_contains;
    };
    const Case cases[] = {
        {NcMutation::NONE, "valid control", nullptr},
        {NcMutation::FAMILY_RANK_1, "rmncc rank 1", nullptr},
        {NcMutation::FAMILY_RANK_3, "rmncc rank 3", nullptr},
        {NcMutation::FAMILY_SWAPPED_DIMS, "rmncc swapped dimensions", nullptr},
        {NcMutation::SCALAR_AS_ARRAY, "scalar outcome as array", nullptr},
        {NcMutation::STAGE_RANK_0, "stage_ns rank 0", nullptr},
        {NcMutation::STAGE_RANK_2, "stage_ns rank 2", nullptr},
        {NcMutation::WRONG_FAMILY_TYPE, "rmncc wrong datatype", nullptr},
        {NcMutation::RESOURCE_CAP, "state resource cap", "resource cap"},
        {NcMutation::HUGE_NS, "ns beyond INT_MAX", "bad dimensions"},
        {NcMutation::DIRTY_OUT_OF_RANGE, "build_dirty out of range", nullptr},
        {NcMutation::STAGE_NS_NONPOSITIVE, "stage_ns nonpositive", nullptr},
        {NcMutation::STAGE_CONV_OUT_OF_RANGE, "stage_converged out of range",
         nullptr},
        {NcMutation::RESTART_NEGATIVE, "restart iteration negative", nullptr},
    };
    for (const Case& c : cases) {
        const std::string path = dir.path() + "/" + c.label + ".nc";
        check(write_netcdf_fixture(path, c.mut),
              format("netcdf: fixture written ({})", c.label));
        std::unique_ptr<Reader> reader = make_reader(OutputFormat::NETCDF);
        RunReport rep;
        rep.build.revision = "sentinel";
        rep.total_effective_iterations = 12345;
        const auto res = reader->read(path, &rep);
        remove(path.c_str());
        if (c.mut == NcMutation::NONE) {
            check(res.has_value(), format("netcdf: {}", c.label));
        } else {
            check(!res.has_value(),
                  format("netcdf: {} rejected with typed failure", c.label));
            check(rep.build.revision == "sentinel" &&
                      rep.total_effective_iterations == 12345,
                  format("netcdf: {} leaves report transactional", c.label));
            if (c.error_contains != nullptr && !res.has_value()) {
                check(res.error().find(c.error_contains) != std::string::npos,
                      format("netcdf: {} fails at expected boundary", c.label));
            }
        }
    }
}
#endif

#ifdef CUMES_HAVE_HDF5
static void test_hdf5(const TempDir& dir) {
    struct Case {
        H5Mutation mut;
        const char* label;
        const char* error_contains;
    };
    const Case cases[] = {
        {H5Mutation::NONE, "valid control", nullptr},
        {H5Mutation::RMNCC_RANK_1, "rmncc rank 1", nullptr},
        {H5Mutation::RMNCC_RANK_3, "rmncc rank 3", nullptr},
        {H5Mutation::FAMILY_EXTENT_DIFFERS, "zmnsc extent differs", nullptr},
        {H5Mutation::STAGE_RANK_0, "stage_ns rank 0", nullptr},
        {H5Mutation::STAGE_RANK_2, "stage_ns rank 2", nullptr},
        {H5Mutation::INT_ATTR_ARRAY, "multi-element int attribute", nullptr},
        {H5Mutation::STR_ATTR_ARRAY, "multi-element string attribute", nullptr},
        {H5Mutation::VARIABLE_STR_ATTR, "variable-length string attribute",
         nullptr},
        {H5Mutation::WRONG_FAMILY_TYPE, "rmncc wrong datatype", nullptr},
        {H5Mutation::RESOURCE_CAP, "state resource cap", "resource cap"},
        {H5Mutation::HUGE_NS, "ns beyond INT_MAX", "bad state dimensions"},
        {H5Mutation::UNSIGNED_INT_ATTR, "unsigned scalar int attribute",
         nullptr},
        {H5Mutation::STAGE_UNSIGNED, "unsigned stage integer", nullptr},
        {H5Mutation::STAGE_U8, "8-bit stage integer", nullptr},
        {H5Mutation::STAGE_I64, "64-bit stage integer", nullptr},
        {H5Mutation::DIRTY_OUT_OF_RANGE, "build_dirty out of range", nullptr},
        {H5Mutation::STAGE_NS_NONPOSITIVE, "stage_ns nonpositive", nullptr},
        {H5Mutation::STAGE_CONV_OUT_OF_RANGE, "stage_converged out of range",
         nullptr},
        {H5Mutation::RESTART_NEGATIVE, "restart iteration negative", nullptr},
    };
    for (const Case& c : cases) {
        const std::string path = dir.path() + "/" + c.label + ".h5";
        check(write_hdf5_fixture(path, c.mut),
              format("hdf5: fixture written ({})", c.label));
        std::unique_ptr<Reader> reader = make_reader(OutputFormat::HDF5);
        RunReport rep;
        rep.build.revision = "sentinel";
        rep.total_effective_iterations = 12345;
        const auto res = reader->read(path, &rep);
        remove(path.c_str());
        if (c.mut == H5Mutation::NONE) {
            check(res.has_value(), format("hdf5: {}", c.label));
        } else {
            check(!res.has_value(),
                  format("hdf5: {} rejected with typed failure", c.label));
            check(rep.build.revision == "sentinel" &&
                      rep.total_effective_iterations == 12345,
                  format("hdf5: {} leaves report transactional", c.label));
            if (c.error_contains != nullptr && !res.has_value()) {
                check(res.error().find(c.error_contains) != std::string::npos,
                      format("hdf5: {} fails at expected boundary", c.label));
            }
        }
    }
}
#endif

int main() {
    TempDir dir;
    check(dir.ok(), "temp directory created");
    if (!dir.ok()) return 1;

#ifdef CUMES_HAVE_NETCDF
    test_netcdf(dir);
#else
    std::cout << "SKIP netcdf malformed-shape cases (backend not compiled)\n";
#endif
#ifdef CUMES_HAVE_HDF5
    test_hdf5(dir);
#else
    std::cout << "SKIP hdf5 malformed-shape cases (backend not compiled)\n";
#endif

    return summary();
}
