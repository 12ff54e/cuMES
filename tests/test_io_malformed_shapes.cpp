// test_io_malformed_shapes.cpp — v1 container rank/type/extent hardening
// (reader-rank-hardening-handoff.md §5).
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

using cumes::OutputFormat;
using cumes::OutputSchema;
using cumes::Reader;
using cumes::RunReport;

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            printf("PASS %s\n", msg);                                        \
        } else {                                                             \
            printf("FAIL %s\n", msg);                                        \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

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
    kNone,
    kFamilyRank1,        // rmncc declared with rank 1
    kFamilyRank3,        // rmncc declared with rank 3
    kFamilySwappedDims,  // rmncc declared [mnmax, ns] instead of [ns, mnmax]
    kScalarAsArray,      // "precision" declared as a rank-1 array of 3 ints
    kStageRank0,         // stage_ns declared scalar
    kStageRank2,         // stage_ns declared rank 2
    kWrongFamilyType,    // rmncc declared NC_INT
    kHugeNs,             // ns dimension = 2^40 (beyond the INT_MAX bound)
};

static bool writeNetcdfFixture(const std::string& path, NcMutation mut) {
    int ncid = -1;
    // CDF-5 (NC_64BIT_DATA) is required for the kHugeNs case: the classic
    // and 64-bit-offset formats cap dimension lengths at 32 bits, which can
    // never exceed INT_MAX. The reader (nc_open) accepts the format
    // transparently.
    if (nc_create(path.c_str(), NC_CLOBBER | NC_64BIT_DATA, &ncid) !=
        NC_NOERR) {
        return false;
    }
#define NC_F(expr)                                                            \
    do {                                                                      \
        if ((expr) != NC_NOERR) return false;                                 \
    } while (0)
    const bool ok = [&]() -> bool {
        const size_t huge = (size_t)1 << 40;
        const size_t ns_len = (mut == NcMutation::kHugeNs) ? huge : 3;
        int d_ns, d_mn, d_st, d_rs;
        NC_F(nc_def_dim(ncid, "ns", ns_len, &d_ns));
        NC_F(nc_def_dim(ncid, "mnmax", 2, &d_mn));
        NC_F(nc_def_dim(ncid, "nstages", 2, &d_st));
        NC_F(nc_def_dim(ncid, "nrestarts", 1, &d_rs));
        const int fam_dims[2] = {d_ns, d_mn};
        const char* fams[6] = {"rmncc", "zmnsc", "lmnsc",
                               "rmnss", "zmncs", "lmncs"};
        int v_fam[6];
        // kHugeNs: no family variables at all. Defining a variable over the
        // 2^40 surface dimension makes nc_enddef fail (the library checks
        // the prospective file size), and the reader must reject at the
        // dimension bound BEFORE reaching any family read anyway.
        for (int c = 0; c < 6 && mut != NcMutation::kHugeNs; ++c) {
            if (c == 0 && mut == NcMutation::kFamilyRank1) {
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 1, &d_ns, &v_fam[c]));
            } else if (c == 0 && mut == NcMutation::kFamilyRank3) {
                const int three[3] = {d_ns, d_mn, d_st};
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 3, three, &v_fam[c]));
            } else if (c == 0 && mut == NcMutation::kFamilySwappedDims) {
                const int swapped[2] = {d_mn, d_ns};
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 2, swapped,
                                &v_fam[c]));
            } else if (c == 0 && mut == NcMutation::kWrongFamilyType) {
                NC_F(nc_def_var(ncid, fams[c], NC_INT, 2, fam_dims, &v_fam[c]));
            } else {
                NC_F(nc_def_var(ncid, fams[c], NC_DOUBLE, 2, fam_dims,
                                &v_fam[c]));
            }
        }
        int v_prec, v_status, v_total, v_dirty;
        if (mut == NcMutation::kScalarAsArray) {
            NC_F(nc_def_var(ncid, "precision", NC_INT, 1, &d_st, &v_prec));
        } else {
            NC_F(nc_def_var(ncid, "precision", NC_INT, 0, nullptr, &v_prec));
        }
        NC_F(nc_def_var(ncid, "status", NC_INT, 0, nullptr, &v_status));
        NC_F(nc_def_var(ncid, "total_iterations", NC_INT, 0, nullptr,
                        &v_total));
        NC_F(nc_def_var(ncid, "build_dirty", NC_INT, 0, nullptr, &v_dirty));
        int v_ns, v_iter, v_conv, v_fsqr, v_fsqz, v_fsql, v_off, v_riter;
        if (mut == NcMutation::kStageRank0) {
            NC_F(nc_def_var(ncid, "stage_ns", NC_INT, 0, nullptr, &v_ns));
        } else if (mut == NcMutation::kStageRank2) {
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
        NC_F(nc_def_var(ncid, "restart_stage_offset", NC_INT, 1, &d_st,
                        &v_off));
        NC_F(nc_def_var(ncid, "restart_iteration", NC_INT, 1, &d_rs,
                        &v_riter));
        const char* str_attrs[][2] = {
            {"revision", "r1"},
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
        // kHugeNs: the families span the 2^40 surface dimension; writing them
        // whole would itself be enormous — the reader must reject at the
        // dimension bound BEFORE reaching any family read, so the values are
        // left unwritten on purpose.
        if (mut != NcMutation::kHugeNs) {
            for (int c = 0; c < 6; ++c) {
                NC_F(nc_put_var_double(ncid, v_fam[c], fambuf));
            }
        }
        const int zero = 0;
        if (mut == NcMutation::kScalarAsArray) {
            const int three[3] = {0, 0, 0};
            NC_F(nc_put_var_int(ncid, v_prec, three));
        } else {
            NC_F(nc_put_var_int(ncid, v_prec, &zero));
        }
        NC_F(nc_put_var_int(ncid, v_status, &zero));
        NC_F(nc_put_var_int(ncid, v_total, &zero));
        NC_F(nc_put_var_int(ncid, v_dirty, &zero));
        if (mut != NcMutation::kStageRank0 && mut != NcMutation::kStageRank2) {
            std::vector<int> stage_ns(2, 3), stage_iter(2, 1), stage_conv(2, 1);
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
        const int one = 7;
        NC_F(nc_put_var_int(ncid, v_riter, &one));
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
    kNone,
    kRmnccRank1,          // rmncc with rank 1
    kRmnccRank3,          // rmncc with rank 3
    kFamilyExtentDiffers, // zmnsc with extent [ns, mnmax+1]
    kStageRank0,          // stage_ns scalar
    kStageRank2,          // stage_ns rank 2
    kIntAttrArray,        // "precision" attribute with 3 elements
    kStrAttrArray,        // "revision" attribute with 2 elements
    kWrongFamilyType,     // rmncc stored as H5T_NATIVE_INT
    kHugeNs,              // family extent ns = 2^40
};

static bool writeHdf5Fixture(const std::string& path, H5Mutation mut) {
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
        auto putStrAttr = [&](const char* name, const std::string& value)
            -> herr_t {
            hid_t s1 = H5Tcopy(H5T_C_S1);
            if (s1 < 0) return -1;
            herr_t r0 = H5Tset_size(s1, value.size() + 1);
            if (r0 < 0) { H5Tclose(s1); return -1; }
            hid_t sid = H5Screate(H5S_SCALAR);
            if (sid < 0) { H5Tclose(s1); return -1; }
            hid_t aid = H5Acreate2(fid, name, s1, sid, H5P_DEFAULT,
                                   H5P_DEFAULT);
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
            hid_t ds = H5Dcreate2(fid, name, dtype, sp, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return -1;
            const herr_t r = H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                      data);
            H5Dclose(ds);
            return r;
        };

        const hsize_t huge = (hsize_t)1 << 40;
        const hsize_t ns_ext = (mut == H5Mutation::kHugeNs) ? huge : 3;
        const hsize_t state_dims[2] = {ns_ext, 2};
        const char* fams[6] = {"rmncc", "zmnsc", "lmnsc",
                               "rmnss", "zmncs", "lmncs"};
        const double fambuf[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int c = 0; c < 6; ++c) {
            hid_t sp = -1;
            if (c == 0 && mut == H5Mutation::kRmnccRank1) {
                sp = H5Screate_simple(1, &ns_ext, nullptr);
            } else if (c == 0 && mut == H5Mutation::kRmnccRank3) {
                const hsize_t three[3] = {3, 2, 2};
                sp = H5Screate_simple(3, three, nullptr);
            } else if (c == 1 && mut == H5Mutation::kFamilyExtentDiffers) {
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
            const hid_t dtype =
                (c == 0 && mut == H5Mutation::kWrongFamilyType)
                    ? H5T_NATIVE_INT
                    : H5T_NATIVE_DOUBLE;
            hid_t ds = H5Dcreate2(fid, fams[c], dtype, sp, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(sp);
            if (ds < 0) return false;
            if (npts == 6 &&
                H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, fambuf) <
                    0) {
                H5Dclose(ds);
                return false;
            }
            H5Dclose(ds);
        }
        if (mut == H5Mutation::kIntAttrArray) {
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
        } else {
            H5_F(putIntAttr("precision", 0));
        }
        H5_F(putIntAttr("status", 0));
        H5_F(putIntAttr("total_iterations", 0));
        H5_F(putIntAttr("build_dirty", 0));
        if (mut == H5Mutation::kStrAttrArray) {
            // "revision" as a 2-element string array attribute.
            const hsize_t ext[1] = {2};
            hid_t sp = H5Screate_simple(1, ext, nullptr);
            if (sp < 0) return false;
            hid_t s1 = H5Tcopy(H5T_C_S1);
            if (s1 < 0) { H5Sclose(sp); return false; }
            if (H5Tset_size(s1, 3) < 0) { H5Tclose(s1); H5Sclose(sp); return false; }
            hid_t aid = H5Acreate2(fid, "revision", s1, sp, H5P_DEFAULT,
                                   H5P_DEFAULT);
            H5Sclose(sp);
            if (aid < 0) { H5Tclose(s1); return false; }
            const char* two[2] = {"ab", "cd"};
            const herr_t r = H5Awrite(aid, s1, two);
            H5Aclose(aid);
            H5Tclose(s1);
            H5_F(r);
        } else {
            H5_F(putStrAttr("revision", "r1"));
        }
        H5_F(putStrAttr("build_type", "Release"));
        H5_F(putStrAttr("precision_policy", "verify-double"));
        H5_F(putStrAttr("compile_flags", ""));
        H5_F(putStrAttr("source_path", "in.json"));
        H5_F(putStrAttr("source_hash", "h"));
        H5_F(putStrAttr("gpu_name", "g"));
        H5_F(putStrAttr("driver", "d"));
        H5_F(putStrAttr("runtime", "rt"));
        H5_F(putStrAttr("toolkit", "t"));
        std::vector<int> stage_ns(2, 3), stage_iter(2, 1), stage_conv(2, 1);
        std::vector<double> stage_res(2, 0.0);
        std::vector<int> rst_iter(1, 7);
        if (mut == H5Mutation::kStageRank0) {
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
        } else if (mut == H5Mutation::kStageRank2) {
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
        } else {
            H5_F(putArray("stage_ns", H5T_NATIVE_INT, 2, stage_ns.data()));
        }
        H5_F(putArray("stage_iterations", H5T_NATIVE_INT, 2, stage_iter.data()));
        H5_F(putArray("stage_converged", H5T_NATIVE_INT, 2, stage_conv.data()));
        H5_F(putArray("stage_fsqr", H5T_NATIVE_DOUBLE, 2, stage_res.data()));
        H5_F(putArray("stage_fsqz", H5T_NATIVE_DOUBLE, 2, stage_res.data()));
        H5_F(putArray("stage_fsql", H5T_NATIVE_DOUBLE, 2, stage_res.data()));
        {
            const int offs[2] = {0, 1};  // valid offsets for the control case
            H5_F(putArray("restart_stage_offset", H5T_NATIVE_INT, 2, offs));
        }
        H5_F(putArray("restart_iteration", H5T_NATIVE_INT, 1,
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
// One backend: the unmutated container must round-trip; every mutation must
// fail with a typed Result error (no crash, no partial report).
// ---------------------------------------------------------------------------
#ifdef CUMES_HAVE_NETCDF
static void testNetcdf(const TempDir& dir) {
    struct Case {
        NcMutation mut;
        const char* label;
    };
    const Case cases[] = {
        {NcMutation::kNone, "valid control"},
        {NcMutation::kFamilyRank1, "rmncc rank 1"},
        {NcMutation::kFamilyRank3, "rmncc rank 3"},
        {NcMutation::kFamilySwappedDims, "rmncc swapped dimensions"},
        {NcMutation::kScalarAsArray, "scalar outcome as array"},
        {NcMutation::kStageRank0, "stage_ns rank 0"},
        {NcMutation::kStageRank2, "stage_ns rank 2"},
        {NcMutation::kWrongFamilyType, "rmncc wrong datatype"},
        {NcMutation::kHugeNs, "ns beyond INT_MAX"},
    };
    for (const Case& c : cases) {
        const std::string path =
            dir.path() + "/" + c.label + ".nc";
        char msg[192];
        snprintf(msg, sizeof msg, "netcdf: fixture written (%s)", c.label);
        CHECK(writeNetcdfFixture(path, c.mut), msg);
        std::unique_ptr<Reader> reader =
            make_reader(OutputFormat::kNetCdf, OutputSchema::kV1);
        RunReport rep;
        const auto res = reader->read(path, &rep);
        remove(path.c_str());
        if (c.mut == NcMutation::kNone) {
            snprintf(msg, sizeof msg, "netcdf: %s", c.label);
            CHECK(res.has_value(), msg);
        } else {
            snprintf(msg, sizeof msg, "netcdf: %s rejected with typed failure",
                     c.label);
            CHECK(!res.has_value(), msg);
        }
    }
}
#endif

#ifdef CUMES_HAVE_HDF5
static void testHdf5(const TempDir& dir) {
    struct Case {
        H5Mutation mut;
        const char* label;
    };
    const Case cases[] = {
        {H5Mutation::kNone, "valid control"},
        {H5Mutation::kRmnccRank1, "rmncc rank 1"},
        {H5Mutation::kRmnccRank3, "rmncc rank 3"},
        {H5Mutation::kFamilyExtentDiffers, "zmnsc extent differs"},
        {H5Mutation::kStageRank0, "stage_ns rank 0"},
        {H5Mutation::kStageRank2, "stage_ns rank 2"},
        {H5Mutation::kIntAttrArray, "multi-element int attribute"},
        {H5Mutation::kStrAttrArray, "multi-element string attribute"},
        {H5Mutation::kWrongFamilyType, "rmncc wrong datatype"},
        {H5Mutation::kHugeNs, "ns beyond INT_MAX"},
    };
    for (const Case& c : cases) {
        const std::string path = dir.path() + "/" + c.label + ".h5";
        char msg[192];
        snprintf(msg, sizeof msg, "hdf5: fixture written (%s)", c.label);
        CHECK(writeHdf5Fixture(path, c.mut), msg);
        std::unique_ptr<Reader> reader =
            make_reader(OutputFormat::kHdf5, OutputSchema::kV1);
        RunReport rep;
        const auto res = reader->read(path, &rep);
        remove(path.c_str());
        if (c.mut == H5Mutation::kNone) {
            snprintf(msg, sizeof msg, "hdf5: %s", c.label);
            CHECK(res.has_value(), msg);
        } else {
            snprintf(msg, sizeof msg, "hdf5: %s rejected with typed failure",
                     c.label);
            CHECK(!res.has_value(), msg);
        }
    }
}
#endif

int main() {
    TempDir dir;
    CHECK(dir.ok(), "temp directory created");
    if (!dir.ok()) return 1;

#ifdef CUMES_HAVE_NETCDF
    testNetcdf(dir);
#else
    printf("SKIP netcdf malformed-shape cases (backend not compiled)\n");
#endif
#ifdef CUMES_HAVE_HDF5
    testHdf5(dir);
#else
    printf("SKIP hdf5 malformed-shape cases (backend not compiled)\n");
#endif

    if (failures == 0) {
        printf("test_io_malformed_shapes: all checks passed\n");
        return 0;
    }
    printf("test_io_malformed_shapes: %d check(s) FAILED\n", failures);
    return 1;
}
