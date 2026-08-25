// test_output_failure.cu — writer failure-injection matrix.
// Exercises every output backend's failure contract deterministically:
//
//   open failure   -> a path in a nonexistent directory (fopen/nc_create/
//                     H5Fcreate fail before any state is read)
//   rename failure -> the target is a non-empty directory: the writer writes
//                     a same-directory temp fine, the atomic rename() fails,
//                     it returns false, removes the temp, and leaves the
//                     target untouched
//   truncation     -> a pre-existing file is atomically replaced (write temp
//                     + rename), so a stale reader can never mistake old
//                     bytes for the new run's output
//   close failure  -> exercised via the close-safe cleanup path (see the
//                     writers); a failing close must not be re-entered
//
// The contract under test (output.cuh): every writer returns true only when
// the file was fully written AND closed AND atomically published, false on
// any failure (after removing partial temp files). main folds this into the
// CLI exit code, so a run can never report success without a durable result.
//
// All backends are exercised only when compiled in (CUMES_HAVE_NETCDF /
// CUMES_HAVE_HDF5), so the same source works in the no/one/both backend
// build matrix.
#include "cumes/io/output_spec.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/io/writer.hpp"
#include "cumes/io/writer_helpers.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "solver.cuh"
#include "vmec_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace cumes::test;

// A tiny but valid state/params bundle for the writers. The state need not be
// physical — the writers copy whatever is there. ns=5, mnmax=2.
template <typename T>
struct TinyBundle {
    using val_type = T;

    DeviceParams<T> p;
    cumes::ValidatedProblem vp;
    SolverResult<T> res;
    cumes::SpectralStorage<T> st;

    TinyBundle() {
        p.ns = 5;
        p.mnmax = 2;
        p.ntheta = 18;
        p.nzeta = 1;
        p.nfp = 1;
        p.nZnT = 18;
        p.mpol = 2;
        p.ntor = 0;
        p.ncurr = 0;
        p.delt = T(0.9);
        p.ftol = T(1e-14);
        p.max_iter = 10;
        p.tcon0 = T(1.0);
        p.lamscale = T(0.1);
        cumes::ProblemSpec spec;
        spec.mpol = p.mpol;
        spec.ntor = p.ntor;
        spec.nfp = 1;
        spec.angular.ntheta = p.ntheta;
        spec.angular.nzeta = p.nzeta;
        spec.mass.coefficients = {1.0};
        spec.toroidal_flux.coefficients.assign(1, 1.0);
        spec.rbc = {{1, 0, 1.0}};
        spec.zbs = {{1, 0, 0.5}};
        spec.stages = {{static_cast<std::size_t>(p.ns),
                        static_cast<std::size_t>(p.max_iter),
                        static_cast<double>(p.ftol)}};
        vp = validate_spec(std::move(spec));
        res =
            SolverResult<T>{false, 3, T(1e-10), T(2e-10), T(3e-10), T(0.9), {}};
        // The state/velocity slabs are allocated + zeroed by SpectralStorage
        // (RAII); the writers only read the six state families.
        st.allocate(p.ns, p.mnmax);
    }
};

static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static void write_garbage(const char* path, const char* bytes, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        std::cerr << format("cannot seed {}\n", path);
        exit(1);
    }
    fwrite(bytes, 1, n, fp);
    fclose(fp);
}

// Drive the HOST writers (completion plan step 2.2): build the single host
// snapshot, then dispatch through the Writer interface — the failure matrix
// exercises the same publication protocol every backend uses.
template <typename T>
static bool write_via_host(cumes::SpectralStorage<T>& st,
                           const DeviceParams<T>& p,
                           const cumes::ValidatedProblem& vp,
                           const SolverResult<T>& res,
                           const char* path,
                           cumes::OutputFormat fmt) {
    cumes::OutputSpec spec;
    spec.format = fmt;
    spec.path = path;
    auto snap = cumes::snapshot_from_device(st);
    cumes::RunReport rep;
    rep.input.source_path = "inputs/solovev.json";
    rep.build.scalar_type = sizeof(T) == sizeof(double) ? "double" : "float";
    // One stage record, like a real run: the classic-format NetCDF writer
    // cannot define two zero-size dimensions (nstages=0 and nrestarts=0 both
    // map to the single unlimited dimension), and a solver run always records
    // at least one stage.
    cumes::StageReport stage;
    stage.ns = p.ns;
    stage.effective_iterations = res.iterations;
    stage.converged = res.converged;
    stage.final_residual = {(double)res.fsqr, (double)res.fsqz,
                            (double)res.fsql};
    rep.stages.push_back(stage);
    auto w = cumes::make_writer(fmt);
    if (!w) return false;
    return w->write_atomic(snap, rep, spec, vp).has_value();
}

template <typename T>
static void run_all() {
    std::cout << format("== {} precision ==\n",
                        sizeof(T) == sizeof(double) ? "double" : "float");
    TinyBundle<T> b;

    // ---- open failure: path in a nonexistent directory ----
    const char* no_dir = "no_such_dir_cumes/state.bin";
    {
        // The binary writer must fail at fopen before reading the state.
        bool ok = write_via_host<T>(b.st, b.p, b.vp, b.res, no_dir,
                                    cumes::OutputFormat::BINARY);
        check(!ok, "open failure: binary returns false");
        check(!file_exists(no_dir), "open failure: no partial file created");
    }
#ifdef CUMES_HAVE_NETCDF
    {
        bool ok = write_via_host<T>(b.st, b.p, b.vp, b.res,
                                    "no_such_dir_cumes/state.nc",
                                    cumes::OutputFormat::NETCDF);
        check(!ok, "open failure: netcdf returns false");
        check(!file_exists("no_such_dir_cumes/state.nc"),
              "open failure: netcdf no partial file");
    }
#endif
#ifdef CUMES_HAVE_HDF5
    {
        bool ok = write_via_host<T>(b.st, b.p, b.vp, b.res,
                                    "no_such_dir_cumes/state.h5",
                                    cumes::OutputFormat::HDF5);
        check(!ok, "open failure: hdf5 returns false");
        check(!file_exists("no_such_dir_cumes/state.h5"),
              "open failure: hdf5 no partial file");
    }
#endif

    // ---- atomic-rename failure: target is a directory with a valid suffix
    // ---- The dispatcher must route to the binary writer, so the target
    // carries a .bin suffix; the target IS a directory, so the writer writes
    // the temp fine but rename(temp, target) fails. It must return false,
    // remove the temp, and leave the directory untouched. Note: the temp path
    // is <target>.tmp.<pid>, which sits NEXT to the directory and writes fine.
    {
        const char* dir = "test_output_target_dir.bin";
        // Remove any prior run's leftover, then make a directory as the target.
        remove(dir);
        if (mkdir(dir, 0755) != 0) {
            std::cerr << format("cannot mkdir {}\n", dir);
            exit(1);
        }
        bool ok = write_via_host<T>(b.st, b.p, b.vp, b.res, dir,
                                    cumes::OutputFormat::BINARY);
        check(!ok, "rename failure: target directory -> returns false");
        check(file_exists(dir), "rename failure: target directory untouched");
        // No stray temp file may remain next to the target.
        std::string tmp = std::string(dir) + ".tmp.";
        bool stray = false;
        // scan CWD for <dir>.tmp.* leftovers
        DIR* d = opendir(".");
        if (d) {
            struct dirent* e;
            while ((e = readdir(d))) {
                if (std::string(e->d_name).compare(0, tmp.size(), tmp) == 0)
                    stray = true;
            }
            closedir(d);
        }
        check(!stray, "rename failure: no stray temp file left");
        rmdir(dir);
    }

    // ---- truncation: a pre-existing file is clobbered, old bytes gone ----
    {
        const char* path = "test_output_trunc.bin";
        write_garbage(path, "GARBAGE-GARBAGE-GARBAGE-GARBAGE", 32);
        bool ok = write_via_host<T>(b.st, b.p, b.vp, b.res, path,
                                    cumes::OutputFormat::BINARY);
        check(ok, "truncation: binary overwrite succeeds");
        // v1 header: magic (8) + version (4) + ns (4) + mnmax (4); the old
        // 'GARBAGE' bytes must be gone.
        FILE* fp = fopen(path, "rb");
        if (fp) {
            char magic[8] = {0};
            int version = 0, ns = 0, mnmax = 0;
            bool hdr = fread(magic, 1, 8, fp) == 8 &&
                       fread(&version, sizeof(int), 1, fp) == 1 &&
                       fread(&ns, sizeof(int), 1, fp) == 1 &&
                       fread(&mnmax, sizeof(int), 1, fp) == 1;
            fclose(fp);
            check(
                hdr && memcmp(magic, "CUMES001", 8) == 0 && version == 4 &&
                    ns == b.p.ns && mnmax == b.p.mnmax,
                "truncation: binary header is the new run's (magic/ns/mnmax)");
        } else {
            check(false, "truncation: binary file readable");
        }
        remove(path);
    }
#ifdef CUMES_HAVE_NETCDF
    {
        const char* path = "test_output_trunc.nc";
        write_garbage(path, "GARBAGE-GARBAGE-GARBAGE-GARBAGE", 32);
        bool ok = write_via_host<T>(b.st, b.p, b.vp, b.res, path,
                                    cumes::OutputFormat::NETCDF);
        check(ok, "truncation: netcdf overwrite succeeds");
        // NC_CLOBBER must have replaced the garbage; the file now starts with
        // the netCDF magic "CDF".
        FILE* fp = fopen(path, "rb");
        if (fp) {
            char magic[4] = {0};
            size_t got = fread(magic, 1, 3, fp);
            fclose(fp);
            check(got == 3 && memcmp(magic, "CDF", 3) == 0,
                  "truncation: netcdf file is a fresh CDF (not stale garbage)");
        } else {
            check(false, "truncation: netcdf file readable");
        }
        remove(path);
    }
#endif
#ifdef CUMES_HAVE_HDF5
    {
        const char* path = "test_output_trunc.h5";
        write_garbage(path, "GARBAGE-GARBAGE-GARBAGE-GARBAGE", 32);
        bool ok = write_via_host<T>(b.st, b.p, b.vp, b.res, path,
                                    cumes::OutputFormat::HDF5);
        check(ok, "truncation: hdf5 overwrite succeeds");
        // HDF5 signature is "\211HDF\r\n\032\n".
        FILE* fp = fopen(path, "rb");
        if (fp) {
            const unsigned char sig[8] = {0x89, 'H',  'D',  'F',
                                          '\r', '\n', 0x1a, '\n'};
            unsigned char buf[8] = {0};
            size_t got = fread(buf, 1, 8, fp);
            fclose(fp);
            check(got == 8 && memcmp(buf, sig, 8) == 0,
                  "truncation: hdf5 file is a fresh HDF5 (not stale garbage)");
        } else {
            check(false, "truncation: hdf5 file readable");
        }
        remove(path);
    }
#endif
}

// ---------------------------------------------------------------------------
// Library-managed publication boundaries (completion-plan follow-up §3): the
// checked reopen/fsync/close/rename/directory-fsync chain publish_library_file
// implements for the NetCDF/HDF5 writers. Injected failures():
//
//   reopen  -> a missing temp fails before any rename;
//   fsync   -> /proc files reject fsync with EINVAL (a real, deterministic
//              fault on Linux; no fault-injection hook is needed);
//   rename  -> the target is a directory (covered end to end above too);
//   dir-fsync -> fsync_directory_of on /proc propagates the EINVAL instead of
//              ignoring it (the old helper ignored fsync AND close errors);
//
// close failures() cannot be injected portably without interposition; the
// checked close-after-fsync is the same three-line pattern as the fsync
// check and is exercised by the same code path here.
static void run_publication_boundaries() {
    // Reopen boundary: the temp must exist and be readable.
    const char* dest1 = "test_output_pub_dest1.bin";
    remove(dest1);
    write_garbage(dest1, "OLD-DESTINATION", 15);
    {
        const std::string err = cumes::io_detail::publish_library_file(
            "test_output_pub_missing.tmp.x", dest1);
        check(!err.empty() && err.find("reopen") != std::string::npos,
              "library publish: missing temp fails at the reopen boundary");
        FILE* fp = fopen(dest1, "rb");
        char buf[16] = {0};
        const size_t got = fp ? fread(buf, 1, 15, fp) : 0;
        if (fp) fclose(fp);
        check(got == 15 && std::memcmp(buf, "OLD-DESTINATION", 15) == 0,
              "library publish: destination preserved on reopen failure");
    }
    remove(dest1);

    // fsync boundary: open succeeds, fsync fails (EINVAL on procfs).
    {
        const std::string err = cumes::io_detail::publish_library_file(
            "/proc/self/stat", "test_output_pub_dest2.bin");
        check(!err.empty() && err.find("fsync") != std::string::npos,
              "library publish: fsync failure propagates a typed reason");
        check(!file_exists("test_output_pub_dest2.bin"),
              "library publish: no destination after fsync failure");
    }

    // Directory-fsync boundary: checked and propagated (the old helper
    // ignored both the fsync and the close result).
    {
        const std::string err =
            cumes::io_detail::fsync_directory_of("/proc/self");
        check(!err.empty() && err.find("fsync") != std::string::npos,
              "directory fsync failure propagates a typed reason");
    }

    // Write/fflush boundary of the FILE* protocol: buffered writes to
    // /dev/full succeed until flush, where fflush fails — publish_atomic must
    // report it and remove nothing but its temp.
    {
        FILE* fp = fopen("/dev/full", "wb");
        check(fp != nullptr, "write fault: /dev/full opens");
        if (fp) {
            (void)fwrite("x", 1, 1, fp);  // buffered; fails at flush time
            const std::string err = cumes::io_detail::publish_atomic(
                fp, "/dev/full.tmp.x", "test_output_pub_dest3.bin");
            check(!err.empty() && err.find("fflush") != std::string::npos,
                  "write fault: fflush failure propagates a typed reason");
            check(!file_exists("test_output_pub_dest3.bin"),
                  "write fault: no destination after flush failure");
        }
    }
}

int main() {
    std::cout << "=== Output failure-injection matrix ===\n";
    run_all<double>();
    run_all<float>();
    run_publication_boundaries();
    return summary();
}
