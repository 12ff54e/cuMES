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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vmec_types.h"
#include "solver.cuh"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/io/writer.hpp"
#include "cumes_test_support.cuh"

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


// A tiny but valid state/params bundle for the writers. The state need not be
// physical — the writers copy whatever is there. ns=5, mnmax=2.
template <typename T>
struct TinyBundle {
    DeviceParams<T> p;
    cumes::ValidatedProblem vp;
    SolverResult<T> res;
    cumes::SpectralStorage<T> st;

    TinyBundle() {
        p.ns = 5; p.mnmax = 2; p.ntheta = 18; p.nzeta = 1; p.nfp = 1;
        p.nZnT = 18; p.mpol = 2; p.ntor = 0; p.ncurr = 0;
        p.delt = T(0.9); p.ftol = T(1e-14); p.max_iter = 10; p.tcon0 = T(1.0);
        p.lamscale = T(0.1);
        cumes::ProblemSpec spec;
        spec.mpol = p.mpol; spec.ntor = p.ntor; spec.nfp = 1;
        spec.angular.ntheta = p.ntheta; spec.angular.nzeta = p.nzeta;
        spec.mass.coefficients = {1.0};
        spec.toroidal_flux.coefficients = {1.0};
        spec.rbc = {{1, 0, 1.0}};
        spec.zbs = {{1, 0, 0.5}};
        spec.stages = {{static_cast<std::size_t>(p.ns),
                        static_cast<std::size_t>(p.max_iter),
                        static_cast<double>(p.ftol)}};
        vp = validateSpec(std::move(spec));
        res = SolverResult<T>{false, 3, T(1e-10), T(2e-10), T(3e-10), T(0.9)};
        // The state/velocity slabs are allocated + zeroed by SpectralStorage
        // (RAII); the writers only read the six state families.
        st.allocate(p.ns, p.mnmax);
    }
};

static bool fileExists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static void writeGarbage(const char* path, const char* bytes, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot seed %s\n", path); exit(1); }
    fwrite(bytes, 1, n, fp);
    fclose(fp);
}

// Drive the HOST writers (completion plan step 2.2): build the single host
// snapshot, then dispatch through the Writer interface — the failure matrix
// exercises the same publication protocol every backend uses.
template <typename T>
static bool writeViaHost(cumes::SpectralStorage<T>& st, const DeviceParams<T>& p,
                         const cumes::ValidatedProblem& vp,
                         const SolverResult<T>& res, const char* path,
                         cumes::OutputFormat fmt) {
    cumes::OutputSpec spec;
    spec.format = fmt;
    spec.schema = cumes::OutputSchema::kLegacyV0;
    spec.path = path;
    cumes::LegacyRunScalars s;
    s.mpol = p.mpol; s.ntor = p.ntor; s.nfp = p.nfp;
    s.ntheta = p.ntheta; s.nzeta = p.nzeta;
    s.ns = p.ns; s.mnmax = p.mnmax; s.nZnT = p.nZnT;
    s.ncurr = p.ncurr; s.max_iter = p.max_iter;
    s.delt = (double)p.delt; s.ftol = (double)p.ftol;
    s.lamscale = (double)p.lamscale;
    s.iterations = res.iterations;
    s.converged = res.converged;
    s.fsqr = (double)res.fsqr; s.fsqz = (double)res.fsqz;
    s.fsql = (double)res.fsql;
    auto snap = cumes::snapshot_from_device(st);
    cumes::RunReport rep;
    rep.input.source_path = "inputs/solovev.json";
    rep.build.scalar_type = sizeof(T) == sizeof(double) ? "double" : "float";
    auto w = cumes::make_writer(fmt, cumes::OutputSchema::kLegacyV0);
    if (!w) return false;
    return w->write_atomic(snap, rep, spec, vp, s).has_value();
}

template <typename T>
static void runAll() {
    printf("== %s precision ==\n", sizeof(T) == sizeof(double) ? "double" : "float");
    TinyBundle<T> b;

    // ---- open failure: path in a nonexistent directory ----
    const char* no_dir = "no_such_dir_cumes/state.bin";
    {
        // The binary writer must fail at fopen before reading the state.
        bool ok = writeViaHost<T>(b.st, b.p, b.vp, b.res, no_dir,
                                  cumes::OutputFormat::kBinary);
        CHECK(!ok, "open failure: binary returns false");
        CHECK(!fileExists(no_dir), "open failure: no partial file created");
    }
#ifdef CUMES_HAVE_NETCDF
    {
        bool ok = writeViaHost<T>(b.st, b.p, b.vp, b.res,
                                  "no_such_dir_cumes/state.nc",
                                  cumes::OutputFormat::kNetCdf);
        CHECK(!ok, "open failure: netcdf returns false");
        CHECK(!fileExists("no_such_dir_cumes/state.nc"),
              "open failure: netcdf no partial file");
    }
#endif
#ifdef CUMES_HAVE_HDF5
    {
        bool ok = writeViaHost<T>(b.st, b.p, b.vp, b.res,
                                  "no_such_dir_cumes/state.h5",
                                  cumes::OutputFormat::kHdf5);
        CHECK(!ok, "open failure: hdf5 returns false");
        CHECK(!fileExists("no_such_dir_cumes/state.h5"),
              "open failure: hdf5 no partial file");
    }
#endif

    // ---- atomic-rename failure: target is a directory with a valid suffix ----
    // The dispatcher must route to the binary writer, so the target carries a
    // .bin suffix; the target IS a directory, so the writer writes the temp
    // fine but rename(temp, target) fails. It must return false, remove the
    // temp, and leave the directory untouched. Note: the temp path is
    // <target>.tmp.<pid>, which sits NEXT to the directory and writes fine.
    {
        const char* dir = "test_output_target_dir.bin";
        // Remove any prior run's leftover, then make a directory as the target.
        remove(dir);
        if (mkdir(dir, 0755) != 0) { fprintf(stderr, "cannot mkdir %s\n", dir); exit(1); }
        bool ok = writeViaHost<T>(b.st, b.p, b.vp, b.res, dir,
                                  cumes::OutputFormat::kBinary);
        CHECK(!ok, "rename failure: target directory -> returns false");
        CHECK(fileExists(dir), "rename failure: target directory untouched");
        // No stray temp file may remain next to the target.
        std::string tmp = std::string(dir) + ".tmp.";
        bool stray = false;
        // scan CWD for <dir>.tmp.* leftovers
        DIR* d = opendir(".");
        if (d) {
            struct dirent* e;
            while ((e = readdir(d))) {
                if (std::string(e->d_name).compare(0, tmp.size(), tmp) == 0) stray = true;
            }
            closedir(d);
        }
        CHECK(!stray, "rename failure: no stray temp file left");
        rmdir(dir);
    }

    // ---- truncation: a pre-existing file is clobbered, old bytes gone ----
    {
        const char* path = "test_output_trunc.bin";
        writeGarbage(path, "GARBAGE-GARBAGE-GARBAGE-GARBAGE", 32);
        bool ok = writeViaHost<T>(b.st, b.p, b.vp, b.res, path,
                                  cumes::OutputFormat::kBinary);
        CHECK(ok, "truncation: binary overwrite succeeds");
        // Header is 2 ints = 8 bytes, then data; old 'GARBAGE' must be gone.
        FILE* fp = fopen(path, "rb");
        if (fp) {
            int ns = 0, mnmax = 0;
            bool hdr = fread(&ns, sizeof(int), 1, fp) == 1 &&
                       fread(&mnmax, sizeof(int), 1, fp) == 1;
            fclose(fp);
            CHECK(hdr && ns == b.p.ns && mnmax == b.p.mnmax,
                  "truncation: binary header is the new run's (ns/mnmax)");
        } else {
            CHECK(false, "truncation: binary file readable");
        }
        remove(path);
    }
#ifdef CUMES_HAVE_NETCDF
    {
        const char* path = "test_output_trunc.nc";
        writeGarbage(path, "GARBAGE-GARBAGE-GARBAGE-GARBAGE", 32);
        bool ok = writeViaHost<T>(b.st, b.p, b.vp, b.res, path,
                                  cumes::OutputFormat::kNetCdf);
        CHECK(ok, "truncation: netcdf overwrite succeeds");
        // NC_CLOBBER must have replaced the garbage; the file now starts with
        // the netCDF magic "CDF".
        FILE* fp = fopen(path, "rb");
        if (fp) {
            char magic[4] = {0};
            size_t got = fread(magic, 1, 3, fp);
            fclose(fp);
            CHECK(got == 3 && memcmp(magic, "CDF", 3) == 0,
                  "truncation: netcdf file is a fresh CDF (not stale garbage)");
        } else {
            CHECK(false, "truncation: netcdf file readable");
        }
        remove(path);
    }
#endif
#ifdef CUMES_HAVE_HDF5
    {
        const char* path = "test_output_trunc.h5";
        writeGarbage(path, "GARBAGE-GARBAGE-GARBAGE-GARBAGE", 32);
        bool ok = writeViaHost<T>(b.st, b.p, b.vp, b.res, path,
                                  cumes::OutputFormat::kHdf5);
        CHECK(ok, "truncation: hdf5 overwrite succeeds");
        // HDF5 signature is "\211HDF\r\n\032\n".
        FILE* fp = fopen(path, "rb");
        if (fp) {
            const unsigned char sig[8] = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
            unsigned char buf[8] = {0};
            size_t got = fread(buf, 1, 8, fp);
            fclose(fp);
            CHECK(got == 8 && memcmp(buf, sig, 8) == 0,
                  "truncation: hdf5 file is a fresh HDF5 (not stale garbage)");
        } else {
            CHECK(false, "truncation: hdf5 file readable");
        }
        remove(path);
    }
#endif
}

int main() {
    printf("=== Output failure-injection matrix ===\n");
    runAll<double>();
    runAll<float>();
    if (failures == 0) {
        printf("test_output_failure: ALL PASS\n");
        return 0;
    }
    printf("test_output_failure: %d FAILURES\n", failures);
    return 1;
}
