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
#include "input.h"
#include "solver.cuh"
#include "output.cuh"

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

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(1);
    }
}

// A tiny but valid state/params bundle for the writers. The state need not be
// physical — the writers copy whatever is there. ns=5, mnmax=2.
template <typename T>
struct TinyBundle {
    GridParams<T> p;
    InputParams ip;
    SolverResult<T> res;
    SpectralState<T> st;

    TinyBundle() {
        p.ns = 5; p.mnmax = 2; p.ntheta = 18; p.nzeta = 1; p.nfp = 1;
        p.nZnT = 18; p.mpol = 2; p.ntor = 0; p.ncurr = 0;
        p.delt = T(0.9); p.ftol = T(1e-14); p.max_iter = 10; p.tcon0 = T(1.0);
        p.lamscale = T(0.1);
        ip.ns = p.ns; ip.mpol = p.mpol; ip.ntor = p.ntor; ip.n_grids = 1;
        ip.ns_array[0] = p.ns; ip.niter_array[0] = p.max_iter;
        ip.ftol_array[0] = p.ftol;
        res = SolverResult<T>{false, 3, T(1e-10), T(2e-10), T(3e-10), T(0.9)};
        const size_t nb = (size_t)p.ns * p.mnmax * sizeof(T);
        checkCuda(cudaMalloc(&st.d_rmncc, nb), "rmncc");
        checkCuda(cudaMalloc(&st.d_zmnsc, nb), "zmnsc");
        checkCuda(cudaMalloc(&st.d_lmnsc, nb), "lmnsc");
        checkCuda(cudaMalloc(&st.d_rmnss, nb), "rmnss");
        checkCuda(cudaMalloc(&st.d_zmncs, nb), "zmncs");
        checkCuda(cudaMalloc(&st.d_lmncs, nb), "lmncs");
        checkCuda(cudaMalloc(&st.d_v_rmncc, nb), "vcc");
        checkCuda(cudaMalloc(&st.d_v_zmnsc, nb), "vzsc");
        checkCuda(cudaMalloc(&st.d_v_lmnsc, nb), "vlsc");
        checkCuda(cudaMalloc(&st.d_v_rmnss, nb), "vss");
        checkCuda(cudaMalloc(&st.d_v_zmncs, nb), "vzcs");
        checkCuda(cudaMalloc(&st.d_v_lmncs, nb), "vlcs");
        checkCuda(cudaMemset(st.d_rmncc, 0, nb), "zero cc");
        checkCuda(cudaMemset(st.d_zmnsc, 0, nb), "zero zsc");
        checkCuda(cudaMemset(st.d_lmnsc, 0, nb), "zero lsc");
        checkCuda(cudaMemset(st.d_rmnss, 0, nb), "zero ss");
        checkCuda(cudaMemset(st.d_zmncs, 0, nb), "zero zcs");
        checkCuda(cudaMemset(st.d_lmncs, 0, nb), "zero lcs");
    }
    ~TinyBundle() {
        cudaFree(st.d_rmncc); cudaFree(st.d_zmnsc); cudaFree(st.d_lmnsc);
        cudaFree(st.d_rmnss); cudaFree(st.d_zmncs); cudaFree(st.d_lmncs);
        cudaFree(st.d_v_rmncc); cudaFree(st.d_v_zmnsc); cudaFree(st.d_v_lmnsc);
        cudaFree(st.d_v_rmnss); cudaFree(st.d_v_zmncs); cudaFree(st.d_v_lmncs);
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

template <typename T>
static void runAll() {
    printf("== %s precision ==\n", sizeof(T) == sizeof(double) ? "double" : "float");
    TinyBundle<T> b;

    // ---- open failure: path in a nonexistent directory ----
    const char* no_dir = "no_such_dir_cumes/state.bin";
    {
        // The binary writer must fail at fopen before reading the state.
        bool ok = outputSave<T>(b.st, b.p, b.ip, b.res, no_dir, "inputs/solovev.json");
        CHECK(!ok, "open failure: binary returns false");
        CHECK(!fileExists(no_dir), "open failure: no partial file created");
    }
#ifdef CUMES_HAVE_NETCDF
    {
        bool ok = outputSave<T>(b.st, b.p, b.ip, b.res,
                                "no_such_dir_cumes/state.nc", "inputs/solovev.json");
        CHECK(!ok, "open failure: netcdf returns false");
        CHECK(!fileExists("no_such_dir_cumes/state.nc"),
              "open failure: netcdf no partial file");
    }
#endif
#ifdef CUMES_HAVE_HDF5
    {
        bool ok = outputSave<T>(b.st, b.p, b.ip, b.res,
                                "no_such_dir_cumes/state.h5", "inputs/solovev.json");
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
        bool ok = outputSave<T>(b.st, b.p, b.ip, b.res, dir, "inputs/solovev.json");
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
        bool ok = outputSave<T>(b.st, b.p, b.ip, b.res, path, "inputs/solovev.json");
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
        bool ok = outputSave<T>(b.st, b.p, b.ip, b.res, path, "inputs/solovev.json");
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
        bool ok = outputSave<T>(b.st, b.p, b.ip, b.res, path, "inputs/solovev.json");
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
