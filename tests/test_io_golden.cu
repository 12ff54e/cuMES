// test_io_golden.cu — the golden byte-identity gate for the versioned writers.
//
// Phase 5 handover §6.1: before main.cu swaps outputSaveBinary for the
// versioned legacy-v0 writer, the two must be proven byte-identical, because
// scripts/compare_bitwise.py compares the on-disk cumes_state.bin. This test is
// that proof: the same device state is written through (a) outputSaveBinary
// (the reference, src/output.cpp) and (b) make_writer(Binary, LegacyV0) fed by
// snapshot_from_device (the bridge), and the two files must be byte-for-byte
// equal. It also closes the bridge->reader loop: the v1 writer and the
// versioned checkpoint both round-trip the bridged snapshot.
//
// Runs both precisions (double is the verification config; float proves the
// T->double conversion is identical for the single-precision build).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "cumes/io/checkpoint.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/reader.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/io/writer.hpp"
#ifdef CUMES_HAVE_NETCDF
#include <netcdf.h>
#endif
#ifdef CUMES_HAVE_HDF5
#include <hdf5.h>
#endif
#include "cumes/state/spectral_storage.hpp"
#include "output.cuh"
#include "vmec_types.h"
#include "cumes_test_support.cuh"

using cumes::EquilibriumSnapshot;
using cumes::OutputFormat;
using cumes::OutputSchema;
using cumes::OutputSpec;
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

static std::string scratch(const char* name) {
    return std::string("test_io_golden_") + name + "_" +
           std::to_string((long)getpid());
}

// Deterministic pattern: family c, index i -> value = c*1000 + i + 0.25
// (0.25 is exactly representable in both float and double, so the T->double
// conversion is lossless in both precisions).
template <typename T>
static cumes::SpectralStorage<T> makeStorage(int ns, int mnmax) {
    cumes::SpectralStorage<T> storage(ns, mnmax);
    const std::size_t one = static_cast<std::size_t>(ns) * mnmax;
    const std::size_t count = 6 * one;
    std::vector<T> host(count);
    for (int c = 0; c < 6; ++c) {
        for (std::size_t i = 0; i < one; ++i) {
            host[c * one + i] =
                static_cast<T>(c * 1000.0 + static_cast<double>(i) + 0.25);
        }
    }
    checkCuda(cudaMemcpy(storage.state_slab(), host.data(), count * sizeof(T),
                         cudaMemcpyHostToDevice),
              "upload");
    return storage;
}

static bool readFileBytes(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream os;
    os << in.rdbuf();
    out = os.str();
    return true;
}

static bool snapshotsEqual(const EquilibriumSnapshot& a,
                           const EquilibriumSnapshot& b) {
    if (a.ns != b.ns || a.mnmax != b.mnmax) return false;
    for (int c = 0; c < 6; ++c) {
        if (a.families[c] != b.families[c]) return false;
    }
    return true;
}

// A rich report exercising every field the v1 containers must round-trip
// (completion plan step 2.3).
static RunReport makeIoReport() {
    RunReport r;
    r.status = cumes::RunStatus::kConverged;
    r.total_effective_iterations = 21;
    cumes::StageReport s1;
    s1.ns = 5; s1.effective_iterations = 9; s1.converged = true;
    s1.final_residual = {1e-10, 2e-11, 3e-12};
    s1.restarts = {cumes::RestartEvent{2}, cumes::RestartEvent{5}};
    cumes::StageReport s2;
    s2.ns = 8; s2.effective_iterations = 12; s2.converged = true;
    s2.final_residual = {4e-13, 5e-14, 6e-15};
    s2.restarts = {cumes::RestartEvent{3}};
    r.stages = {s1, s2};
    r.build.revision = "abc123";
    r.build.dirty = true;
    r.build.build_type = "Release";
    r.build.scalar_type = "double";
    r.build.precision_policy = "verify-double";
    r.build.compile_flags = "";
    r.input.source_path = "inputs/w7x.json";
    r.input.source_hash = "0123456789abcdef";
    r.runtime.gpu_name = "NVIDIA TITAN Xp";
    r.runtime.driver = "580.173.02";
    r.runtime.runtime = "12010";
    r.runtime.toolkit = "12.1";
    return r;
}

static bool reportsEqual(const RunReport& a, const RunReport& b) {
    if (a.status != b.status ||
        a.total_effective_iterations != b.total_effective_iterations) {
        return false;
    }
    if (a.build.revision != b.build.revision ||
        a.build.dirty != b.build.dirty ||
        a.build.build_type != b.build.build_type ||
        a.build.scalar_type != b.build.scalar_type ||
        a.build.precision_policy != b.build.precision_policy ||
        a.build.compile_flags != b.build.compile_flags ||
        a.input.source_path != b.input.source_path ||
        a.input.source_hash != b.input.source_hash ||
        a.runtime.gpu_name != b.runtime.gpu_name ||
        a.runtime.driver != b.runtime.driver ||
        a.runtime.runtime != b.runtime.runtime ||
        a.runtime.toolkit != b.runtime.toolkit) {
        return false;
    }
    if (a.stages.size() != b.stages.size()) return false;
    for (size_t g = 0; g < a.stages.size(); ++g) {
        const auto& x = a.stages[g];
        const auto& y = b.stages[g];
        if (x.ns != y.ns || x.effective_iterations != y.effective_iterations ||
            x.converged != y.converged ||
            x.final_residual.fsqr != y.final_residual.fsqr ||
            x.final_residual.fsqz != y.final_residual.fsqz ||
            x.final_residual.fsql != y.final_residual.fsql ||
            x.restarts.size() != y.restarts.size()) {
            return false;
        }
        for (size_t k = 0; k < x.restarts.size(); ++k) {
            if (x.restarts[k].iteration != y.restarts[k].iteration) return false;
        }
    }
    return true;
}

// v1 round trip shared by both backends: state + complete RunReport.
template <typename T>
static void checkV1RoundTrip(const EquilibriumSnapshot& snap,
                             const cumes::ValidatedProblem& vp,
                             const cumes::LegacyRunScalars& scalars,
                             OutputFormat fmt, const char* tag) {
    const std::string path = scratch(tag);
    OutputSpec spec;
    spec.format = fmt;
    spec.schema = OutputSchema::kV1;
    spec.path = path;
    auto w = cumes::make_writer(spec.format, spec.schema);
    auto r = cumes::make_reader(spec.format, spec.schema);
    CHECK(w != nullptr && r != nullptr, "v1 factories");
    if (w && r) {
        RunReport report = makeIoReport();
        report.build.scalar_type =
            sizeof(T) == sizeof(double) ? "double" : "float";
        CHECK(w->write_atomic(snap, report, spec, vp, scalars).has_value(),
              "v1 write succeeds");
        RunReport back;
        auto snap_back = r->read(path, &back);
        CHECK(snap_back.has_value() && snapshotsEqual(snap, snap_back.value()),
              "v1 state round trip");
        CHECK(reportsEqual(report, back),
              "v1 complete RunReport + restart metadata round trip");
    }
    remove(path.c_str());
}

#ifdef CUMES_HAVE_NETCDF
template <typename T>
static void testNetcdfLayouts(const EquilibriumSnapshot& snap,
                              const cumes::ValidatedProblem& vp,
                              const cumes::LegacyRunScalars& scalars, int ns,
                              int mnmax) {
    // v0 documented layout.
    {
        const std::string path = scratch("v0nc");
        OutputSpec spec;
        spec.format = OutputFormat::kNetCdf;
        spec.schema = OutputSchema::kLegacyV0;
        spec.path = path;
        auto w = cumes::make_writer(spec.format, spec.schema);
        CHECK(w != nullptr, "nc v0: factory");
        if (w) {
            RunReport report;
            report.input.source_path = "inputs/solovev.json";
            report.build.scalar_type =
                sizeof(T) == sizeof(double) ? "double" : "float";
            CHECK(w->write_atomic(snap, report, spec, vp, scalars).has_value(),
                  "nc v0: write");
            int ncid = -1;
            CHECK(nc_open(path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR,
                  "nc v0: readable");
            if (ncid >= 0) {
                auto dimlen = [&](const char* name) -> size_t {
                    int id = -1;
                    size_t n = 0;
                    if (nc_inq_dimid(ncid, name, &id) != NC_NOERR) return 0;
                    nc_inq_dimlen(ncid, id, &n);
                    return n;
                };
                CHECK(dimlen("ngrids") == 8 && dimlen("ncoeff") == 16 &&
                          dimlen("naxis") == 32 && dimlen("nbm") == 16 &&
                          dimlen("nbn") == 16,
                      "nc v0: fixed capacities");
                CHECK(dimlen("ns") == (size_t)ns &&
                          dimlen("mnmax") == (size_t)mnmax,
                      "nc v0: active state dims");
                int vid = -1;
                int iters = 0;
                CHECK(nc_inq_varid(ncid, "iterations", &vid) == NC_NOERR &&
                          nc_get_var_int(ncid, vid, &iters) == NC_NOERR &&
                          iters == scalars.iterations,
                      "nc v0: scalar value");
                double delt = 0.0;
                CHECK(nc_inq_varid(ncid, "delt", &vid) == NC_NOERR &&
                          nc_get_var_double(ncid, vid, &delt) == NC_NOERR &&
                          delt == scalars.delt,
                      "nc v0: double scalar value");
                CHECK(nc_inq_varid(ncid, "rmncc", &vid) == NC_NOERR,
                      "nc v0: rmncc variable");
                if (vid >= 0) {
                    size_t start[2] = {0, 0};
                    size_t count[2] = {(size_t)ns, 1};
                    std::vector<double> col(ns);
                    nc_get_vara_double(ncid, vid, start, count, col.data());
                    bool same = true;
                    for (int j = 0; j < ns; ++j) {
                        same = same && col[j] == snap.families[0][j];
                    }
                    CHECK(same, "nc v0: mode-0 column matches the snapshot");
                }
                nc_close(ncid);
            }
        }
        remove(path.c_str());
    }
    // v1 round trip.
    checkV1RoundTrip<T>(snap, vp, scalars, OutputFormat::kNetCdf, "v1nc");
}
#endif  // CUMES_HAVE_NETCDF

#ifdef CUMES_HAVE_HDF5
template <typename T>
static void testHdf5Layouts(const EquilibriumSnapshot& snap,
                            const cumes::ValidatedProblem& vp,
                            const cumes::LegacyRunScalars& scalars, int ns,
                            int mnmax) {
    // v0 documented layout (the byte-level contract is the layout: libhdf5
    // embeds a per-second timestamp, so structure + values are asserted).
    {
        const std::string path = scratch("v0h5");
        OutputSpec spec;
        spec.format = OutputFormat::kHdf5;
        spec.schema = OutputSchema::kLegacyV0;
        spec.path = path;
        auto w = cumes::make_writer(spec.format, spec.schema);
        CHECK(w != nullptr, "h5 v0: factory");
        if (w) {
            RunReport report;
            report.input.source_path = "inputs/solovev.json";
            report.build.scalar_type =
                sizeof(T) == sizeof(double) ? "double" : "float";
            CHECK(w->write_atomic(snap, report, spec, vp, scalars).has_value(),
                  "h5 v0: write");
            hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            CHECK(fid >= 0, "h5 v0: readable");
            if (fid >= 0) {
                auto attrInt = [&](const char* name, int& out) -> bool {
                    hid_t aid = H5Aopen(fid, name, H5P_DEFAULT);
                    if (aid < 0) return false;
                    const herr_t r = H5Aread(aid, H5T_NATIVE_INT, &out);
                    H5Aclose(aid);
                    return r >= 0;
                };
                auto attrDbl = [&](const char* name, double& out) -> bool {
                    hid_t aid = H5Aopen(fid, name, H5P_DEFAULT);
                    if (aid < 0) return false;
                    const herr_t r = H5Aread(aid, H5T_NATIVE_DOUBLE, &out);
                    H5Aclose(aid);
                    return r >= 0;
                };
                int iters = 0;
                double delt = 0.0;
                CHECK(attrInt("iterations", iters) &&
                          iters == scalars.iterations,
                      "h5 v0: scalar value");
                CHECK(attrDbl("delt", delt) && delt == scalars.delt,
                      "h5 v0: double scalar value");
                // fixed-capacity provenance arrays
                auto arrDims = [&](const char* name, hsize_t* dims) -> bool {
                    hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
                    if (ds < 0) return false;
                    hid_t sp = H5Dget_space(ds);
                    const bool ok =
                        H5Sget_simple_extent_dims(sp, dims, nullptr) == 1;
                    H5Sclose(sp);
                    H5Dclose(ds);
                    return ok;
                };
                hsize_t dims[1] = {0};
                CHECK(arrDims("ns_array", dims) && dims[0] == 8,
                      "h5 v0: ns_array fixed capacity");
                CHECK(arrDims("am", dims) && dims[0] == 16,
                      "h5 v0: am fixed capacity");
                // state dataset dims + mode-0 column
                hid_t ds = H5Dopen2(fid, "rmncc", H5P_DEFAULT);
                if (ds >= 0) {
                    hid_t sp = H5Dget_space(ds);
                    hsize_t sdims[2] = {0, 0};
                    H5Sget_simple_extent_dims(sp, sdims, nullptr);
                    H5Sclose(sp);
                    CHECK(sdims[0] == (hsize_t)ns &&
                              sdims[1] == (hsize_t)mnmax,
                          "h5 v0: state dataset dims");
                    std::vector<double> col(ns);
                    hsize_t start[2] = {0, 0};
                    hsize_t count[2] = {(hsize_t)ns, 1};
                    hid_t fs = H5Dget_space(ds);
                    H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr,
                                        count, nullptr);
                    hid_t ms = H5Screate_simple(1, count, nullptr);
                    const herr_t rr = H5Dread(ds, H5T_NATIVE_DOUBLE, ms, fs,
                                              H5P_DEFAULT, col.data());
                    H5Sclose(ms);
                    H5Sclose(fs);
                    H5Dclose(ds);
                    bool same = (rr >= 0);
                    for (int j = 0; j < ns; ++j) {
                        same = same && col[j] == snap.families[0][j];
                    }
                    CHECK(same, "h5 v0: mode-0 column matches the snapshot");
                } else {
                    CHECK(false, "h5 v0: rmncc dataset");
                }
                H5Fclose(fid);
            }
        }
        remove(path.c_str());
    }
    // v1 round trip.
    checkV1RoundTrip<T>(snap, vp, scalars, OutputFormat::kHdf5, "v1h5");
}
#endif  // CUMES_HAVE_HDF5

template <typename T>
static void runPrecision() {
    printf("== %s precision ==\n",
           sizeof(T) == sizeof(double) ? "double" : "float");
    const int ns = 5, mnmax = 3;
    auto storage = makeStorage<T>(ns, mnmax);

    DeviceParams<T> p{};
    p.ns = ns;
    p.mnmax = mnmax;

    // Minimal valid problem + scalar pack for the writer call sites.
    cumes::ProblemSpec pspec;
    pspec.mpol = 2; pspec.ntor = 0; pspec.nfp = 1;
    pspec.mass.coefficients = {1.0};
    pspec.toroidal_flux.coefficients = {1.0};
    pspec.rbc = {{1, 0, 1.0}};
    pspec.zbs = {{1, 0, 0.5}};
    pspec.stages = {{(size_t)ns, 100, 1e-12}};
    auto vpres = cumes::validate(pspec, cumes::SolverOptions{});
    if (!vpres.has_value()) { fprintf(stderr, "test_io_golden: validate failed\n"); exit(1); }
    cumes::ValidatedProblem vp = std::move(vpres.value());
    cumes::LegacyRunScalars scalars;
    scalars.mpol = 2; scalars.ntor = 0; scalars.nfp = 1;
    scalars.ntheta = 8; scalars.nzeta = 1; scalars.nZnT = 8;
    scalars.ns = ns; scalars.mnmax = mnmax; scalars.ncurr = 0;
    scalars.max_iter = 100; scalars.delt = 0.9; scalars.ftol = 1e-12;
    scalars.lamscale = 0.1; scalars.iterations = 1; scalars.converged = false;
    scalars.fsqr = 1.0; scalars.fsqz = 1.0; scalars.fsql = 1.0;

    // ---- the golden gate: legacy-v0 writer == outputSaveBinary ------------
    const std::string legacyPath = scratch("legacy");
    const std::string v0Path = scratch("v0");
    {
        const bool ok = outputSaveBinary<T>(storage, p,
                                            legacyPath.c_str());
        CHECK(ok, "golden: outputSaveBinary (reference) writes");

        const EquilibriumSnapshot snap = cumes::snapshot_from_device(storage);
        OutputSpec spec;
        spec.format = OutputFormat::kBinary;
        spec.schema = OutputSchema::kLegacyV0;
        spec.path = v0Path;
        auto w = cumes::make_writer(spec.format, spec.schema);
        CHECK(w != nullptr, "golden: legacy-v0 writer factory");
        if (w) {
            RunReport report;  // v0 ignores it
            CHECK(w->write_atomic(snap, report, spec, vp, scalars).has_value(),
                  "golden: legacy-v0 writer writes");
        }

        std::string a, b;
        const bool ra = readFileBytes(legacyPath, a);
        const bool rb = readFileBytes(v0Path, b);
        CHECK(ra && rb, "golden: both files readable");
        if (ra && rb) {
            CHECK(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0,
                  "golden: legacy-v0 writer is byte-identical to outputSaveBinary");
        }
    }

    // ---- v1 writer round-trips the bridged snapshot -----------------------
    {
        const EquilibriumSnapshot snap = cumes::snapshot_from_device(storage);
        const std::string v1Path = scratch("v1");
        OutputSpec spec;
        spec.format = OutputFormat::kBinary;
        spec.schema = OutputSchema::kV1;
        spec.path = v1Path;
        auto w = cumes::make_writer(spec.format, spec.schema);
        auto r = cumes::make_reader(spec.format, spec.schema);
        CHECK(w != nullptr && r != nullptr, "v1: writer+reader factories");
        if (w && r) {
            RunReport report;
            report.status = cumes::RunStatus::kConverged;
            report.build.scalar_type = sizeof(T) == sizeof(double) ? "double" : "float";
            CHECK(w->write_atomic(snap, report, spec, vp, scalars).has_value(), "v1: write succeeds");
            auto back = r->read(v1Path);
            CHECK(back.has_value() && snapshotsEqual(snap, back.value()),
                  "v1: round-trip preserves bridged state");
        }
        remove(v1Path.c_str());
    }

    // ---- NetCDF/HDF5 adapters (completion plan steps 2.2/2.3) -------------
    // v0: the documented legacy layout — fixed capacities (ngrids=8,
    // ncoeff=16, naxis=32, nbm=nbn=16), active state dims, scalar values, and
    // the [surface, mode] hyperslab order. (Byte-exactness against the frozen
    // tree is proven for NetCDF by the full-run compare_bitwise gate; HDF5
    // embeds a library-managed per-second timestamp, so its contract is the
    // layout, verified structurally here.)
    // v1: the complete RunReport + restart metadata round trip.
    const EquilibriumSnapshot snap2 = cumes::snapshot_from_device(storage);
#ifdef CUMES_HAVE_NETCDF
    testNetcdfLayouts<T>(snap2, vp, scalars, ns, mnmax);
#endif
#ifdef CUMES_HAVE_HDF5
    testHdf5Layouts<T>(snap2, vp, scalars, ns, mnmax);
#endif

    // ---- versioned checkpoint round-trips the bridged snapshot ------------
    {
        const EquilibriumSnapshot snap = cumes::snapshot_from_device(storage);
        const std::string ckpt = scratch("ckpt");
        CHECK(cumes::write_checkpoint(snap, ckpt).has_value(),
              "checkpoint: write succeeds");
        auto back = cumes::read_checkpoint(ckpt);
        CHECK(back.has_value() && snapshotsEqual(snap, back.value()),
              "checkpoint: round-trip preserves bridged state");
        remove(ckpt.c_str());
    }

    remove(legacyPath.c_str());
    remove(v0Path.c_str());
}

int main() {
    printf("=== Golden writer byte-identity gate ===\n");
    runPrecision<double>();
    runPrecision<float>();
    if (failures == 0) {
        printf("test_io_golden: ALL PASS\n");
        return 0;
    }
    printf("test_io_golden: %d FAILURES\n", failures);
    return 1;
}
