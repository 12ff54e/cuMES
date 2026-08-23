// test_io_golden.cu — the v1 writer/reader round-trip gate.
//
// Closes the bridge->reader loop: the same device state is bridged through
// snapshot_from_device and round-tripped by the v1 binary writer (including
// the historical version-1 trailer fixture), the v1 NetCDF/HDF5 adapters
// (state + complete RunReport + restart metadata), and the versioned
// checkpoint.
//
// Runs both precisions (double is the verification config; float proves the
// T->double conversion is identical for the single-precision build).
#include "cumes/io/checkpoint.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/reader.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/io/writer.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>
using namespace cumes::test;

using cumes::EquilibriumSnapshot;
using cumes::OutputFormat;
using cumes::OutputSpec;
using cumes::RunReport;

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
    check_cuda(cudaMemcpy(storage.state_slab(), host.data(), count * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upload");
    return storage;
}

// Hand-write a version = 1 container with the HISTORICAL trailer layout:
// after build_type came a scalar_type string, where v2 carries
// precision_policy + compile_flags instead. The reader must consume and
// discard the historical slot so the fields after it land correctly.
static bool writeHistoricalV1(const EquilibriumSnapshot& snap,
                              const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    auto w_i32 = [&](std::int32_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto w_u8 = [&](std::uint8_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto w_str = [&](const char* s) {
        const std::int32_t n = static_cast<std::int32_t>(std::strlen(s));
        w_i32(n);
        f.write(s, n);
    };
    const char MAGIC[8] = {'C', 'U', 'M', 'E', 'S', '0', '0', '1'};
    f.write(MAGIC, 8);
    w_i32(1);  // version (historical)
    w_i32(snap.ns);
    w_i32(snap.mnmax);
    for (const auto& fam : snap.families) {
        f.write(reinterpret_cast<const char*>(fam.data()),
                static_cast<std::streamsize>(fam.size() * sizeof(double)));
    }
    w_i32(0);          // precision = double
    w_i32(0);          // status = CONVERGED
    w_i32(42);         // total_effective_iterations
    w_i32(0);          // nstages
    w_str("r1");       // revision
    w_u8(0);           // dirty
    w_str("Release");  // build_type
    w_str("double");   // scalar_type (the historical v1 slot)
    w_str("");         // source_path
    w_str("");         // source_hash
    w_str("");         // gpu_name
    w_str("");         // driver
    w_str("");         // runtime
    w_str("");         // toolkit
    return f.good();
}

// Hand-write a version = 3 container: the trailer carries the embedded input
// record in the HISTORICAL layout WITHOUT the three profile-type strings
// (version 4 appends them after the schema tag). The reader must walk the old
// record and keep the "power_series" defaults.
static bool writeHistoricalV3(const EquilibriumSnapshot& snap,
                              const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    auto w_i32 = [&](std::int32_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto w_u8 = [&](std::uint8_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto w_f64 = [&](double v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto w_str = [&](const char* s) {
        const std::int32_t n = static_cast<std::int32_t>(std::strlen(s));
        w_i32(n);
        f.write(s, n);
    };
    auto w_f64_vec = [&](const std::vector<double>& v) {
        w_i32(static_cast<std::int32_t>(v.size()));
        if (!v.empty())
            f.write(reinterpret_cast<const char*>(v.data()),
                    static_cast<std::streamsize>(v.size() * sizeof(double)));
    };
    auto w_i32_vec = [&](const std::vector<int>& v) {
        w_i32(static_cast<std::int32_t>(v.size()));
        if (!v.empty())
            f.write(
                reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(std::int32_t)));
    };
    const char MAGIC[8] = {'C', 'U', 'M', 'E', 'S', '0', '0', '1'};
    f.write(MAGIC, 8);
    w_i32(3);  // version (historical: record without profile-type strings)
    w_i32(snap.ns);
    w_i32(snap.mnmax);
    for (const auto& fam : snap.families) {
        f.write(reinterpret_cast<const char*>(fam.data()),
                static_cast<std::streamsize>(fam.size() * sizeof(double)));
    }
    w_i32(0);                // precision = double
    w_i32(0);                // status = CONVERGED
    w_i32(42);               // total_effective_iterations
    w_i32(0);                // nstages
    w_str("r3");             // revision
    w_u8(0);                 // dirty
    w_str("Release");        // build_type
    w_str("verify-double");  // precision_policy (the v2 trailer pair)
    w_str("");               // compile_flags
    w_str("");               // source_path
    w_str("");               // source_hash
    w_str("");               // gpu_name
    w_str("");               // driver
    w_str("");               // runtime
    w_str("");               // toolkit
    // Old input record: 6*i32 + 8*f64 + schema + vectors, no type strings.
    w_i32(2);    // mpol
    w_i32(0);    // ntor
    w_i32(1);    // nfp
    w_i32(2);    // ntheta
    w_i32(1);    // nzeta
    w_i32(0);    // ncurr
    w_f64(0.9);  // delt
    w_f64(1.0);  // phiedge
    w_f64(1.0);  // pres_scale
    w_f64(0.0);  // adiabatic_index
    w_f64(1.0);  // spres_ped
    w_f64(1.0);  // bloat
    w_f64(0.0);  // curtor
    w_f64(1.0);  // tcon0
    w_str("cumes-config-v1");
    w_f64_vec({1.0, 2.0});  // am (proves the old layout walks correctly)
    w_f64_vec({});          // ac
    w_f64_vec({});          // ai
    w_f64_vec({1.0});       // aphi
    w_f64_vec({});          // raxis_c
    w_f64_vec({});          // zaxis_s
    w_i32(0);               // nstages_in
    w_i32_vec({});          // rbc_m
    w_i32_vec({});          // rbc_n
    w_f64_vec({});          // rbc_value
    w_i32_vec({});          // zbs_m
    w_i32_vec({});          // zbs_n
    w_f64_vec({});          // zbs_value
    w_f64_vec({});          // rbcc
    w_f64_vec({});          // rbss
    w_f64_vec({});          // zbsc
    w_f64_vec({});          // zbcs
    return f.good();
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
// (completion plan step 2.3). The embedded input record derives from the
// same validated problem the writers use for the native boundary fields, so
// the NetCDF/HDF5 round trip is exact.
static RunReport makeIoReport(const cumes::ValidatedProblem& vp) {
    RunReport r;
    r.status = cumes::RunStatus::CONVERGED;
    r.total_effective_iterations = 21;
    cumes::StageReport s1;
    s1.ns = 5;
    s1.effective_iterations = 9;
    s1.converged = true;
    s1.final_residual = {1e-10, 2e-11, 3e-12};
    s1.restarts = {cumes::RestartEvent{2}, cumes::RestartEvent{5}};
    cumes::StageReport s2;
    s2.ns = 8;
    s2.effective_iterations = 12;
    s2.converged = true;
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
    r.input_params = cumes::make_input_params(vp);
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
    if (!(a.input_params == b.input_params)) return false;
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
            if (x.restarts[k].iteration != y.restarts[k].iteration)
                return false;
        }
    }
    return true;
}

// v1 round trip shared by both backends: state + complete RunReport.
template <typename T>
static void checkV1RoundTrip(const EquilibriumSnapshot& snap,
                             const cumes::ValidatedProblem& vp,
                             OutputFormat fmt,
                             const char* tag) {
    const std::string path = scratch(tag);
    OutputSpec spec;
    spec.format = fmt;
    spec.path = path;
    auto w = cumes::make_writer(spec.format);
    auto r = cumes::make_reader(spec.format);
    check(w != nullptr && r != nullptr, "v1 factories");
    if (w && r) {
        RunReport report = makeIoReport(vp);
        report.build.scalar_type =
            sizeof(T) == sizeof(double) ? "double" : "float";
        check(w->write_atomic(snap, report, spec, vp).has_value(),
              "v1 write succeeds");
        RunReport back;
        auto snap_back = r->read(path, &back);
        check(snap_back.has_value() && snapshotsEqual(snap, snap_back.value()),
              "v1 state round trip");
        check(reportsEqual(report, back),
              "v1 complete RunReport + restart metadata round trip");
    }
    remove(path.c_str());
}

template <typename T>
static void runPrecision() {
    std::cout << format("== {} precision ==\n",
                        sizeof(T) == sizeof(double) ? "double" : "float");
    const int ns = 5, mnmax = 3;
    auto storage = makeStorage<T>(ns, mnmax);

    // Minimal valid problem for the writer call sites.
    cumes::ProblemSpec pspec;
    pspec.mpol = 2;
    pspec.ntor = 0;
    pspec.nfp = 1;
    pspec.mass.coefficients = {1.0};
    pspec.toroidal_flux.coefficients = {1.0};
    pspec.rbc = {{1, 0, 1.0}};
    pspec.zbs = {{1, 0, 0.5}};
    pspec.stages = {{(size_t)ns, 100, 1e-12}};
    auto vpres = cumes::validate(pspec, cumes::SolverOptions{});
    if (!vpres.has_value()) {
        std::cerr << "test_io_golden: validate failed\n";
        exit(1);
    }
    cumes::ValidatedProblem vp = std::move(vpres.value());
    // ---- v1 writer round-trips the bridged snapshot + provenance trailer --
    {
        const EquilibriumSnapshot snap = cumes::snapshot_from_device(storage);
        const std::string v1Path = scratch("v1");
        OutputSpec spec;
        spec.format = OutputFormat::BINARY;
        spec.path = v1Path;
        auto w = cumes::make_writer(spec.format);
        auto r = cumes::make_reader(spec.format);
        check(w != nullptr && r != nullptr, "v1: writer+reader factories");
        if (w && r) {
            // Read back WITH the report: the v2 trailer (precision-policy
            // fields + stage records) must round-trip, not just the state.
            // (The trailer sequences were once out of sync and a state-only
            // read could not see it — see docs/output-formats.md.)
            RunReport report = makeIoReport(vp);
            report.build.scalar_type =
                sizeof(T) == sizeof(double) ? "double" : "float";
            check(w->write_atomic(snap, report, spec, vp).has_value(),
                  "v1: write succeeds");
            RunReport back;
            auto snap_back = r->read(v1Path, &back);
            check(snap_back.has_value() &&
                      snapshotsEqual(snap, snap_back.value()),
                  "v1: round-trip preserves bridged state");
            check(reportsEqual(report, back),
                  "v1: round-trip preserves the provenance trailer");
        }
        remove(v1Path.c_str());
    }

    // ---- v1 historical layout (version = 1): the trailer carried a
    // scalar_type string where v2 carries the precision-policy pair; the
    // reader must consume and discard it so the later fields land right. --
    {
        const EquilibriumSnapshot snap = cumes::snapshot_from_device(storage);
        const std::string v1OldPath = scratch("v1old");
        check(writeHistoricalV1(snap, v1OldPath),
              "v1 historical: fixture written");
        auto r = cumes::make_reader(OutputFormat::BINARY);
        check(r != nullptr, "v1 historical: reader factory");
        if (r) {
            RunReport back;
            auto snap_back = r->read(v1OldPath, &back);
            check(snap_back.has_value() &&
                      snapshotsEqual(snap, snap_back.value()),
                  "v1 historical: state round trip");
            check(back.build.scalar_type == "double" &&
                      back.build.revision == "r1" &&
                      back.build.build_type == "Release" &&
                      back.build.precision_policy == "" &&
                      back.build.compile_flags == "" &&
                      back.input.source_path == "" &&
                      back.total_effective_iterations == 42 &&
                      back.stages.empty() &&
                      back.input_params == cumes::InputParams{},
                  "v1 historical: trailer fields land correctly");
        }
        remove(v1OldPath.c_str());
    }

    // ---- v3 historical layout: the embedded record lacks the three
    // profile-type strings (version 4 appends them); the reader must walk
    // the old record and keep the "power_series" defaults. --
    {
        const EquilibriumSnapshot snap = cumes::snapshot_from_device(storage);
        const std::string v3OldPath = scratch("v3old");
        check(writeHistoricalV3(snap, v3OldPath),
              "v3 historical: fixture written");
        auto r = cumes::make_reader(OutputFormat::BINARY);
        check(r != nullptr, "v3 historical: reader factory");
        if (r) {
            RunReport back;
            auto snap_back = r->read(v3OldPath, &back);
            check(snap_back.has_value() &&
                      snapshotsEqual(snap, snap_back.value()),
                  "v3 historical: state round trip");
            check(back.build.scalar_type == "double" &&
                      back.build.revision == "r3" &&
                      back.total_effective_iterations == 42 &&
                      back.stages.empty(),
                  "v3 historical: trailer fields land correctly");
            check(back.input_params.pmass_type == "power_series" &&
                      back.input_params.piota_type == "power_series" &&
                      back.input_params.pcurr_type == "power_series",
                  "v3 historical: profile types default to power_series");
            check(back.input_params.am.size() == 2 &&
                      back.input_params.am[0] == 1.0 &&
                      back.input_params.am[1] == 2.0 &&
                      back.input_params.aphi.size() == 1 &&
                      back.input_params.aphi[0] == 1.0,
                  "v3 historical: old record vectors land correctly");
        }
        remove(v3OldPath.c_str());
    }

    // ---- NetCDF/HDF5 adapters (completion plan steps 2.2/2.3) -------------
    // v1: the complete RunReport + restart metadata round trip.
    const EquilibriumSnapshot snap2 = cumes::snapshot_from_device(storage);
#ifdef CUMES_HAVE_NETCDF
    checkV1RoundTrip<T>(snap2, vp, OutputFormat::NETCDF, "v1nc");
#endif
#ifdef CUMES_HAVE_HDF5
    checkV1RoundTrip<T>(snap2, vp, OutputFormat::HDF5, "v1h5");
#endif

    // ---- versioned checkpoint round-trips the bridged snapshot ------------
    {
        const EquilibriumSnapshot snap = cumes::snapshot_from_device(storage);
        const std::string ckpt = scratch("ckpt");
        const cumes::InputParams ip = cumes::make_input_params(vp);
        check(cumes::write_checkpoint(snap, ip, ckpt).has_value(),
              "checkpoint: write succeeds");
        cumes::InputParams back_params;
        auto back = cumes::read_checkpoint(ckpt, &back_params);
        check(back.has_value() && snapshotsEqual(snap, back.value()),
              "checkpoint: round-trip preserves bridged state");
        check(back_params == ip,
              "checkpoint: round-trip preserves the input record");
        remove(ckpt.c_str());
    }
}

int main() {
    std::cout << "=== v1 writer round-trip gate ===\n";
    runPrecision<double>();
    runPrecision<float>();
    return summary();
}
