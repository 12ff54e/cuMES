// test_host_io.cpp — host I/O: output-spec dispatch, binary v0 byte layout and
// round-trip, versioned v1 round-trip, and the writer failure matrix.
//
// This is the Phase 2 gate for the I/O half: the legacy binary v0 container is
// byte-exact (header + six mode-major double families), the versioned v1
// container round-trips its state, and every writer publishes atomically and
// fails cleanly on open/rename errors.
#include "cumes/io/checkpoint.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/reader.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/io/writer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using cumes::EquilibriumSnapshot;
using cumes::OutputFormat;
using cumes::OutputSchema;
using cumes::OutputSpec;
using cumes::RunReport;
using cumes::RunStatus;

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        const std::string _m(msg);                                           \
        if (cond) {                                                          \
            printf("PASS %s\n", _m.c_str());                                 \
        } else {                                                             \
            printf("FAIL %s\n", _m.c_str());                                 \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static std::string scratch(const char* name) {
    return std::string("test_host_io_") + name + "_" +
           std::to_string((long)getpid());
}

// Deterministic snapshot: family c, index i -> value = c*1000 + i + 0.25.
static EquilibriumSnapshot makeSnapshot(int ns, int mnmax) {
    EquilibriumSnapshot s;
    s.ns = ns;
    s.mnmax = mnmax;
    const std::size_t n = s.family_size();
    for (int c = 0; c < 6; ++c) {
        s.families[c].resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            s.families[c][i] = c * 1000.0 + static_cast<double>(i) + 0.25;
        }
    }
    return s;
}

static RunReport makeReport() {
    RunReport r;
    r.status = RunStatus::kConverged;
    r.total_effective_iterations = 7;
    cumes::StageReport stage;
    stage.ns = 5;
    stage.effective_iterations = 7;
    stage.converged = true;
    stage.final_residual = {1e-14, 2e-15, 3e-16};
    r.stages.push_back(stage);
    r.build.revision = "deadbeef";
    r.build.build_type = "Release";
    r.build.scalar_type = "double";
    r.input.source_path = "inputs/solovev.json";
    r.runtime.gpu_name = "test-gpu";
    return r;
}

static bool snapshotsEqual(const EquilibriumSnapshot& a,
                           const EquilibriumSnapshot& b) {
    if (a.ns != b.ns || a.mnmax != b.mnmax) return false;
    for (int c = 0; c < 6; ++c) {
        if (a.families[c] != b.families[c]) return false;
    }
    return true;
}

static void testOutputSpec() {
    CHECK(cumes::output_format_available(OutputFormat::kBinary),
          "output spec: binary always available");
    auto r = cumes::resolve_output_spec("state.bin", false);
    CHECK(r.has_value() && r.value().format == OutputFormat::kBinary,
          "output spec: .bin resolves to binary");
    r = cumes::resolve_output_spec("state.H5", false);
    CHECK(r.has_value() && r.value().format == OutputFormat::kHdf5,
          "output spec: .H5 (case-insensitive) resolves to hdf5");
    r = cumes::resolve_output_spec("state.unknown", false);
    CHECK(!r.has_value(), "output spec: unknown suffix rejected (strict)");
    r = cumes::resolve_output_spec("state.unknown", true);
    CHECK(r.has_value() && r.value().format == OutputFormat::kBinary,
          "output spec: unknown suffix falls back to binary (compat)");
}

static void testBinaryByteLayout() {
    // A tiny 2x1 snapshot; verify the exact on-disk bytes (little-endian
    // int32 header + six mode-major double families).
    EquilibriumSnapshot s;
    s.ns = 2;
    s.mnmax = 1;
    for (int c = 0; c < 6; ++c) {
        s.families[c] = {c * 10.0, c * 10.0 + 1.0};
    }
    OutputSpec spec;
    spec.format = OutputFormat::kBinary;
    spec.schema = OutputSchema::kLegacyV0;
    spec.path = scratch("byte").c_str();
    auto w = cumes::make_writer(spec.format, spec.schema);
    CHECK(w != nullptr, "binary v0: writer factory returns a writer");
    if (!w) return;
    auto st = w->write_atomic(s, makeReport(), spec);
    CHECK(st.has_value(), "binary v0: byte-layout write succeeds");

    std::ifstream in(spec.path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
    // header: int32 ns=2, int32 mnmax=1
    const std::int32_t ns = 2, mnmax = 1;
    CHECK(bytes.size() == 8 + 6 * 2 * 8, "binary v0: file size == 8 + 6*2*8");
    CHECK(std::memcmp(bytes.data(), &ns, 4) == 0 &&
              std::memcmp(bytes.data() + 4, &mnmax, 4) == 0,
          "binary v0: int32 header (ns, mnmax)");
    // first family: {0.0, 1.0}
    const double* fam0 = reinterpret_cast<const double*>(bytes.data() + 8);
    CHECK(fam0[0] == 0.0 && fam0[1] == 1.0, "binary v0: family 0 (rmncc) bytes");
    // family 1 (zmnsc) = {10.0, 11.0}
    const double* fam1 = reinterpret_cast<const double*>(bytes.data() + 8 + 2 * 8);
    CHECK(fam1[0] == 10.0 && fam1[1] == 11.0, "binary v0: family 1 (zmnsc) bytes");
    remove(spec.path.c_str());
}

static void testBinaryRoundTrip() {
    EquilibriumSnapshot s = makeSnapshot(5, 3);
    OutputSpec spec;
    spec.format = OutputFormat::kBinary;
    spec.schema = OutputSchema::kLegacyV0;
    spec.path = scratch("rt").c_str();
    auto w = cumes::make_writer(spec.format, spec.schema);
    auto r = cumes::make_reader(spec.format, spec.schema);
    CHECK(w != nullptr && r != nullptr, "binary v0: writer+reader factories");
    if (!w || !r) return;
    CHECK(w->write_atomic(s, makeReport(), spec).has_value(),
          "binary v0: write succeeds");
    auto back = r->read(spec.path);
    CHECK(back.has_value() && snapshotsEqual(s, back.value()),
          "binary v0: round-trip preserves state");
    remove(spec.path.c_str());
}

static void testV1RoundTrip() {
    EquilibriumSnapshot s = makeSnapshot(4, 2);
    OutputSpec spec;
    spec.format = OutputFormat::kBinary;
    spec.schema = OutputSchema::kV1;
    spec.path = scratch("v1").c_str();
    auto w = cumes::make_writer(spec.format, spec.schema);
    auto r = cumes::make_reader(spec.format, spec.schema);
    CHECK(w != nullptr && r != nullptr, "v1: writer+reader factories");
    if (!w || !r) return;
    CHECK(w->write_atomic(s, makeReport(), spec).has_value(), "v1: write succeeds");
    auto back = r->read(spec.path);
    CHECK(back.has_value() && snapshotsEqual(s, back.value()),
          "v1: round-trip preserves state");

    // Corrupt the magic -> reader rejects.
    {
        std::fstream f(spec.path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(0);
        f.put('X');
        f.close();
    }
    auto bad = r->read(spec.path);
    CHECK(!bad.has_value(), "v1: bad magic rejected");
    remove(spec.path.c_str());
}

static void testFailureMatrix() {
    EquilibriumSnapshot s = makeSnapshot(4, 2);
    // open failure: a path in a nonexistent directory.
    OutputSpec spec;
    spec.format = OutputFormat::kBinary;
    spec.schema = OutputSchema::kLegacyV0;
    spec.path = "no_such_dir/test_host_io_open.bin";
    auto w = cumes::make_writer(spec.format, spec.schema);
    CHECK(w->write_atomic(s, makeReport(), spec).has_value() == false,
          "failure: open failure returns false");

    // rename failure: target is a non-empty directory (temp writes fine,
    // rename() fails, temp removed, target untouched).
    const std::string dir = scratch("dir");
    mkdir(dir.c_str(), 0755);
    {
        FILE* f = fopen((dir + "/keep").c_str(), "w");
        if (f) { fputs("x", f); fclose(f); }
    }
    spec.path = dir;  // a directory, not a file
    CHECK(w->write_atomic(s, makeReport(), spec).has_value() == false,
          "failure: rename-over-directory returns false");
    // the directory and its contents are untouched
    FILE* keep = fopen((dir + "/keep").c_str(), "r");
    CHECK(keep != nullptr, "failure: target directory untouched");
    if (keep) fclose(keep);
    remove((dir + "/keep").c_str());
    rmdir(dir.c_str());
}

int main() {
    testOutputSpec();
    testBinaryByteLayout();
    testBinaryRoundTrip();
    testV1RoundTrip();
    testFailureMatrix();
    if (failures == 0) {
        printf("test_host_io: ALL PASS\n");
        return 0;
    }
    printf("test_host_io: %d FAILURES\n", failures);
    return 1;
}
