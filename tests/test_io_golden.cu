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

template <typename T>
static void runPrecision() {
    printf("== %s precision ==\n",
           sizeof(T) == sizeof(double) ? "double" : "float");
    const int ns = 5, mnmax = 3;
    auto storage = makeStorage<T>(ns, mnmax);

    DeviceParams<T> p{};
    p.ns = ns;
    p.mnmax = mnmax;

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
            CHECK(w->write_atomic(snap, report, spec).has_value(),
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
            CHECK(w->write_atomic(snap, report, spec).has_value(), "v1: write succeeds");
            auto back = r->read(v1Path);
            CHECK(back.has_value() && snapshotsEqual(snap, back.value()),
                  "v1: round-trip preserves bridged state");
        }
        remove(v1Path.c_str());
    }

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
