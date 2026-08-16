// test_checkpoint.cpp — versioned checkpoint and legacy init converter.
//
// Verifies the Phase 2 "versioned checkpoint reader/converter replacing
// CUMES_LOAD_INIT" deliverable: a versioned checkpoint round-trips, a corrupt
// magic/version/dimension is rejected, and the legacy six-family
// vmecpp_init.bin payload converts to the same host snapshot (with header
// mismatch rejected).
#include "cumes/io/checkpoint.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>

using cumes::EquilibriumSnapshot;

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
    return std::string("test_checkpoint_") + name + "_" +
           std::to_string((long)getpid());
}

static EquilibriumSnapshot makeSnapshot(int ns, int mnmax) {
    EquilibriumSnapshot s;
    s.ns = ns;
    s.mnmax = mnmax;
    const std::size_t n = s.family_size();
    for (int c = 0; c < 6; ++c) {
        s.families[c].resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            s.families[c][i] = c * 1000.0 + static_cast<double>(i) + 0.5;
        }
    }
    return s;
}

static bool snapshotsEqual(const EquilibriumSnapshot& a,
                           const EquilibriumSnapshot& b) {
    if (a.ns != b.ns || a.mnmax != b.mnmax) return false;
    for (int c = 0; c < 6; ++c) {
        if (a.families[c] != b.families[c]) return false;
    }
    return true;
}

static void testCheckpointRoundTrip() {
    EquilibriumSnapshot s = makeSnapshot(5, 3);
    const std::string path = scratch("rt");
    CHECK(cumes::write_checkpoint(s, path).has_value(), "checkpoint: write succeeds");
    auto back = cumes::read_checkpoint(path);
    CHECK(back.has_value() && snapshotsEqual(s, back.value()),
          "checkpoint: round-trip preserves state");
    remove(path.c_str());
}

static void testCheckpointRejection() {
    EquilibriumSnapshot s = makeSnapshot(4, 2);
    const std::string path = scratch("bad");
    cumes::write_checkpoint(s, path);
    // Corrupt the magic byte.
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(0);
        f.put('X');
        f.close();
    }
    auto bad = cumes::read_checkpoint(path);
    CHECK(!bad.has_value(), "checkpoint: bad magic rejected");
    remove(path.c_str());

    // A truncated checkpoint (header only) is rejected.
    const std::string tpath = scratch("trunc");
    FILE* f = fopen(tpath.c_str(), "wb");
    fwrite("CUMECKP1", 1, 8, f);
    fclose(f);
    auto trunc = cumes::read_checkpoint(tpath);
    CHECK(!trunc.has_value(), "checkpoint: truncated header rejected");
    remove(tpath.c_str());
}

static void testCheckpointCorruptHugeDimensions() {
    // A corrupt checkpoint header with ns*mnmax*48 bytes in [2^63, 2^64) used
    // to wrap the size_t -> long long cast in checkStateDimensions negative,
    // pass the file-size bound, and die in fam.resize(~2e17) with an uncaught
    // std::bad_alloc (std::terminate). The reader must return the
    // "dimensions implausible" error instead.
    const std::string path = scratch("hugedim");
    {
        FILE* f = fopen(path.c_str(), "wb");
        fwrite("CUMECKP1", 1, 8, f);
        const std::int32_t version = 1, precision = 0;
        const std::int32_t ns = 2147483647, mnmax = 95000000;
        fwrite(&version, sizeof(version), 1, f);
        fwrite(&precision, sizeof(precision), 1, f);
        fwrite(&ns, sizeof(ns), 1, f);
        fwrite(&mnmax, sizeof(mnmax), 1, f);
        fclose(f);
    }
    auto got = cumes::read_checkpoint(path);
    CHECK(!got.has_value(),
          "checkpoint: huge ns*mnmax rejected as implausible (no bad_alloc/terminate)");
    remove(path.c_str());
}

static void testConvertLegacyInit() {
    // Write a legacy six-family payload (int32 ns, int32 mnmax, 6 families).
    EquilibriumSnapshot s = makeSnapshot(4, 2);
    const std::string path = scratch("init");
    FILE* f = fopen(path.c_str(), "wb");
    const std::int32_t ns = 4, mnmax = 2;
    fwrite(&ns, sizeof(ns), 1, f);
    fwrite(&mnmax, sizeof(mnmax), 1, f);
    for (int c = 0; c < 6; ++c) {
        fwrite(s.families[c].data(), sizeof(double), s.families[c].size(), f);
    }
    fclose(f);

    auto conv = cumes::convert_legacy_init(path, 4, 2);
    CHECK(conv.has_value() && snapshotsEqual(s, conv.value()),
          "legacy init: converts to the same snapshot");

    // Header mismatch is rejected.
    auto mism = cumes::convert_legacy_init(path, 5, 2);
    CHECK(!mism.has_value(), "legacy init: ns mismatch rejected");
    remove(path.c_str());
}

int main() {
    testCheckpointRoundTrip();
    testCheckpointRejection();
    testCheckpointCorruptHugeDimensions();
    testConvertLegacyInit();
    if (failures == 0) {
        printf("test_checkpoint: ALL PASS\n");
        return 0;
    }
    printf("test_checkpoint: %d FAILURES\n", failures);
    return 1;
}
