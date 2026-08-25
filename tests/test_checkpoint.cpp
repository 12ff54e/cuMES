// test_checkpoint.cpp — versioned checkpoint.
//
// Verifies the Phase 2 "versioned checkpoint replacing CUMES_LOAD_INIT"
// deliverable: a versioned checkpoint round-trips and a corrupt
// magic/version/dimension is rejected.
#include "cumes/io/checkpoint.hpp"
#include "cumes_test.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <unistd.h>
using namespace cumes::test;

using cumes::EquilibriumSnapshot;

static std::string scratch(const char* name) {
    return std::string("test_checkpoint_") + name + "_" +
           std::to_string((long)getpid());
}

static EquilibriumSnapshot make_snapshot(int ns, int mnmax) {
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

static bool snapshots_equal(const EquilibriumSnapshot& a,
                            const EquilibriumSnapshot& b) {
    if (a.ns != b.ns || a.mnmax != b.mnmax) return false;
    for (int c = 0; c < 6; ++c) {
        if (a.families[c] != b.families[c]) return false;
    }
    return true;
}

// A small but non-trivial embedded input record for the checkpoint trailer.
static cumes::InputParams make_params() {
    cumes::InputParams p;
    p.mpol = 2;
    p.ntor = 0;
    p.nfp = 1;
    p.ntheta = 10;
    p.nzeta = 1;
    p.ncurr = 0;
    p.delt = 0.9;
    p.phiedge = 1.0;
    p.am = {1.0};
    p.aphi = {1.0};
    p.rbc_m = {1, 0};
    p.rbc_n = {0, 0};
    p.rbc_value = {1.0, 2.0};
    p.zbs_m = {1};
    p.zbs_n = {0};
    p.zbs_value = {0.5};
    p.rbcc = {1.0, 2.0};
    p.rbss = {0.0, 0.0};
    p.zbsc = {0.0, 0.0};
    p.zbcs = {0.5, 0.0};
    p.stages = {{5, 100, 1e-12}};
    return p;
}

static void test_checkpoint_round_trip() {
    EquilibriumSnapshot s = make_snapshot(5, 3);
    const cumes::InputParams params = make_params();
    const std::string path = scratch("rt");
    check(cumes::write_checkpoint(s, params, path).has_value(),
          "checkpoint: write succeeds");
    cumes::InputParams back_params;
    auto back = cumes::read_checkpoint(path, back_params);
    check(back.has_value() && snapshots_equal(s, back.value()),
          "checkpoint: round-trip preserves state");
    check(back_params == params,
          "checkpoint: round-trip preserves the input record");
    remove(path.c_str());
}

static void test_checkpoint_version_gate() {
    // The version-2 input record rides after the families: a version-1
    // checkpoint (no record) stays readable with a default-empty record, and
    // a version beyond the writer's is rejected.
    EquilibriumSnapshot s = make_snapshot(4, 2);
    const std::string path = scratch("vgate");
    cumes::write_checkpoint(s, make_params(), path);
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(8);
        const std::int32_t v1 = 1;
        f.write(reinterpret_cast<const char*>(&v1), sizeof(v1));
        f.close();
    }
    {
        cumes::InputParams back_params;
        auto back = cumes::read_checkpoint(path, back_params);
        check(back.has_value() && snapshots_equal(s, back.value()),
              "checkpoint: version-1 file still readable");
        check(back_params == cumes::InputParams{},
              "checkpoint: version-1 record absent (defaults)");
    }
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(8);
        const std::int32_t unsupported_version = 7;
        f.write(reinterpret_cast<const char*>(&unsupported_version),
                sizeof(unsupported_version));
        f.close();
    }
    {
        auto bad = cumes::read_checkpoint(path);
        check(!bad.has_value(), "checkpoint: unsupported version rejected");
    }
    remove(path.c_str());
}

static void test_checkpoint_rejection() {
    EquilibriumSnapshot s = make_snapshot(4, 2);
    const std::string path = scratch("bad");
    cumes::write_checkpoint(s, make_params(), path);
    // Corrupt the magic byte.
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(0);
        f.put('X');
        f.close();
    }
    auto bad = cumes::read_checkpoint(path);
    check(!bad.has_value(), "checkpoint: bad magic rejected");
    remove(path.c_str());

    // A truncated checkpoint (header only) is rejected.
    const std::string tpath = scratch("trunc");
    FILE* f = fopen(tpath.c_str(), "wb");
    fwrite("CUMECKP1", 1, 8, f);
    fclose(f);
    auto trunc = cumes::read_checkpoint(tpath);
    check(!trunc.has_value(), "checkpoint: truncated header rejected");
    remove(tpath.c_str());
}

static void test_checkpoint_corrupt_huge_dimensions() {
    // A corrupt checkpoint header with ns*mnmax*48 bytes in [2^63, 2^64) used
    // to wrap the size_t -> long long cast in check_state_dimensions negative,
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
    check(!got.has_value(),
          "checkpoint: huge ns*mnmax rejected as implausible (no "
          "bad_alloc/terminate)");
    remove(path.c_str());
}

int main() {
    test_checkpoint_round_trip();
    test_checkpoint_version_gate();
    test_checkpoint_rejection();
    test_checkpoint_corrupt_huge_dimensions();
    return summary();
}
