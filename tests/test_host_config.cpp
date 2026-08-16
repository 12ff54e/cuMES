// test_host_config.cpp — validated host model: normalization goldens, folding,
// mode table, precision floor, and the malformed-input matrix.
//
// This is the Phase 2 gate for the config half: `read_and_validate` maps the
// legacy parser's defaults/folding into the immutable model, and its canonical
// `normalize_to_json()` must match the checked-in goldens.
//
// The negative matrix pins the containment-series validation into the new
// model: nonzero gamma, out-of-range boundary modes (skipped with a warning),
// empty/mismatched/non-monotonic schedules, integer narrowing, wrong-type
// aux/asym keys, unsupported physics, and unknown-key strict vs compatibility.
#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using cumes::PrecisionPolicy;
using cumes::ProblemSpec;
using cumes::Severity;
using cumes::SolverOptions;
using cumes::ValidationResult;
using cumes::read_and_validate;

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

static const char* scratchPath() {
    static char buf[64];
    snprintf(buf, sizeof buf, "test_host_config_scratch_%d.json", (int)getpid());
    return buf;
}

static void writeScratch(const std::string& content) {
    FILE* fp = fopen(scratchPath(), "w");
    if (!fp) { fprintf(stderr, "cannot write scratch file\n"); exit(1); }
    fputs(content.c_str(), fp);
    fclose(fp);
}

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

// Return the first error message whose text contains `fragment`, or "" if none.
static std::string findError(const ValidationResult& vr, const std::string& fragment) {
    if (vr.has_value()) return "";
    for (const auto& issue : vr.error().issues()) {
        if (issue.severity == Severity::kError &&
            issue.message.find(fragment) != std::string::npos) {
            return issue.message;
        }
    }
    return "";
}
// ---- golden emission (--emit-golden <dir>) ---------------------------------
static void emitGoldens(const char* dir) {
    SolverOptions opts;  // compatibility (defaults); goldens are config-only
    const std::string prefix = std::string(dir) + "/";
    const char* cases[2] = {"solovev", "w7x"};
    for (const char* name : cases) {
        auto vr = read_and_validate(std::string("inputs/") + name + ".json", opts);
        if (!vr.has_value()) {
            fprintf(stderr, "emit-golden: %s failed validation\n", name);
            exit(1);
        }
        const std::string out = prefix + name + ".normalized.json";
        FILE* fp = fopen(out.c_str(), "w");
        if (!fp) { fprintf(stderr, "cannot write %s\n", out.c_str()); exit(1); }
        fputs(vr.value().normalize_to_json().c_str(), fp);
        fclose(fp);
        printf("wrote %s\n", out.c_str());
    }
}

// ---- tests ------------------------------------------------------------------

static void testGoldens() {
    SolverOptions opts;
    const char* names[2] = {"solovev", "w7x"};
    for (const char* name : names) {
        auto vr = read_and_validate(std::string("inputs/") + name + ".json", opts);
        CHECK(vr.has_value(), std::string(name) + ": validates");
        if (!vr.has_value()) continue;
        const std::string golden =
            readFile(std::string("tests/fixtures/") + name + ".normalized.json");
        CHECK(!golden.empty(), std::string(name) + ": golden fixture present");
        CHECK(vr.value().normalize_to_json() == golden,
              std::string(name) + ": normalize_to_json matches golden");
    }
}

static void testModeTable() {
    SolverOptions opts;
    auto vr = read_and_validate("inputs/solovev.json", opts);
    CHECK(vr.has_value(), "solovev validates for mode table");
    if (!vr.has_value()) return;
    const auto& vp = vr.value();
    const auto& table = vp.mode_table();
    CHECK(table.size() == static_cast<std::size_t>(6 * 1), "solovev: modes()=6");
    // mode = m*(ntor+1)+n; m=0,n=0 -> even parity, mn_scale 1, xmpq 0.
    CHECK(table[0].m == 0 && table[0].n == 0 && table[0].parity == cumes::ModeParity::kEven &&
              table[0].mn_scale == 1.0 && table[0].xmpq == 0.0,
          "solovev: (0,0) entry");
    // m=1,n=0 -> odd, mn_scale sqrt(2), xmpq 0, first_surface 1.
    const auto& m1 = table[1];
    CHECK(m1.m == 1 && m1.n == 0 && m1.parity == cumes::ModeParity::kOdd &&
              m1.mn_scale == std::sqrt(2.0) && m1.first_surface == 1,
          "solovev: (1,0) entry");
    // m=2,n=0 -> even, mn_scale sqrt(2), xmpq 2.
    const auto& m2 = table[2];
    CHECK(m2.parity == cumes::ModeParity::kEven && m2.mn_scale == std::sqrt(2.0) &&
              m2.xmpq == 2.0 && m2.first_surface == 1,
          "solovev: (2,0) entry");

    // w7x: physical_n = n*nfp.
    auto vr2 = read_and_validate("inputs/w7x.json", opts);
    CHECK(vr2.has_value(), "w7x validates for mode table");
    if (!vr2.has_value()) return;
    const auto& t2 = vr2.value().mode_table();
    const std::size_t mode11 = 1 * (12 + 1) + 1;  // m=1, n=1
    CHECK(t2[mode11].m == 1 && t2[mode11].n == 1 && t2[mode11].physical_n == 5,
          "w7x: physical_n = n*nfp = 5");
}

static void testPrecisionFloor() {
    SolverOptions opts;
    opts.precision = PrecisionPolicy::kMixedFloat;
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
                 " \"ns_array\": [5], \"niter_array\": [100], \"ftol_array\": [1e-16]}");
    auto vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "floor").empty(),
          "precision floor: float rejects ftol=1e-16");
}

static void testMalformed() {
    SolverOptions opts;
    // Missing file -> throws (JSON syntax/file error).
    bool thrown = false;
    try {
        read_and_validate("no_such_file.json", opts);
    } catch (const std::runtime_error& e) {
        thrown = std::string(e.what()).find("not found") != std::string::npos;
    }
    CHECK(thrown, "malformed: missing file throws");

    // Wrong type for mpol -> collected error.
    writeScratch("{\"mpol\": \"six\"}");
    auto vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "expected an integer").empty(),
          "malformed: wrong type for mpol");

    // Nonzero gamma.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"adiabatic_index\": 0.5,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "gamma").empty(),
          "malformed: nonzero gamma rejected");

    // Empty schedule.
    writeScratch("{\"ns_array\": [], \"niter_array\": [], \"ftol_array\": []}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "at least one stage").empty(),
          "malformed: empty schedule rejected");

    // Non-monotonic schedule.
    writeScratch("{\"ns_array\": [55, 11], \"niter_array\": [100, 200],"
                 " \"ftol_array\": [1e-12, 1e-12]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "strictly increasing").empty(),
          "malformed: non-monotonic schedule rejected");

    // Equal consecutive ns: the grid-prolongation precondition needs strictly
    // finer grids (ns_new > ns_old), so a flat pair must be rejected up front
    // instead of dying inside Prolongation::enqueue (review 1.3).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
                 " \"ns_array\": [11, 11, 55], \"niter_array\": [100, 100, 100],"
                 " \"ftol_array\": [1e-12, 1e-12, 1e-12]}");
    vr = read_and_validate(scratchPath(), opts);
    printf("  equal-ns {11,11,55}: validation %s\n",
           vr.has_value() ? "PASSED" : "rejected");
    CHECK(!vr.has_value() && !findError(vr, "strictly increasing").empty(),
          "malformed: equal consecutive ns rejected");

    // 9 stages: the v0 provenance writers truncate at kMaxGrids=8, so more
    // than 8 stages must be rejected up front (review 1.8).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
                 " \"ns_array\": [5, 11, 17, 23, 29, 35, 41, 47, 53],"
                 " \"niter_array\": [100, 100, 100, 100, 100, 100, 100, 100, 100],"
                 " \"ftol_array\": [1e-12, 1e-12, 1e-12, 1e-12, 1e-12,"
                 " 1e-12, 1e-12, 1e-12, 1e-12]}");
    vr = read_and_validate(scratchPath(), opts);
    printf("  9-stage: validation %s\n", vr.has_value() ? "PASSED" : "rejected");
    CHECK(!vr.has_value() && !findError(vr, "exceed the 8-entry capacity").empty(),
          "malformed: 9-stage schedule rejected");

    // 8 strictly-increasing stages sit exactly at the capacity and stay valid.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
                 " \"ns_array\": [5, 11, 17, 23, 29, 35, 41, 47],"
                 " \"niter_array\": [100, 100, 100, 100, 100, 100, 100, 100],"
                 " \"ftol_array\": [1e-12, 1e-12, 1e-12, 1e-12, 1e-12,"
                 " 1e-12, 1e-12, 1e-12]}");
    vr = read_and_validate(scratchPath(), opts);
    printf("  8-stage: validation %s\n", vr.has_value() ? "PASSED" : "rejected");
    CHECK(vr.has_value(), "malformed: 8-stage schedule still accepted");

    // Schedule length mismatch.
    writeScratch("{\"ns_array\": [5, 11], \"niter_array\": [100, 200],"
                 " \"ftol_array\": [1e-12]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "ftol_array length must match ns_array").empty(),
          "malformed: ftol_array length mismatch rejected");

    // Integer narrowing.
    writeScratch("{\"mpol\": 4294967297}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "out of range").empty(),
          "malformed: integer overflow rejected");

    // Out-of-range boundary mode is skipped with a warning (still succeeds).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"ns_array\": [5],"
                 " \"niter_array\": [100], \"ftol_array\": [1e-12],"
                 " \"am\": [1.0], \"aphi\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0},"
                 "          {\"n\": 0, \"m\": 99, \"value\": 9.9}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(vr.has_value(), "malformed: out-of-range mode still validates");
    if (vr.has_value()) {
        const auto& kept = vr.value().spec().rbc;
        CHECK(kept.size() == 1 && kept[0].value == 1.0,
              "malformed: out-of-range mode skipped, valid entry kept");
    }

    // Negative boundary m skipped (kept entry survives).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": -1, \"value\": 9.9},"
                 "          {\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(vr.has_value(), "malformed: negative m still validates");
    if (vr.has_value()) {
        const auto& kept = vr.value().spec().rbc;
        CHECK(kept.size() == 1 && kept[0].value == 1.0,
              "malformed: negative m skipped, valid entry kept");
    }

    // Empty rbc -> hard error.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbc\": []}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "at least one boundary coefficient").empty(),
          "malformed: empty rbc rejected");

    // Wrong-type auxiliary key.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"am_aux_s\": 5,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "am_aux_s': expected an array").empty(),
          "malformed: scalar am_aux_s rejected");

    // Non-empty asymmetric array is unsupported.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbs\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "asymmetric (lasym) input is not supported").empty(),
          "malformed: rbs content rejected");

    // Non-power_series profile.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"pmass_type\": \"spline\","
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "only \"power_series\"").empty(),
          "malformed: non-power_series profile rejected");

    // Unknown key: compatibility warns (succeeds); strict errors.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"n_theta\": 6,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(vr.has_value(), "malformed: unknown key (compat) still validates");
    if (vr.has_value()) {
        bool warned = false;
        for (const auto& issue : vr.value().warnings().issues()) {
            if (issue.message.find("unknown input key 'n_theta'") != std::string::npos) {
                warned = true;
            }
        }
        CHECK(warned, "malformed: unknown key (compat) recorded as a warning");
    }
    SolverOptions strict;
    strict.strict_schema = true;
    vr = read_and_validate(scratchPath(), strict);
    CHECK(!vr.has_value() && !findError(vr, "unknown input key 'n_theta'").empty(),
          "malformed: unknown key (strict) rejected");

    // Negative niter_array is rejected (a negative int must not wrap to size_t
    // and bypass the >= 1 check).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
                 " \"ns_array\": [5], \"niter_array\": [-1], \"ftol_array\": [1e-12]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "niter_array entries must be >= 1").empty(),
          "malformed: negative niter_array rejected");

    // Present-but-empty raxis_c is rejected (not silently zero-padded).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"raxis_c\": [],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "raxis_c must have exactly ntor+1 entries").empty(),
          "malformed: empty raxis_c rejected");

    // lasym=true.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lasym\": true,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratchPath(), opts);
    CHECK(!vr.has_value() && !findError(vr, "lasym=true").empty(),
          "malformed: lasym=true rejected");
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--emit-golden") {
        emitGoldens(argv[2]);
        return 0;
    }
    testGoldens();
    testModeTable();
    testPrecisionFloor();
    testMalformed();
    remove(scratchPath());
    if (failures == 0) {
        printf("test_host_config: ALL PASS\n");
        return 0;
    }
    printf("test_host_config: %d FAILURES\n", failures);
    return 1;
}
