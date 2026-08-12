// test_input_json.cu — JSON input parsing: mapping and error paths.
// Exercises src/input_json.cu (initInputParamsFromJson) against the two
// shipped configs (inputs/*.json, run from the cuMES folder) and the
// vmecpp-schema error handling. The expected values below pin the
// mapping: a change here must be a deliberate schema change.
//
// The negative tests pin the containment-series validation: nonzero gamma,
// negative/out-of-range boundary m, empty/oversized/mismatched/non-monotonic
// multigrid schedules, integer narrowing, wrong-type auxiliary/asymmetric
// keys, unsupported physics (lasym/lfreeb/spline profiles), and the
// unknown-key warning. A change to any of these must be a deliberate schema
// decision, not an accidental loosening.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

#include "input_json.h"

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

// Expect expr to throw std::runtime_error whose message contains `fragment`.
#define CHECK_THROWS(expr, fragment, msg)                                    \
    do {                                                                     \
        bool thrown = false;                                                 \
        try {                                                                \
            (void)(expr);                                                    \
        } catch (const std::runtime_error& e) {                              \
            thrown = std::string(e.what()).find(fragment) != std::string::npos; \
        }                                                                    \
        CHECK(thrown, msg);                                                  \
    } while (0)

static void writeScratch(const std::string& content) {
    FILE* fp = fopen("test_input_json_scratch.json", "w");
    if (!fp) { fprintf(stderr, "cannot write scratch file\n"); exit(1); }
    fputs(content.c_str(), fp);
    fclose(fp);
}

// Return the stderr written while running fn() (for the unknown-key warning).
template <typename Fn>
static std::string captureStderr(Fn fn) {
    const char* path = "test_input_json_stderr.txt";
    fflush(stderr);
    int saved = dup(STDERR_FILENO);
    FILE* tmp = fopen(path, "w");
    if (!tmp) { fprintf(stderr, "cannot open stderr capture\n"); exit(1); }
    int fd = fileno(tmp);
    dup2(fd, STDERR_FILENO);
    fclose(tmp);
    fn();
    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot read stderr capture\n"); exit(1); }
    std::string out;
    char buf[1024];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, fp)) > 0) out.append(buf, n);
    fclose(fp);
    remove(path);
    return out;
}

static void testSolovev() {
    InputParams p = initInputParams("inputs/solovev.json");
    CHECK(p.mpol == 6 && p.ntor == 0 && p.nfp == 1, "solovev: mpol/ntor/nfp");
    CHECK(p.ntheta == 18 && p.nzeta == 1, "solovev: resolution defaults");
    CHECK(p.ncurr == 0 && p.delt == 0.9 && p.phiedge == 1.0, "solovev: ncurr/delt/phiedge");
    CHECK(p.n_grids == 3 && p.ns_array[0] == 5 && p.ns_array[1] == 11 &&
              p.ns_array[2] == 55, "solovev: ns_array 5/11/55");
    CHECK(p.niter_array[0] == 1000 && p.niter_array[1] == 2000 &&
              p.niter_array[2] == 2000, "solovev: niter_array");
    CHECK(p.ftol_array[0] == 1e-16 && p.ftol_array[2] == 1e-16, "solovev: ftol_array");
    // stage-0 invariant: GridParams and other tests read these directly
    CHECK(p.ns == 5 && p.max_iter == 1000 && p.ftol == 1e-16, "solovev: stage-0 invariant");
    CHECK(p.am_n == 2 && p.am[0] == 0.125 && p.am[1] == -0.125, "solovev: am");
    CHECK(p.ac_n == 0 && p.ai_n == 1 && p.ai[0] == 1.0, "solovev: ac/ai");
    CHECK(p.aphi_n == 1 && p.aphi[0] == 1.0, "solovev: aphi default {1.0}");
    CHECK(p.raxis_n == 1 && p.raxis_c[0] == 4.0, "solovev: raxis");
    CHECK(p.rbc_n == 3 && p.zbs_n == 3, "solovev: boundary counts");
    // folded product basis (foldBoundary)
    CHECK(p.rbcc[0][0] == 3.999 && p.rbcc[1][0] == 1.026 && p.rbcc[2][0] == -0.068,
          "solovev: folded rbcc");
    CHECK(p.zbsc[1][0] == 1.580 && p.zbsc[2][0] == 0.010, "solovev: folded zbsc");
}

static void testW7x() {
    InputParams p = initInputParams("inputs/w7x.json");
    CHECK(p.mpol == 12 && p.ntor == 12 && p.nfp == 5, "w7x: mpol/ntor/nfp");
    CHECK(p.ntheta == 30 && p.nzeta == 36, "w7x: resolution defaults");
    CHECK(p.ncurr == 1 && p.delt == 1.0 && p.phiedge == -1.74, "w7x: ncurr/delt/phiedge");
    CHECK(p.curtor == 5000.0 && p.bloat == 1.0 && p.spres_ped == 1.0 && p.tcon0 == 1.0,
          "w7x: current/profile scalars");
    CHECK(p.n_grids == 3 && p.ns_array[2] == 99, "w7x: ns_array 33/66/99");
    CHECK(p.niter_array[2] == 5000 && p.ftol_array[2] == 1e-12, "w7x: multigrid tails");
    CHECK(p.ns == 33 && p.max_iter == 3000 && p.ftol == 1e-12, "w7x: stage-0 invariant");
    CHECK(p.am_n == 2 && p.am[0] == 166000.0, "w7x: am");
    CHECK(p.ac_n == 2 && p.ac[0] == 0.0 && p.ac[1] == 1.0, "w7x: ac");
    CHECK(p.ai_n == 0, "w7x: no ai");
    CHECK(p.raxis_n == 13 && p.raxis_c[0] == 5.6343 && p.raxis_c[1] == 0.35209 &&
              p.zaxis_s[1] == -0.29578, "w7x: raxis/zaxis");
    CHECK(p.rbc_n == 85 && p.zbs_n == 84, "w7x: boundary counts (85 rbc, 84 zbs)");
    // folded product basis: a few asymmetric-mode checks (n can be negative)
    CHECK(p.rbcc[1][0] == 0.49093, "w7x: rbcc[1][0]");
    CHECK(std::fabs(p.rbss[1][1] - (-0.25107 - 0.033555)) < 1e-12, "w7x: rbss[1][1] fold");
    CHECK(p.zbsc[1][0] == 0.61965, "w7x: zbsc[1][0]");
    CHECK(std::fabs(p.zbcs[1][1] - (0.036669 - 0.17897)) < 1e-12, "w7x: zbcs[1][1] fold");
}

static void testErrors() {
    CHECK_THROWS(initInputParamsFromJson("no_such_file.json"), "not found",
                 "error: missing file");
    writeScratch("{\"mpol\": \"six\"}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "expected an integer", "error: wrong type for mpol");
    // (lasym check runs after the boundary-required check, so include one)
    writeScratch("{\"lasym\": true, \"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "lasym=true", "error: lasym rejected");
    writeScratch("{\"ns_array\": [5, 11], \"niter_array\": [1000, 2000, 2000],"
                 " \"ftol_array\": [1e-16, 1e-16, 1e-16]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "niter_array length must match ns_array", "error: multigrid length mismatch");
    // out-of-range boundary mode is warned about and skipped (vmecpp semantics)
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"ns_array\": [5],"
                 " \"niter_array\": [100], \"ftol_array\": [1e-12],"
                 " \"am\": [1.0], \"aphi\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0},"
                 "          {\"n\": 0, \"m\": 99, \"value\": 9.9}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    InputParams p = initInputParams("test_input_json_scratch.json");
    CHECK(p.rbc_n == 1 && p.rbc[0].value == 1.0, "error: out-of-range mode skipped");
    // missing boundary is a hard error
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbc\": []}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "rbc: at least one boundary coefficient is required",
                 "error: empty rbc rejected");
    // minimal document without multigrid arrays -> single stage, defaults
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    p = initInputParams("test_input_json_scratch.json");
    CHECK(p.n_grids == 1 && p.ns == p.ns_array[0] && p.aphi_n == 1 && p.aphi[0] == 1.0,
          "minimal doc: single stage + aphi default");
}

// ---- containment-series validation negatives (cuMES-issues.md fixes) ----
// Each case pins an accepted validation path: the offending input must be
// rejected before any allocation, with a message naming the cause.

static void testNegative() {
    // Nonzero gamma / adiabatic_index is rejected (input_json.cu:217).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"adiabatic_index\": 0.5,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "gamma", "neg: nonzero gamma (adiabatic_index) rejected");
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"gamma\": 1.5,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "gamma", "neg: nonzero gamma (alias) rejected");

    // Negative boundary m is skipped with a warning (input_json.cu:151),
    // matching vmecpp's ignore-and-continue for out-of-range modes. The
    // folded tables stay in-bounds.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": -1, \"value\": 9.9},"
                 "          {\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    {
        InputParams p = initInputParams("test_input_json_scratch.json");
        CHECK(p.rbc_n == 1 && p.rbc[0].value == 1.0,
              "neg: negative boundary m skipped, valid entry kept");
    }
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 7, \"value\": 9.9},"
                 "          {\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    {
        InputParams p = initInputParams("test_input_json_scratch.json");
        CHECK(p.rbc_n == 1 && p.rbc[0].value == 1.0,
              "neg: m >= mpol boundary skipped");
    }

    // Empty multigrid schedule is rejected (input_json.cu:270): a zero-stage
    // run would save/print a null state.
    writeScratch("{\"ns_array\": [], \"niter_array\": [], \"ftol_array\": []}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "at least one stage", "neg: empty ns_array rejected");
    // Oversized schedule rejected by the kMaxGrids capacity (readNumberArray).
    writeScratch("{\"ns_array\": [5,6,7,8,9,10,11,12,13],"
                 " \"niter_array\": [1,1,1,1,1,1,1,1,1],"
                 " \"ftol_array\": [1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "exceed the 8-entry capacity", "neg: oversized ns_array rejected");
    // ftol_array length mismatch (only niter length was covered before).
    writeScratch("{\"ns_array\": [5, 11], \"niter_array\": [100, 200],"
                 " \"ftol_array\": [1e-12]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "ftol_array length must match ns_array",
                 "neg: ftol_array length mismatch rejected");
    // Non-monotonic schedule.
    writeScratch("{\"ns_array\": [55, 11], \"niter_array\": [100, 200],"
                 " \"ftol_array\": [1e-12, 1e-12]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "monotonically non-decreasing", "neg: non-monotonic ns rejected");

    // Integer narrowing: a huge literal must not silently wrap into a
    // valid-looking resolution (input_json.cu:67).
    writeScratch("{\"mpol\": 4294967297}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "out of range", "neg: integer overflow rejected");

    // Wrong-type auxiliary/asymmetric keys are hard errors (input_json.cu:317).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"am_aux_s\": 5,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "am_aux_s': expected an array", "neg: scalar am_aux_s rejected");
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"raxis_s\": 5,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "raxis_s': expected an array", "neg: scalar raxis_s rejected");
    // Non-empty asymmetric array is unsupported physics.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbs\": [1.0],"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "asymmetric (lasym) input is not supported", "neg: rbs content rejected");

    // Unsupported physics keys: lasym, lfreeb, non-power_series profiles.
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lasym\": true,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "lasym=true", "neg: lasym=true rejected");
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lfreeb\": true,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "free-boundary", "neg: lfreeb=true rejected");
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
                 " \"pmass_type\": \"spline\","
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    CHECK_THROWS(initInputParamsFromJson("test_input_json_scratch.json"),
                 "only \"power_series\"", "neg: non-power_series profile rejected");

    // A typo'd key warns to stderr (not silently ignored) but does not fail
    // the parse (unknown keys outside the supported/known-ignored sets warn).
    writeScratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"n_theta\": 6,"
                 " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
                 " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    {
        std::string err = captureStderr([&]() {
            InputParams p = initInputParams("test_input_json_scratch.json");
            CHECK(p.mpol == 2, "neg: unknown key parse succeeds");
        });
        CHECK(err.find("unknown input key 'n_theta'") != std::string::npos,
              "neg: unknown key warned to stderr");
    }
}

int main() {
    testSolovev();
    testW7x();
    testErrors();
    testNegative();
    remove("test_input_json_scratch.json");
    if (failures == 0) {
        printf("test_input_json: ALL PASS\n");
        return 0;
    }
    printf("test_input_json: %d FAILURES\n", failures);
    return 1;
}
