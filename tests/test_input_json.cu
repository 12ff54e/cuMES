// test_input_json.cu — JSON input parsing: mapping and error paths.
// Exercises src/input_json.cu (initInputParamsFromJson) against the two
// shipped configs (inputs/*.json, run from the cuMES folder) and the
// vmecpp-schema error handling. The expected values below pin the
// mapping: a change here must be a deliberate schema change.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

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

int main() {
    testSolovev();
    testW7x();
    testErrors();
    remove("test_input_json_scratch.json");
    if (failures == 0) {
        printf("test_input_json: ALL PASS\n");
        return 0;
    }
    printf("test_input_json: %d FAILURES\n", failures);
    return 1;
}
