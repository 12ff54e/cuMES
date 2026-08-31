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
#include "cumes/config/json_writer.hpp"
#include "cumes/config/profile_functions.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes_test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <unistd.h>
using namespace cumes::test;

using cumes::parse_problem_spec;
using cumes::PrecisionPolicy;
using cumes::ProblemSpec;
using cumes::read_and_validate;
using cumes::Severity;
using cumes::SolverOptions;
using cumes::ValidationResult;

// Per-test temp directory with RAII cleanup (completion-plan follow-up §5):
// the old PID-based scratch path wrote into the repository root, so an
// interrupted or parallel test left tracked `test_host_config_scratch_*.json`
// debris behind. The directory is created at startup, destroyed at exit, and
// never touches the repo root.
class TempDir {
   public:
    TempDir() {
        char tmpl[] = "/tmp/cumes_host_config_XXXXXX";
        dir_ = mkdtemp(tmpl);
    }
    ~TempDir() {
        if (dir_.empty()) return;
        const std::string cmd = "rm -rf '" + dir_ + "'";
        (void)!system(cmd.c_str());
    }
    const std::string& path() const { return dir_; }
    bool ok() const { return !dir_.empty(); }
    std::string file(const char* name) const { return dir_ + "/" + name; }

   private:
    std::string dir_;
};
static TempDir g_tmp;

static std::string scratch_path() {
    return g_tmp.file("scratch.json");
}

static void write_scratch(const std::string& content) {
    FILE* fp = fopen(scratch_path().c_str(), "w");
    if (!fp) {
        std::cerr << "cannot write scratch file\n";
        exit(1);
    }
    fputs(content.c_str(), fp);
    fclose(fp);
}

static std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

// Return the first error message whose text contains `fragment`, or "" if none.
static std::string find_error(const ValidationResult& vr,
                              const std::string& fragment) {
    if (vr.has_value()) return "";
    for (const auto& issue : vr.error().issues()) {
        if (issue.severity == Severity::ERROR &&
            issue.message.find(fragment) != std::string::npos) {
            return issue.message;
        }
    }
    return "";
}
// ---- golden emission (--emit-golden <dir>) ---------------------------------
static void emit_goldens(const char* dir) {
    SolverOptions opts;  // compatibility (defaults); goldens are config-only
    const std::string prefix = std::string(dir) + "/";
    const char* cases[2] = {"solovev", "w7x"};
    for (const char* name : cases) {
        auto vr =
            read_and_validate(std::string("inputs/") + name + ".json", opts);
        if (!vr.has_value()) {
            std::cerr << format("emit-golden: {} failed validation\n", name);
            exit(1);
        }
        const std::string out = prefix + name + ".normalized.json";
        FILE* fp = fopen(out.c_str(), "w");
        if (!fp) {
            std::cerr << format("cannot write {}\n", out.c_str());
            exit(1);
        }
        fputs(vr.value().normalize_to_json().c_str(), fp);
        fclose(fp);
        std::cout << format("wrote {}\n", out.c_str());
    }
}

// ---- tests ------------------------------------------------------------------

static void test_goldens() {
    SolverOptions opts;
    const char* names[2] = {"solovev", "w7x"};
    for (const char* name : names) {
        auto vr =
            read_and_validate(std::string("inputs/") + name + ".json", opts);
        check(vr.has_value(), std::string(name) + ": validates");
        if (!vr.has_value()) continue;
        const std::string golden = read_file(std::string("tests/fixtures/") +
                                             name + ".normalized.json");
        check(!golden.empty(), std::string(name) + ": golden fixture present");
        check(vr.value().normalize_to_json() == golden,
              std::string(name) + ": normalize_to_json matches golden");
    }
}

static void test_problem_spec_json_round_trip() {
    SolverOptions opts;
    auto original = read_and_validate("inputs/w7x.json", opts);
    check(original.has_value(), "ProblemSpec writer source validates");
    if (!original.has_value()) return;

    const std::string serialized =
        cumes::problem_spec_to_json(original.value().spec());
    auto parsed = parse_problem_spec(serialized, opts);
    check(parsed.report.ok(), "ProblemSpec writer output parses strictly");
    if (!parsed.report.ok()) return;
    auto round_trip = cumes::validate(std::move(parsed.spec), opts);
    check(round_trip.has_value(), "ProblemSpec writer output validates");
    if (!round_trip.has_value()) return;
    check(round_trip.value().normalize_to_json() ==
              original.value().normalize_to_json(),
          "ProblemSpec writer preserves the validated solver input");
}

static void test_mode_table() {
    SolverOptions opts;
    auto vr = read_and_validate("inputs/solovev.json", opts);
    check(vr.has_value(), "solovev validates for mode table");
    if (!vr.has_value()) return;
    const auto& vp = vr.value();
    const auto& table = vp.mode_table();
    check(table.size() == static_cast<std::size_t>(6 * 1),
          "solovev: modes()=6");
    // mode = m*(ntor+1)+n; m=0,n=0 -> even parity, mn_scale 1, xmpq 0.
    check(table[0].m == 0 && table[0].n == 0 &&
              table[0].parity == cumes::ModeParity::EVEN &&
              table[0].mn_scale == 1.0 && table[0].xmpq == 0.0,
          "solovev: (0,0) entry");
    // m=1,n=0 -> odd, mn_scale sqrt(2), xmpq 0, first_surface 1.
    const auto& m1 = table[1];
    check(m1.m == 1 && m1.n == 0 && m1.parity == cumes::ModeParity::ODD &&
              m1.mn_scale == std::sqrt(2.0) && m1.first_surface == 1,
          "solovev: (1,0) entry");
    // m=2,n=0 -> even, mn_scale sqrt(2), xmpq 2.
    const auto& m2 = table[2];
    check(m2.parity == cumes::ModeParity::EVEN &&
              m2.mn_scale == std::sqrt(2.0) && m2.xmpq == 2.0 &&
              m2.first_surface == 1,
          "solovev: (2,0) entry");

    // w7x: physical_n = n*nfp.
    auto vr2 = read_and_validate("inputs/w7x.json", opts);
    check(vr2.has_value(), "w7x validates for mode table");
    if (!vr2.has_value()) return;
    const auto& t2 = vr2.value().mode_table();
    const std::size_t mode11 = 1 * (12 + 1) + 1;  // m=1, n=1
    check(t2[mode11].m == 1 && t2[mode11].n == 1 && t2[mode11].physical_n == 5,
          "w7x: physical_n = n*nfp = 5");
}

static void test_precision_floor() {
    SolverOptions opts;
    opts.precision = PrecisionPolicy::MIXED_FLOAT;
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
        " \"ns_array\": [5], \"niter_array\": [100], \"ftol_array\": [1e-16]}");
    auto vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "floor").empty(),
          "precision floor: float rejects ftol=1e-16");
}

static void test_malformed() {
    SolverOptions opts;
    // Missing file -> throws (JSON syntax/file error).
    bool thrown = false;
    try {
        read_and_validate("no_such_file.json", opts);
    } catch (const std::runtime_error& e) {
        thrown = std::string(e.what()).find("not found") != std::string::npos;
    }
    check(thrown, "malformed: missing file throws");

    // Wrong type for mpol -> collected error.
    write_scratch("{\"mpol\": \"six\"}");
    auto vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "expected an integer").empty(),
          "malformed: wrong type for mpol");

    // Nonzero gamma.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"adiabatic_index\": 0.5,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "gamma").empty(),
          "malformed: nonzero gamma rejected");

    // Empty schedule.
    write_scratch(
        "{\"ns_array\": [], \"niter_array\": [], \"ftol_array\": []}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "at least one stage").empty(),
          "malformed: empty schedule rejected");

    // Non-monotonic schedule.
    write_scratch(
        "{\"ns_array\": [55, 11], \"niter_array\": [100, 200],"
        " \"ftol_array\": [1e-12, 1e-12]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "strictly increasing").empty(),
          "malformed: non-monotonic schedule rejected");

    // Equal consecutive ns: the grid-prolongation precondition needs strictly
    // finer grids (ns_new > ns_old), so a flat pair must be rejected up front
    // instead of dying inside Prolongation::enqueue (review 1.3).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
        " \"ns_array\": [11, 11, 55], \"niter_array\": [100, 100, 100],"
        " \"ftol_array\": [1e-12, 1e-12, 1e-12]}");
    vr = read_and_validate(scratch_path(), opts);
    std::cout << format("  equal-ns {{11,11,55}}: validation {}\n",
                        vr.has_value() ? "PASSED" : "rejected");
    check(!vr.has_value() && !find_error(vr, "strictly increasing").empty(),
          "malformed: equal consecutive ns rejected");

    // 9 stages: no fixed capacity cap remains (the v1 output records active
    // dimensions), so an arbitrary strictly-increasing schedule is accepted.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
        " \"ns_array\": [5, 11, 17, 23, 29, 35, 41, 47, 53],"
        " \"niter_array\": [100, 100, 100, 100, 100, 100, 100, 100, 100],"
        " \"ftol_array\": [1e-12, 1e-12, 1e-12, 1e-12, 1e-12,"
        " 1e-12, 1e-12, 1e-12, 1e-12]}");
    vr = read_and_validate(scratch_path(), opts);
    std::cout << format("  9-stage: validation {}\n",
                        vr.has_value() ? "PASSED" : "rejected");
    check(vr.has_value(),
          "malformed: 9-stage schedule accepted (no capacity cap)");

    // Schedule length mismatch.
    write_scratch(
        "{\"ns_array\": [5, 11], \"niter_array\": [100, 200],"
        " \"ftol_array\": [1e-12]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "ftol_array length must match ns_array").empty(),
          "malformed: ftol_array length mismatch rejected");

    // Integer narrowing.
    write_scratch("{\"mpol\": 4294967297}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "out of range").empty(),
          "malformed: integer overflow rejected");

    // Out-of-range boundary mode is skipped with a warning (still succeeds).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"ns_array\": [5],"
        " \"niter_array\": [100], \"ftol_array\": [1e-12],"
        " \"am\": [1.0], \"aphi\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0},"
        "          {\"n\": 0, \"m\": 99, \"value\": 9.9}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(vr.has_value(), "malformed: out-of-range mode still validates");
    if (vr.has_value()) {
        const auto& kept = vr.value().spec().rbc;
        check(kept.size() == 1 && kept[0].value == 1.0,
              "malformed: out-of-range mode skipped, valid entry kept");
    }

    // Negative boundary m skipped (kept entry survives).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": -1, \"value\": 9.9},"
        "          {\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(vr.has_value(), "malformed: negative m still validates");
    if (vr.has_value()) {
        const auto& kept = vr.value().spec().rbc;
        check(kept.size() == 1 && kept[0].value == 1.0,
              "malformed: negative m skipped, valid entry kept");
    }

    // Empty rbc -> hard error.
    write_scratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbc\": []}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "at least one boundary coefficient").empty(),
          "malformed: empty rbc rejected");

    // Wrong-type auxiliary key.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"am_aux_s\": 5,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "am_aux_s': expected an array").empty(),
          "malformed: scalar am_aux_s rejected");

    // Non-empty asymmetric array is unsupported.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbs\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "asymmetric (lasym) input is not supported")
                   .empty(),
          "malformed: rbs content rejected");

    // Unsupported profile type.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"pmass_type\": \"spline\","
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "unsupported profile type \"spline\"").empty(),
          "malformed: spline profile rejected");
    // "two_power" is not applicable to the iota profile.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"piota_type\": "
        "\"two_power\","
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "\"two_power\" is not applicable").empty(),
          "malformed: two_power piota_type rejected");
    // two_power needs its three coefficients (c0, c1, c2).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0, 5.0],"
        " \"pmass_type\": \"two_power\","
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "two_power profile needs at least 3").empty(),
          "malformed: short two_power am rejected");

    // Unknown key: strict is the DEFAULT now (completion plan step 2.1),
    // so the compat path must opt out explicitly.
    SolverOptions compat = opts;
    compat.strict_schema = false;
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"n_theta\": 6,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), compat);
    check(vr.has_value(), "malformed: unknown key (compat) still validates");
    if (vr.has_value()) {
        bool warned = false;
        for (const auto& issue : vr.value().warnings().issues()) {
            if (issue.message.find("unknown input key 'n_theta'") !=
                std::string::npos) {
                warned = true;
            }
        }
        check(warned, "malformed: unknown key (compat) recorded as a warning");
    }
    SolverOptions strict;
    strict.strict_schema = true;
    vr = read_and_validate(scratch_path(), strict);
    check(!vr.has_value() &&
              !find_error(vr, "unknown input key 'n_theta'").empty(),
          "malformed: unknown key (strict) rejected");

    // Negative niter_array is rejected (a negative int must not wrap to size_t
    // and bypass the >= 1 check).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}],"
        " \"ns_array\": [5], \"niter_array\": [-1], \"ftol_array\": [1e-12]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "niter_array entries must be >= 1").empty(),
          "malformed: negative niter_array rejected");

    // Present-but-empty raxis_c is rejected (not silently zero-padded).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"raxis_c\": [],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(
        !vr.has_value() &&
            !find_error(vr, "raxis_c must have exactly ntor+1 entries").empty(),
        "malformed: empty raxis_c rejected");

    // lasym=true.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lasym\": true,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "lasym=true").empty(),
          "malformed: lasym=true rejected");

    // ---- profile-normalization scalars (completion plan step 1.1) ----
    // T_edge = torflux(1) normalizes the toroidal flux; C_edge = J_C(1)
    // normalizes the prescribed current. Non-finite / zero / ill-scaled values
    // must fail validation BEFORE any CUDA allocation.

    // Non-finite T(1): the JSON parser cannot express inf/NaN tokens, so the
    // non-finite normalization cases drive cumes::validate() directly with an
    // in-memory ProblemSpec (the same gate read_and_validate reaches).
    {
        ProblemSpec spec;
        spec.mpol = 2;
        spec.ntor = 0;
        spec.nfp = 1;
        spec.mass.coefficients = {1.0};
        spec.toroidal_flux.coefficients = {INFINITY};
        spec.rbc = {{1, 0, 1.0}};
        spec.zbs = {{1, 0, 0.5}};
        auto vr2 = cumes::validate(spec, opts);
        check(
            !vr2.has_value() &&
                !find_error(vr2, "non-finite at the edge").empty(),
            "malformed: non-finite toroidal-flux edge normalization rejected");
    }

    // Non-finite phiedge.
    {
        ProblemSpec spec;
        spec.mpol = 2;
        spec.ntor = 0;
        spec.nfp = 1;
        spec.mass.coefficients = {1.0};
        spec.toroidal_flux.coefficients = {1.0};
        spec.physical.phiedge = INFINITY;
        spec.rbc = {{1, 0, 1.0}};
        spec.zbs = {{1, 0, 0.5}};
        auto vr2 = cumes::validate(spec, opts);
        check(!vr2.has_value() &&
                  !find_error(vr2, "phiedge must be finite").empty(),
              "malformed: non-finite phiedge rejected");
    }

    // Non-finite prescribed-current edge integral.
    {
        ProblemSpec spec;
        spec.mpol = 2;
        spec.ntor = 0;
        spec.nfp = 1;
        spec.current_model = cumes::CurrentModel::PRESCRIBED_CURRENT;
        spec.physical.curtor = 1.0;
        spec.mass.coefficients = {1.0};
        spec.toroidal_flux.coefficients = {1.0};
        spec.current.coefficients = {INFINITY};
        spec.rbc = {{1, 0, 1.0}};
        spec.zbs = {{1, 0, 0.5}};
        auto vr2 = cumes::validate(spec, opts);
        check(
            !vr2.has_value() &&
                !find_error(vr2, "non-finite at the edge").empty(),
            "malformed: non-finite prescribed-current edge integral rejected");
    }

    // Zero T(1) (an explicit empty aphi array: the empty series integrates
    // to 0; a MISSING aphi gets the vmecpp default {1.0}, so it stays valid).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"aphi\": [],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "zero at the edge").empty(),
          "malformed: zero toroidal-flux edge normalization rejected");

    // Ill-scaled T(1) (below the 1e-30 floor).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"aphi\": [1e-31],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "ill-scaled at the edge").empty(),
          "malformed: ill-scaled toroidal-flux edge normalization rejected");

    // Prescribed current: C_edge = 0 (no ac at all).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"aphi\": [1.0],"
        " \"ncurr\": 1, \"curtor\": 1.0,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "integrates to zero at the edge").empty(),
          "malformed: zero prescribed-current edge integral rejected");

    // A zero requested total current needs no profile normalization: this is
    // the vacuum prescribed-current form used by the Landreman-Paul QA/QH
    // configurations.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [0.0], \"aphi\": [1.0],"
        " \"ncurr\": 1, \"curtor\": 0.0,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]} ");
    vr = read_and_validate(scratch_path(), opts);
    check(vr.has_value(),
          "vacuum prescribed-current input validates without ac");

    // A healthy prescribed-current fixture still validates (positive control).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"aphi\": [1.0],"
        " \"ncurr\": 1, \"curtor\": 1.0, \"ac\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = read_and_validate(scratch_path(), opts);
    check(vr.has_value(),
          "healthy prescribed-current normalization still validates");
}

// two_power evaluator checks. The closed-form linear case (c1 = c2 = 1) is
// integrated EXACTLY by the 10-point Gauss-Legendre quadrature (exact for
// polynomials of degree ≤ 19), and the axis/edge endpoints are exact for any
// exponent: p(0) = c0·μ0·pres_scale, p(1) = 0, I(0) = 0.
static void test_two_power_evaluators() {
    const double mu0 = DeviceParams<double>::MU_0;

    ProblemSpec sp;
    sp.mass.type = cumes::ProfileType::TWO_POWER;
    sp.mass.coefficients = {1.0, 1.0, 1.0};
    sp.physical.pres_scale = 2.0;
    const double x = 0.375;
    const double want = mu0 * 2.0 * (1.0 - x);
    const double got = cumes::eval_mass_profile<double>(sp, x);
    check(std::fabs(got - want) <= 1e-15 * std::fabs(want),
          "two_power mass profile matches the closed form");

    ProblemSpec sp2;
    sp2.mass.type = cumes::ProfileType::TWO_POWER;
    sp2.mass.coefficients = {2.0, 3.0, 4.0};
    check(cumes::eval_mass_profile<double>(sp2, 0.0) == mu0 * 2.0,
          "two_power mass at the axis is c0·μ0·pres_scale");
    check(cumes::eval_mass_profile<double>(sp2, 1.0) == 0.0,
          "two_power mass at the edge is zero");

    ProblemSpec sp3;
    sp3.current.type = cumes::ProfileType::TWO_POWER;
    sp3.current.coefficients = {1.0, 1.0, 1.0};
    const double want_i = x - 0.5 * x * x;
    const double got_i = cumes::eval_curr_profile<double>(sp3, x);
    check(std::fabs(got_i - want_i) <= 1e-15 * std::fabs(want_i),
          "two_power current profile integrates the linear case exactly");
    check(cumes::eval_curr_profile<double>(sp3, 0.0) == 0.0,
          "two_power current at the axis is zero");
}

static void test_in_memory_json() {
    SolverOptions options;
    const auto parsed = parse_problem_spec(
        R"({"mpol": 3, "ntor": 1, "nfp": 5,
             "rbc": [{"m": 1, "n": 0, "value": 1.25}],
             "zbs": [{"m": 1, "n": 0, "value": 0.4}]})",
        options);
    check(parsed.report.ok(), "in-memory JSON mapping succeeds");
    check(
        parsed.spec.mpol == 3 && parsed.spec.ntor == 1 && parsed.spec.nfp == 5,
        "in-memory JSON maps scalar resolution fields");
    check(parsed.spec.rbc.size() == 1 && parsed.spec.rbc.front().value == 1.25,
          "in-memory JSON maps boundary harmonics");
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--emit-golden") {
        emit_goldens(argv[2]);
        return 0;
    }
    test_goldens();
    test_problem_spec_json_round_trip();
    test_mode_table();
    test_precision_floor();
    test_malformed();
    test_two_power_evaluators();
    test_in_memory_json();
    // No explicit scratch removal: the TempDir RAII destructor cleans the
    // per-test directory even on an interrupted or failing run.
    return summary();
}
