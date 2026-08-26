// test_input_json.cu — JSON input mapping and error paths for the Phase 2
// host model (read_and_validate). Exercises the two shipped configs
// (inputs/*.json, run from the cuMES folder) and the vmecpp-schema error
// handling. The expected values below pin the mapping; a change here must be a
// deliberate schema change.
//
// The negative tests pin the containment-series validation: nonzero gamma,
// negative/out-of-range boundary m, empty/oversized/mismatched/non-monotonic
// multigrid schedules, integer narrowing, wrong-type auxiliary/asymmetric
// keys, unsupported physics (lasym/lfreeb/spline profiles), and the
// unknown-key warning. A change to any of these must be a deliberate schema
// decision, not an accidental loosening.
#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes_test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>
using namespace cumes::test;

// Unique per-process scratch file so parallel CTest runs cannot collide on
// the same path (the old fixed name broke `ctest -j`).
static const char* scratch_path() {
    static const std::string buf =
        format("test_input_json_scratch_{}.json", (int)getpid());
    return buf.c_str();
}

static void write_scratch(const std::string& content) {
    FILE* fp = fopen(scratch_path(), "w");
    if (!fp) {
        std::cerr << "cannot write scratch file\n";
        exit(1);
    }
    fputs(content.c_str(), fp);
    fclose(fp);
}

// Validate a JSON fixture; exit if it fails (the value tests below assume it
// succeeds — a mapping regression should fail loudly, not cascade).
static cumes::ValidatedProblem load(const char* path) {
    cumes::SolverOptions opts;
    auto vr = cumes::read_and_validate(path, opts);
    if (!vr.has_value()) {
        std::cerr << format("load({}) failed validation\n", path);
        exit(1);
    }
    return std::move(vr.value());
}

// Return the first error message whose text contains `fragment`, or "" if none.
static std::string find_error(const cumes::ValidationResult& vr,
                              const std::string& fragment) {
    if (vr.has_value()) return "";
    for (const auto& issue : vr.error().issues()) {
        if (issue.severity == cumes::Severity::ERROR &&
            issue.message.find(fragment) != std::string::npos) {
            return issue.message;
        }
    }
    return "";
}

// Folded-boundary index: mode = m*(ntor+1)+n.
static double fold(const cumes::FoldedBoundary& b,
                   int ntor,
                   int m,
                   int n,
                   int comp) {
    const std::size_t mode = static_cast<std::size_t>(m) * (ntor + 1) + n;
    switch (comp) {
        case 0:
            return b.rbcc[mode];
        case 1:
            return b.rbss[mode];
        case 2:
            return b.zbsc[mode];
        default:
            return b.zbcs[mode];
    }
}

static void test_solovev() {
    cumes::ValidatedProblem vp = load("inputs/solovev.json");
    const cumes::ProblemSpec& s = vp.spec();
    const cumes::FoldedBoundary& b = vp.boundary();
    check(s.mpol == 6 && s.ntor == 0 && s.nfp == 1, "solovev: mpol/ntor/nfp");
    check(s.angular.ntheta == 18 && s.angular.nzeta == 1,
          "solovev: resolution defaults");
    check(s.current_model == cumes::CurrentModel::FIXED_IOTA && s.delt == 0.9 &&
              s.physical.phiedge == 1.0,
          "solovev: ncurr/delt/phiedge");
    check(s.stages.size() == 3 && s.stages[0].radial_surfaces == 5 &&
              s.stages[1].radial_surfaces == 11 &&
              s.stages[2].radial_surfaces == 55,
          "solovev: ns_array 5/11/55");
    check(s.stages[0].max_iterations == 1000 &&
              s.stages[1].max_iterations == 2000 &&
              s.stages[2].max_iterations == 2000,
          "solovev: niter_array");
    check(s.stages[0].tolerance == 1e-16 && s.stages[2].tolerance == 1e-16,
          "solovev: ftol_array");
    check(s.mass.coefficients.size() == 2 && s.mass.coefficients[0] == 0.125 &&
              s.mass.coefficients[1] == -0.125,
          "solovev: am");
    check(s.current.coefficients.empty() && s.iota.coefficients.size() == 1 &&
              s.iota.coefficients[0] == 1.0,
          "solovev: ac/ai");
    check(s.toroidal_flux.coefficients.size() == 1 &&
              s.toroidal_flux.coefficients[0] == 1.0,
          "solovev: aphi default {1.0}");
    check(s.raxis_c.size() == 1 && s.raxis_c[0] == 4.0, "solovev: raxis");
    check(s.rbc.size() == 3 && s.zbs.size() == 3, "solovev: boundary counts");
    // folded product basis
    check(fold(b, s.ntor, 0, 0, 0) == 3.999 &&
              fold(b, s.ntor, 1, 0, 0) == 1.026 &&
              fold(b, s.ntor, 2, 0, 0) == -0.068,
          "solovev: folded rbcc");
    check(
        fold(b, s.ntor, 1, 0, 2) == 1.580 && fold(b, s.ntor, 2, 0, 2) == 0.010,
        "solovev: folded zbsc");
}

static void test_w7x() {
    cumes::ValidatedProblem vp = load("inputs/w7x.json");
    const cumes::ProblemSpec& s = vp.spec();
    const cumes::FoldedBoundary& b = vp.boundary();
    check(s.mpol == 12 && s.ntor == 12 && s.nfp == 5, "w7x: mpol/ntor/nfp");
    check(s.angular.ntheta == 30 && s.angular.nzeta == 36,
          "w7x: resolution defaults");
    check(s.current_model == cumes::CurrentModel::PRESCRIBED_CURRENT &&
              s.delt == 1.0 && s.physical.phiedge == -1.74,
          "w7x: ncurr/delt/phiedge");
    check(s.physical.curtor == 5000.0 && s.physical.bloat == 1.0 &&
              s.physical.spres_ped == 1.0 && s.physical.tcon0 == 1.0,
          "w7x: current/profile scalars");
    check(s.stages.size() == 3 && s.stages[2].radial_surfaces == 99,
          "w7x: ns_array 33/66/99");
    check(s.stages[2].max_iterations == 5000 && s.stages[2].tolerance == 1e-12,
          "w7x: multigrid tails");
    check(s.mass.coefficients.size() == 2 && s.mass.coefficients[0] == 166000.0,
          "w7x: am");
    check(s.current.coefficients.size() == 2 &&
              s.current.coefficients[0] == 0.0 &&
              s.current.coefficients[1] == 1.0,
          "w7x: ac");
    check(s.iota.coefficients.empty(), "w7x: no ai");
    check(s.raxis_c.size() == 13 && s.raxis_c[0] == 5.6343 &&
              s.raxis_c[1] == 0.35209 && s.zaxis_s[1] == -0.29578,
          "w7x: raxis/zaxis");
    check(s.rbc.size() == 85 && s.zbs.size() == 84,
          "w7x: boundary counts (85 rbc, 84 zbs)");
    // folded product basis: a few asymmetric-mode checks (n can be negative)
    check(fold(b, s.ntor, 1, 0, 0) == 0.49093, "w7x: rbcc[1][0]");
    check(std::fabs(fold(b, s.ntor, 1, 1, 1) - (-0.25107 - 0.033555)) < 1e-12,
          "w7x: rbss[1][1] fold");
    check(fold(b, s.ntor, 1, 0, 2) == 0.61965, "w7x: zbsc[1][0]");
    check(std::fabs(fold(b, s.ntor, 1, 1, 3) - (0.036669 - 0.17897)) < 1e-12,
          "w7x: zbcs[1][1] fold");
}

static void test_errors() {
    cumes::SolverOptions opts;
    bool thrown = false;
    try {
        cumes::read_and_validate("no_such_file.json", opts);
    } catch (const std::runtime_error& e) {
        thrown = std::string(e.what()).find("not found") != std::string::npos;
    }
    check(thrown, "error: missing file");
    write_scratch("{\"mpol\": \"six\"}");
    auto vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "expected an integer").empty(),
          "error: wrong type for mpol");
    // (lasym check runs after the boundary-required check, so include one)
    write_scratch(
        "{\"lasym\": true, \"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "lasym=true").empty(),
          "error: lasym rejected");
    write_scratch(
        "{\"ns_array\": [5, 11], \"niter_array\": [1000, 2000, 2000],"
        " \"ftol_array\": [1e-16, 1e-16, 1e-16]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "niter_array length must match ns_array").empty(),
          "error: multigrid length mismatch");
    // out-of-range boundary mode is warned about and skipped (vmecpp semantics)
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"ns_array\": [5],"
        " \"niter_array\": [100], \"ftol_array\": [1e-12],"
        " \"am\": [1.0], \"aphi\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0},"
        "          {\"n\": 0, \"m\": 99, \"value\": 9.9}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(vr.has_value() && vr.value().spec().rbc.size() == 1 &&
              vr.value().spec().rbc[0].value == 1.0,
          "error: out-of-range mode skipped");
    // missing boundary is a hard error
    write_scratch("{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbc\": []}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "at least one boundary coefficient").empty(),
          "error: empty rbc rejected");
    // minimal document without multigrid arrays -> single stage, defaults
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(vr.has_value() && vr.value().spec().stages.size() == 1 &&
              vr.value().spec().toroidal_flux.coefficients.size() == 1 &&
              vr.value().spec().toroidal_flux.coefficients[0] == 1.0,
          "minimal doc: single stage + aphi default");
}

// ---- containment-series validation negatives (cuMES-issues.md fixes) ----
// Each case pins an accepted validation path: the offending input must be
// rejected before any allocation, with a message naming the cause.
static void test_negative() {
    cumes::SolverOptions opts;
    // Nonzero gamma / adiabatic_index is rejected.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"adiabatic_index\": 0.5,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    auto vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "gamma").empty(),
          "neg: nonzero gamma (adiabatic_index) rejected");
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"gamma\": 1.5,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "gamma").empty(),
          "neg: nonzero gamma (alias) rejected");

    // Negative boundary m is skipped with a warning, matching vmecpp.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": -1, \"value\": 9.9},"
        "          {\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(vr.has_value() && vr.value().spec().rbc.size() == 1 &&
              vr.value().spec().rbc[0].value == 1.0,
          "neg: negative boundary m skipped, valid entry kept");
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 7, \"value\": 9.9},"
        "          {\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(vr.has_value() && vr.value().spec().rbc.size() == 1 &&
              vr.value().spec().rbc[0].value == 1.0,
          "neg: m >= mpol boundary skipped");

    // Empty multigrid schedule is rejected.
    write_scratch(
        "{\"ns_array\": [], \"niter_array\": [], \"ftol_array\": []}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "at least one stage").empty(),
          "neg: empty ns_array rejected");
    // ftol_array length mismatch.
    write_scratch(
        "{\"ns_array\": [5, 11], \"niter_array\": [100, 200],"
        " \"ftol_array\": [1e-12]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "ftol_array length must match ns_array").empty(),
          "neg: ftol_array length mismatch rejected");
    // Non-monotonic schedule.
    write_scratch(
        "{\"ns_array\": [55, 11], \"niter_array\": [100, 200],"
        " \"ftol_array\": [1e-12, 1e-12]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "monotonically non-decreasing").empty(),
          "neg: non-monotonic ns rejected");

    // Integer narrowing: a huge literal must not silently wrap.
    write_scratch("{\"mpol\": 4294967297}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "out of range").empty(),
          "neg: integer overflow rejected");

    // Wrong-type auxiliary/asymmetric keys are hard errors.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"am_aux_s\": 5,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "am_aux_s': expected an array").empty(),
          "neg: scalar am_aux_s rejected");
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"raxis_s\": 5,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "raxis_s': expected an array").empty(),
          "neg: scalar raxis_s rejected");
    // Non-empty asymmetric array is unsupported physics.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"rbs\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "asymmetric (lasym) input is not supported")
                   .empty(),
          "neg: rbs content rejected");

    // Unsupported physics keys: lasym and unsupported/ill-formed profile
    // types. Free-boundary input is supported when its source is complete.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lasym\": true,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "lasym=true").empty(),
          "neg: lasym=true rejected");
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lfreeb\": true,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "lfreeb=true requires either mgrid_file").empty(),
          "neg: free boundary without external-field source rejected");
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lfreeb\": true,"
        " \"mgrid_file\": \"mgrid.nc\","
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(
        !vr.has_value() &&
            !find_error(vr, "lfreeb=true requires a non-empty extcur").empty(),
        "neg: free boundary without extcur rejected");
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"lfreeb\": true,"
        " \"mgrid_file\": \"mgrid.nc\", \"extcur\": [1.0],"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
#ifdef CUMES_VACUUM_FIELD_DISABLED
    check(!vr.has_value() &&
              !find_error(vr, "requires optional vacuum-field support").empty(),
          "neg: free boundary rejected without optional dependency");
#else
    check(vr.has_value() && vr.value().spec().free_boundary.lfreeb,
          "pos: free boundary with mgrid accepted");
#endif

    const char* embedded_makegrid =
        "\"normalize_by_currents\":false,"
        "\"assume_stellarator_symmetry\":false,"
        "\"number_of_field_periods\":1,"
        "\"r_grid_minimum\":2.0,\"r_grid_maximum\":6.0,"
        "\"number_of_r_grid_points\":123,"
        "\"z_grid_minimum\":-2.0,\"z_grid_maximum\":2.0,"
        "\"number_of_z_grid_points\":101,"
        "\"number_of_phi_grid_points\":1";
    write_scratch(
        std::string("{\"mpol\":2,\"ntor\":0,\"nfp\":1,\"am\":[1.0],") +
        "\"lfreeb\":true,\"coils_file\":\"coils.test\"," +
        "\"makegrid_parameters_file\":\"ignored.json\"," +
        "\"makegrid_parameters\":{" + embedded_makegrid + "}," +
        "\"extcur\":[1.0]," + "\"rbc\":[{\"n\":0,\"m\":1,\"value\":1.0}]," +
        "\"zbs\":[{\"n\":0,\"m\":1,\"value\":0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
#ifndef CUMES_VACUUM_FIELD_DISABLED
    bool precedence_warned = false;
    if (vr.has_value()) {
        for (const auto& issue : vr.value().warnings().issues()) {
            if (issue.message.find("using embedded makegrid_parameters") !=
                std::string::npos) {
                precedence_warned = true;
            }
        }
    }
    check(vr.has_value() && precedence_warned &&
              vr.value()
                      .spec()
                      .free_boundary.embedded_makegrid_parameters
                      ->number_of_r_grid_points == 123,
          "pos: embedded Makegrid overrides file with warning");
#endif
    write_scratch(
        "{\"mpol\":2,\"ntor\":0,\"am\":[1.0],\"lfreeb\":true,"
        "\"coils_file\":\"coils.test\","
        "\"makegrid_parameters\":{\"normalize_by_currents\":false},"
        "\"extcur\":[1.0],"
        "\"rbc\":[{\"n\":0,\"m\":1,\"value\":1.0}],"
        "\"zbs\":[{\"n\":0,\"m\":1,\"value\":0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() && !find_error(vr, "missing required field").empty(),
          "neg: incomplete embedded Makegrid parameters rejected");
    write_scratch(
        "{\"mpol\":2,\"ntor\":0,\"am\":[1.0],"
        "\"makegrid_paramters\":{},"
        "\"rbc\":[{\"n\":0,\"m\":1,\"value\":1.0}],"
        "\"zbs\":[{\"n\":0,\"m\":1,\"value\":0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "unknown input key 'makegrid_paramters'").empty(),
          "neg: misspelled Makegrid key is not accepted");
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"pmass_type\": \"spline\","
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "unsupported profile type \"spline\"").empty(),
          "neg: spline pmass_type rejected");
    // "two_power" is not applicable to the iota profile.
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
        " \"piota_type\": \"two_power\","
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "\"two_power\" is not applicable").empty(),
          "neg: two_power piota_type rejected");
    // two_power needs its three coefficients (c0, c1, c2).
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0, 5.0],"
        " \"pmass_type\": \"two_power\","
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), opts);
    check(!vr.has_value() &&
              !find_error(vr, "two_power profile needs at least 3").empty(),
          "neg: short two_power am rejected");

    // A typo'd key: strict is the DEFAULT now (completion plan step 2.1),
    // so the warn-and-continue path must opt out explicitly.
    cumes::SolverOptions compat = opts;
    compat.strict_schema = false;
    write_scratch(
        "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0], \"n_theta\": 6,"
        " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
        " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
    vr = cumes::read_and_validate(scratch_path(), compat);
    check(vr.has_value() && vr.value().spec().mpol == 2,
          "neg: unknown key parse succeeds");
    if (vr.has_value()) {
        bool warned = false;
        for (const auto& issue : vr.value().warnings().issues())
            if (issue.message.find("unknown input key 'n_theta'") !=
                std::string::npos)
                warned = true;
        check(warned, "neg: unknown key recorded as a warning");
    }
}

// "two_power" is accepted for the mass and current profiles and lands in the
// ProblemSpec as ProfileType::TWO_POWER; an absent type stays POWER_SERIES.
static void test_two_power_profiles() {
    cumes::SolverOptions opts;

    {
        write_scratch(
            "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0, 5.0, 10.0],"
            " \"pmass_type\": \"two_power\","
            " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
            " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
        auto vr = cumes::read_and_validate(scratch_path(), opts);
        check(
            vr.has_value() &&
                vr.value().spec().mass.type == cumes::ProfileType::TWO_POWER &&
                vr.value().spec().mass.coefficients.size() == 3,
            "pos: two_power pmass_type parsed");
    }

    {
        write_scratch(
            "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0],"
            " \"ncurr\": 1, \"ac\": [1.0, 5.0, 10.0],"
            " \"pcurr_type\": \"two_power\","
            " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
            " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
        auto vr = cumes::read_and_validate(scratch_path(), opts);
        check(
            vr.has_value() &&
                vr.value().spec().current.type ==
                    cumes::ProfileType::TWO_POWER &&
                vr.value().spec().mass.type == cumes::ProfileType::POWER_SERIES,
            "pos: two_power pcurr_type parsed, absent types default");
    }

    // Extra coefficients beyond (c0, c1, c2) are accepted and ignored,
    // matching vmecpp.
    {
        write_scratch(
            "{\"mpol\": 2, \"ntor\": 0, \"am\": [1.0, 5.0, 10.0, 3.0],"
            " \"pmass_type\": \"two_power\","
            " \"rbc\": [{\"n\": 0, \"m\": 1, \"value\": 1.0}],"
            " \"zbs\": [{\"n\": 0, \"m\": 1, \"value\": 0.5}]}");
        auto vr = cumes::read_and_validate(scratch_path(), opts);
        check(vr.has_value(),
              "pos: two_power with extra coefficients still validates");
    }
}

int main() {
    test_solovev();
    test_w7x();
    test_errors();
    test_negative();
    test_two_power_profiles();
    remove(scratch_path());
    return summary();
}
