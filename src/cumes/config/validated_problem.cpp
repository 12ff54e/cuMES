// validated_problem.cpp — validate/fold/resolve the dynamic ProblemSpec into
// the immutable ValidatedProblem, plus the legacy InputParams bridge and the
// canonical normalization emitter.
#include "cumes/config/validated_problem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace cumes {

namespace {

// Fold raw signed-n boundary harmonics into the n>=0 product basis (blueprint
// §4.2), matching the legacy foldBoundary (include/input.h) exactly:
//   rbcc[m,n] = rbc[m,+n] + rbc[m,-n]        (all m)
//   rbss[m,n] = rbc[m,+n] - rbc[m,-n]        (m > 0)
//   zbsc[m,n] = zbs[m,+n] + zbs[m,-n]        (m > 0)
//   zbcs[m,n] = zbs[m,-n] - zbs[m,+n]        (all m)
FoldedBoundary fold_boundary(const ProblemSpec& spec, std::size_t modes) {
    FoldedBoundary fb;
    fb.rbcc.assign(modes, 0.0);
    fb.rbss.assign(modes, 0.0);
    fb.zbsc.assign(modes, 0.0);
    fb.zbcs.assign(modes, 0.0);
    const std::size_t ntorp1 = static_cast<std::size_t>(spec.ntor + 1);
    auto idx = [&](int m, int n) { return static_cast<std::size_t>(m) * ntorp1 +
                                          static_cast<std::size_t>(n); };
    for (const auto& e : spec.rbc) {
        const int n = std::abs(e.n);
        fb.rbcc[idx(e.m, n)] += e.value;
        const double sn = (e.n > 0) ? 1.0 : (e.n < 0) ? -1.0 : 0.0;
        if (e.m > 0) fb.rbss[idx(e.m, n)] += sn * e.value;
    }
    for (const auto& e : spec.zbs) {
        const int n = std::abs(e.n);
        const double sn = (e.n > 0) ? 1.0 : (e.n < 0) ? -1.0 : 0.0;
        if (e.m > 0) fb.zbsc[idx(e.m, n)] += e.value;
        fb.zbcs[idx(e.m, n)] -= sn * e.value;
    }
    return fb;
}

std::string json_number(double v) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

void emit_double_array(std::ostringstream& os, const std::vector<double>& v) {
    os << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << json_number(v[i]);
    }
    os << ']';
}

// Sorted copy of a boundary harmonic list (deterministic by (m, n) ascending).
std::vector<BoundaryHarmonic> sorted_harmonics(const std::vector<BoundaryHarmonic>& in) {
    std::vector<BoundaryHarmonic> out = in;
    std::sort(out.begin(), out.end(), [](const BoundaryHarmonic& a,
                                         const BoundaryHarmonic& b) {
        if (a.m != b.m) return a.m < b.m;
        return a.n < b.n;
    });
    return out;
}

void emit_harmonics(std::ostringstream& os,
                    const std::vector<BoundaryHarmonic>& harmonics) {
    os << '[';
    const auto sorted = sorted_harmonics(harmonics);
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (i) os << ',';
        os << "{\"m\":" << sorted[i].m << ",\"n\":" << sorted[i].n
           << ",\"value\":" << json_number(sorted[i].value) << '}';
    }
    os << ']';
}

}  // namespace

ValidationResult validate(ProblemSpec spec, const SolverOptions& options) {
    ValidationReport report;

    // ---- scalars (legacy VmecINDATA::IsConsistent ranges) ----
    if (spec.mpol < 2 || spec.mpol > 16) {
        report.error("mpol", "mpol must be in [2, 16]");
    }
    if (spec.ntor < 0 || spec.ntor > 15) {
        report.error("ntor", "ntor must be in [0, 15]");
    }
    if (spec.nfp < 1) {
        report.error("nfp", "nfp must be >= 1");
    }
    if (spec.physical.phiedge == 0.0) {
        report.error("phiedge", "phiedge must be nonzero");
    }
    if (spec.delt <= 0.0) {
        report.error("delt", "delt must be positive");
    }
    if (spec.angular.ntheta < 0 || spec.angular.ntheta > 256) {
        report.error("ntheta", "ntheta must be in [0, 256] (0 = default)");
    }
    if (spec.angular.nzeta < 0 || spec.angular.nzeta > 256) {
        report.error("nzeta", "nzeta must be in [0, 256] (0 = default)");
    }
    if (spec.physical.adiabatic_index != 0.0) {
        report.error("adiabatic_index",
                     "adiabatic_index (gamma) must be 0: the gamma != 0 model "
                     "(pres = mass/dVds^gamma) is not implemented by cuMES");
    }

    // ---- axis: a provided axis vector must match ntor+1 (an absent key is
    // zero-padded; a present-but-empty array is malformed) ----
    if (spec.has_raxis_c &&
        spec.raxis_c.size() != static_cast<std::size_t>(spec.ntor + 1)) {
        report.error("raxis_c", "raxis_c must have exactly ntor+1 entries");
    }
    if (spec.has_zaxis_s &&
        spec.zaxis_s.size() != static_cast<std::size_t>(spec.ntor + 1)) {
        report.error("zaxis_s", "zaxis_s must have exactly ntor+1 entries");
    }

    // ---- stage schedule ----
    if (spec.stages.empty()) {
        report.error("ns_array", "ns_array must contain at least one stage");
    }
    for (std::size_t g = 0; g < spec.stages.size(); ++g) {
        const StageRequest& st = spec.stages[g];
        if (st.radial_surfaces < 3 || st.radial_surfaces > 512) {
            report.error("ns_array", "ns_array entries must be in [3, 512]");
        }
        if (g > 0 &&
            st.radial_surfaces < spec.stages[g - 1].radial_surfaces) {
            report.error("ns_array", "ns_array must be monotonically non-decreasing");
        }
        if (st.max_iterations < 1) {
            report.error("niter_array", "niter_array entries must be >= 1");
        }
        if (st.tolerance <= 0.0) {
            report.error("ftol_array", "ftol_array entries must be positive");
        }
        if (!tolerance_achievable(st.tolerance, options.precision)) {
            std::ostringstream os;
            os << "ftol_array entry " << std::setprecision(17) << st.tolerance
               << " is below the " << (options.precision == PrecisionPolicy::kMixedFloat
                                           ? "float"
                                           : "double")
               << " floor " << tolerance_floor(options.precision)
               << "; relax the tolerance or choose a reachable policy";
            report.error("ftol_array", os.str());
        }
    }

    // ---- boundary: out-of-range harmonics are skipped with a warning ----
    std::vector<BoundaryHarmonic> kept_rbc;
    std::vector<BoundaryHarmonic> kept_zbs;
    for (const auto& e : spec.rbc) {
        if (e.m < 0 || e.m >= spec.mpol || std::abs(e.n) > spec.ntor) {
            std::ostringstream os;
            os << "rbc: skipping mode m=" << e.m << " n=" << e.n
               << " (outside 0<=m<" << spec.mpol << ", |n|<=" << spec.ntor << ")";
            report.warn("rbc", os.str());
            continue;
        }
        kept_rbc.push_back(e);
    }
    for (const auto& e : spec.zbs) {
        if (e.m < 0 || e.m >= spec.mpol || std::abs(e.n) > spec.ntor) {
            std::ostringstream os;
            os << "zbs: skipping mode m=" << e.m << " n=" << e.n
               << " (outside 0<=m<" << spec.mpol << ", |n|<=" << spec.ntor << ")";
            report.warn("zbs", os.str());
            continue;
        }
        kept_zbs.push_back(e);
    }
    if (kept_rbc.empty()) {
        report.error("rbc", "rbc: at least one boundary coefficient is required");
    }
    if (kept_zbs.empty()) {
        report.error("zbs", "zbs: at least one boundary coefficient is required");
    }

    if (report.has_errors()) {
        return ValidationResult(report);
    }

    // ---- resolve resolution defaults (Sizes::computeDerivedSizes) ----
    int ntheta = spec.angular.ntheta;
    if (ntheta < 2 * spec.mpol + 6) ntheta = 2 * spec.mpol + 6;
    ntheta = 2 * (ntheta / 2);  // nThetaEven
    int nzeta = spec.angular.nzeta;
    if (spec.ntor == 0) {
        if (nzeta < 1) nzeta = 1;
    } else if (nzeta < 2 * spec.ntor + 4) {
        nzeta = 2 * spec.ntor + 4;
    }

    // ---- assemble the immutable model ----
    ValidatedProblem vp;
    vp.options_ = options;
    vp.spec_ = std::move(spec);
    // Normalize the stored spec to the validated form: resolved angles,
    // axis vectors padded to ntor+1, and boundary reduced to the kept set.
    vp.spec_.angular.ntheta = ntheta;
    vp.spec_.angular.nzeta = nzeta;
    vp.spec_.raxis_c.resize(static_cast<std::size_t>(vp.spec_.ntor + 1), 0.0);
    vp.spec_.zaxis_s.resize(static_cast<std::size_t>(vp.spec_.ntor + 1), 0.0);
    vp.spec_.rbc = std::move(kept_rbc);
    vp.spec_.zbs = std::move(kept_zbs);

    for (const StageRequest& st : vp.spec_.stages) {
        GridShape gs;
        gs.ns = static_cast<int>(st.radial_surfaces);
        gs.ntheta = ntheta;
        gs.nzeta = nzeta;
        gs.mpol = vp.spec_.mpol;
        gs.ntor = vp.spec_.ntor;
        gs.nfp = vp.spec_.nfp;
        vp.stage_shapes_.push_back(gs);
    }
    vp.mode_table_ = ModeTable<double>::build(vp.shape());
    vp.boundary_ = fold_boundary(vp.spec_, vp.shape().modes());
    // On success `report` holds only warnings (boundary skips); carry them on
    // the model so the caller can report them.
    vp.warnings_ = std::move(report);
    return ValidationResult(vp);
}

Result<InputParams> ValidatedProblem::to_input_params() const {
    const ProblemSpec& s = spec_;
    InputParams p;  // legacy defaults

    p.mpol = s.mpol;
    p.ntor = s.ntor;
    p.nfp = s.nfp;
    p.ntheta = s.angular.ntheta;
    p.nzeta = s.angular.nzeta;
    p.ncurr = (s.current_model == CurrentModel::kPrescribedCurrent) ? 1 : 0;
    p.delt = s.delt;
    p.phiedge = s.physical.phiedge;
    p.pres_scale = s.physical.pres_scale;
    p.adiabatic_index = s.physical.adiabatic_index;
    p.spres_ped = s.physical.spres_ped;
    p.bloat = s.physical.bloat;
    p.curtor = s.physical.curtor;
    p.tcon0 = s.physical.tcon0;

    // ---- stage schedule (must fit the legacy 8-entry capacity) ----
    if (s.stages.size() > InputParams::kMaxGrids) {
        return Result<InputParams>("validated problem has " +
                                   std::to_string(s.stages.size()) +
                                   " stages, exceeding legacy capacity " +
                                   std::to_string(InputParams::kMaxGrids));
    }
    p.n_grids = static_cast<int>(s.stages.size());
    for (std::size_t g = 0; g < s.stages.size(); ++g) {
        p.ns_array[g] = static_cast<int>(s.stages[g].radial_surfaces);
        p.niter_array[g] = static_cast<int>(s.stages[g].max_iterations);
        p.ftol_array[g] = s.stages[g].tolerance;
    }
    p.ns = p.ns_array[0];
    p.max_iter = p.niter_array[0];
    p.ftol = p.ftol_array[0];

    // ---- profiles (legacy 16-coefficient capacity) ----
    auto copy_profile = [&](const ProfileSpec& prof, double* dst, int& count) {
        if (prof.coefficients.size() > InputParams::kMaxCoeff) return false;
        count = static_cast<int>(prof.coefficients.size());
        for (std::size_t i = 0; i < prof.coefficients.size(); ++i) {
            dst[i] = prof.coefficients[i];
        }
        return true;
    };
    if (!copy_profile(s.mass, p.am, p.am_n) ||
        !copy_profile(s.iota, p.ai, p.ai_n) ||
        !copy_profile(s.current, p.ac, p.ac_n) ||
        !copy_profile(s.toroidal_flux, p.aphi, p.aphi_n)) {
        return Result<InputParams>(
            "validated profile exceeds legacy 16-coefficient capacity");
    }

    // ---- axis (legacy 32-entry capacity) ----
    if (s.raxis_c.size() > 32 || s.zaxis_s.size() > 32) {
        return Result<InputParams>("validated axis vector exceeds legacy capacity");
    }
    for (std::size_t i = 0; i < s.raxis_c.size(); ++i) p.raxis_c[i] = s.raxis_c[i];
    for (std::size_t i = 0; i < s.zaxis_s.size(); ++i) p.zaxis_s[i] = s.zaxis_s[i];
    p.raxis_n = static_cast<int>(s.raxis_c.size());

    // ---- boundary raw + folded (legacy 256-entry / 16x16 capacities) ----
    if (s.rbc.size() > 256 || s.zbs.size() > 256) {
        return Result<InputParams>("validated boundary exceeds legacy 256-entry capacity");
    }
    p.rbc_n = static_cast<int>(s.rbc.size());
    for (std::size_t i = 0; i < s.rbc.size(); ++i) {
        p.rbc[i] = BoundaryEntry{s.rbc[i].m, s.rbc[i].n, s.rbc[i].value};
    }
    p.zbs_n = static_cast<int>(s.zbs.size());
    for (std::size_t i = 0; i < s.zbs.size(); ++i) {
        p.zbs[i] = BoundaryEntry{s.zbs[i].m, s.zbs[i].n, s.zbs[i].value};
    }
    for (int m = 0; m < p.mpol; ++m) {
        for (int n = 0; n <= p.ntor; ++n) {
            const std::size_t mode =
                static_cast<std::size_t>(m) * (p.ntor + 1) + n;
            p.rbcc[m][n] = boundary_.rbcc[mode];
            p.rbss[m][n] = boundary_.rbss[mode];
            p.zbsc[m][n] = boundary_.zbsc[mode];
            p.zbcs[m][n] = boundary_.zbcs[mode];
        }
    }
    return p;
}

std::string ValidatedProblem::normalize_to_json() const {
    const ProblemSpec& s = spec_;
    std::ostringstream os;
    os << std::setprecision(17);

    os << "{\n";
    os << "  \"schema\":\"cumes-config-v1\",\n";
    os << "  \"mpol\":" << s.mpol << ",\n";
    os << "  \"ntor\":" << s.ntor << ",\n";
    os << "  \"nfp\":" << s.nfp << ",\n";
    os << "  \"ntheta\":" << s.angular.ntheta << ",\n";
    os << "  \"nzeta\":" << s.angular.nzeta << ",\n";
    os << "  \"ncurr\":"
       << (s.current_model == CurrentModel::kPrescribedCurrent ? 1 : 0) << ",\n";
    os << "  \"delt\":" << json_number(s.delt) << ",\n";
    os << "  \"physical\":{\"phiedge\":" << json_number(s.physical.phiedge)
       << ",\"pres_scale\":" << json_number(s.physical.pres_scale)
       << ",\"adiabatic_index\":" << json_number(s.physical.adiabatic_index)
       << ",\"spres_ped\":" << json_number(s.physical.spres_ped)
       << ",\"bloat\":" << json_number(s.physical.bloat)
       << ",\"curtor\":" << json_number(s.physical.curtor)
       << ",\"tcon0\":" << json_number(s.physical.tcon0) << "},\n";
    os << "  \"profiles\":{\"am\":";
    emit_double_array(os, s.mass.coefficients);
    os << ",\"ac\":";
    emit_double_array(os, s.current.coefficients);
    os << ",\"ai\":";
    emit_double_array(os, s.iota.coefficients);
    os << ",\"aphi\":";
    emit_double_array(os, s.toroidal_flux.coefficients);
    os << "},\n";
    os << "  \"axis\":{\"raxis_c\":";
    emit_double_array(os, s.raxis_c);
    os << ",\"zaxis_s\":";
    emit_double_array(os, s.zaxis_s);
    os << "},\n";
    os << "  \"stages\":[";
    for (std::size_t g = 0; g < s.stages.size(); ++g) {
        if (g) os << ',';
        os << "{\"ns\":" << s.stages[g].radial_surfaces
           << ",\"max_iter\":" << s.stages[g].max_iterations
           << ",\"ftol\":" << json_number(s.stages[g].tolerance) << '}';
    }
    os << "],\n";
    os << "  \"boundary\":{\"rbc\":";
    emit_harmonics(os, s.rbc);
    os << ",\"zbs\":";
    emit_harmonics(os, s.zbs);
    os << "},\n";
    os << "  \"folded\":{\"rbcc\":";
    emit_double_array(os, boundary_.rbcc);
    os << ",\"rbss\":";
    emit_double_array(os, boundary_.rbss);
    os << ",\"zbsc\":";
    emit_double_array(os, boundary_.zbsc);
    os << ",\"zbcs\":";
    emit_double_array(os, boundary_.zbcs);
    os << "}\n";
    os << "}\n";
    return os.str();
}

}  // namespace cumes
