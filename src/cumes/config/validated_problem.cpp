// validated_problem.cpp — validate/fold/resolve the dynamic ProblemSpec into
// the immutable ValidatedProblem and emit the canonical normalization.
#include "cumes/config/validated_problem.hpp"

// Profile normalization evaluation (T_edge/C_edge — completion plan step 1).
#include "cumes/config/profile_functions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace cumes {

namespace {

// Fold raw signed-n boundary harmonics into the n>=0 product basis (blueprint
// §4.2), matching the legacy foldBoundary exactly:
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
    auto idx = [&](int m, int n) {
        return static_cast<std::size_t>(m) * ntorp1 +
               static_cast<std::size_t>(n);
    };
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
std::vector<BoundaryHarmonic> sorted_harmonics(
    const std::vector<BoundaryHarmonic>& in) {
    std::vector<BoundaryHarmonic> out = in;
    std::sort(out.begin(), out.end(),
              [](const BoundaryHarmonic& a, const BoundaryHarmonic& b) {
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
    if (spec.nfp < 1) { report.error("nfp", "nfp must be >= 1"); }
    if (!std::isfinite(spec.physical.phiedge)) {
        report.error("phiedge", "phiedge must be finite");
    } else if (spec.physical.phiedge == 0.0) {
        report.error("phiedge", "phiedge must be nonzero");
    }
    if (!std::isfinite(spec.physical.curtor)) {
        report.error("curtor", "curtor must be finite");
    }
    // ---- profile parameterizations ----
    // two_power is f(s) = c0·(1 − s^c1)^c2: reject fewer than three
    // coefficients up front (vmecpp would log and evaluate 0 instead).
    if (spec.mass.type == ProfileType::kTwoPower &&
        spec.mass.coefficients.size() < 3) {
        report.error("am",
                     "am: the two_power profile needs at least 3 "
                     "coefficients (c0, c1, c2)");
    }
    if (spec.current.type == ProfileType::kTwoPower &&
        spec.current.coefficients.size() < 3) {
        report.error("ac",
                     "ac: the two_power profile needs at least 3 "
                     "coefficients (c0, c1, c2)");
    }
    // ---- profile normalization scalars (completion plan step 1) ----
    // The device-side Profiles step divides by these two host-evaluated
    // normalization scalars: maxToroidalFlux = signJ·phiedge/(2π·T_edge) with
    // T_edge = T(1) = torflux(1), and (ncurr=1) Itor = signJ·μ0·curtor/
    // (2π·C_edge) with C_edge = J_C(min(|bloat|, 1)) — the prescribed-current
    // edge integral, independent of T(1). A non-finite, zero, or ill-scaled
    // value here would poison every downstream quantity, so both are rejected
    // BEFORE any CUDA context or stage construction. The evaluation is the
    // shared cumes::torflux/evalCurrProfile used by the upload step, so the
    // validated values are bit-identical to the divided ones.
    const double torflux_edge = torflux<double>(spec, 1.0);
    if (!std::isfinite(torflux_edge)) {
        report.error(
            "aphi",
            "toroidal-flux profile is non-finite at the edge "
            "(T(1) is not finite); the edge normalization would be unusable");
    } else if (torflux_edge == 0.0) {
        report.error("aphi",
                     "toroidal-flux profile is zero at the edge (T(1) = 0); "
                     "the edge normalization would divide by zero");
    } else if (std::fabs(torflux_edge) < 1e-30) {
        report.error("aphi",
                     "toroidal-flux profile is ill-scaled at the edge "
                     "(|T(1)| < 1e-30); the edge normalization would overflow");
    }
    if (spec.current_model == CurrentModel::kPrescribedCurrent) {
        const double curr_edge = evalCurrProfile<double>(spec, 1.0);
        if (!std::isfinite(curr_edge)) {
            report.error("ac",
                         "prescribed-current profile is non-finite at the edge "
                         "(C_edge is not finite); the current normalization "
                         "would be unusable");
        } else if (curr_edge == 0.0) {
            report.error(
                "ac",
                "prescribed-current profile integrates to zero at the edge "
                "(C_edge = 0); the current normalization would divide by zero");
        } else if (std::fabs(curr_edge) < 1e-30) {
            report.error(
                "ac",
                "prescribed-current profile is ill-scaled at the edge "
                "(|C_edge| < 1e-30); the current normalization would overflow");
        }
    }
    if (spec.delt <= 0.0) { report.error("delt", "delt must be positive"); }
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
        if (g > 0 && st.radial_surfaces <= spec.stages[g - 1].radial_surfaces) {
            report.error(
                "ns_array",
                "ns_array must be strictly increasing (monotonically "
                "non-decreasing is not enough: equal consecutive entries "
                "fail the grid-prolongation precondition ns_new > ns_old)");
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
               << " is below the "
               << (options.precision == PrecisionPolicy::kMixedFloat ? "float"
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
               << " (outside 0<=m<" << spec.mpol << ", |n|<=" << spec.ntor
               << ")";
            report.warn("rbc", os.str());
            continue;
        }
        kept_rbc.push_back(e);
    }
    for (const auto& e : spec.zbs) {
        if (e.m < 0 || e.m >= spec.mpol || std::abs(e.n) > spec.ntor) {
            std::ostringstream os;
            os << "zbs: skipping mode m=" << e.m << " n=" << e.n
               << " (outside 0<=m<" << spec.mpol << ", |n|<=" << spec.ntor
               << ")";
            report.warn("zbs", os.str());
            continue;
        }
        kept_zbs.push_back(e);
    }
    if (kept_rbc.empty()) {
        report.error("rbc",
                     "rbc: at least one boundary coefficient is required");
    }
    if (kept_zbs.empty()) {
        report.error("zbs",
                     "zbs: at least one boundary coefficient is required");
    }

    if (report.has_errors()) { return ValidationResult(report); }

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
       << (s.current_model == CurrentModel::kPrescribedCurrent ? 1 : 0)
       << ",\n";
    os << "  \"delt\":" << json_number(s.delt) << ",\n";
    os << "  \"physical\":{\"phiedge\":" << json_number(s.physical.phiedge)
       << ",\"pres_scale\":" << json_number(s.physical.pres_scale)
       << ",\"adiabatic_index\":" << json_number(s.physical.adiabatic_index)
       << ",\"spres_ped\":" << json_number(s.physical.spres_ped)
       << ",\"bloat\":" << json_number(s.physical.bloat)
       << ",\"curtor\":" << json_number(s.physical.curtor)
       << ",\"tcon0\":" << json_number(s.physical.tcon0) << "},\n";
    os << "  \"profiles\":{\"pmass_type\":\""
       << profileTypeToString(s.mass.type) << "\",\"piota_type\":\""
       << profileTypeToString(s.iota.type) << "\",\"pcurr_type\":\""
       << profileTypeToString(s.current.type) << "\",\"am\":";
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
