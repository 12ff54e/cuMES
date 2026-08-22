// input_params.hpp — the embedded normalized-input record. Every output
// container carries this record so a consumer can reconstruct the converged
// equilibrium without the input JSON.
//
// The record mirrors ValidatedProblem::normalize_to_json() field-for-field
// (configs/schema-v1.json): scalars, numeric arrays, the input stages, and
// the raw (signed-n) + folded boundary. All values come from the VALIDATED
// problem, so the angular grid is the resolved one (the ntheta/nzeta defaults
// already applied) and the axis is zero-padded to ntor+1.
//
// Serialization is per container: the versioned binary and the checkpoint
// append a fixed-order typed record (io_common.hpp write/readInputParams);
// the NetCDF/HDF5 writers map the fields to native scalar variables,
// datasets, and attributes. The schema tag names the normalized-input layout
// the record corresponds to.
#pragma once

#include "cumes/config/validated_problem.hpp"

#include <string>
#include <vector>

namespace cumes {

// One stage of the input request (the run-trailer stage records hold
// outcomes; this holds the requested ns/max_iter/ftol).
struct InputStage {
    int ns = 0;
    int max_iter = 0;
    double ftol = 0.0;
};

struct InputParams {
    std::string schema = "cumes-config-v1";
    int mpol = 0;
    int ntor = 0;
    int nfp = 0;
    int ntheta = 0;
    int nzeta = 0;
    int ncurr = 0;
    double delt = 0.0;
    double phiedge = 0.0;
    double pres_scale = 0.0;
    double adiabatic_index = 0.0;
    double spres_ped = 0.0;
    double bloat = 0.0;
    double curtor = 0.0;
    double tcon0 = 0.0;
    std::vector<double> am;       // mass/pressure power series
    std::vector<double> ac;       // prescribed current power series
    std::vector<double> ai;       // prescribed iota power series
    std::vector<double> aphi;     // toroidal flux power series
    std::vector<double> raxis_c;  // R axis cos coefficients (ntor+1 entries)
    std::vector<double> zaxis_s;  // Z axis sin coefficients (ntor+1 entries)
    std::vector<InputStage> stages;
    std::vector<int> rbc_m;  // raw boundary, signed-n VMEC convention
    std::vector<int> rbc_n;
    std::vector<double> rbc_value;
    std::vector<int> zbs_m;
    std::vector<int> zbs_n;
    std::vector<double> zbs_value;
    std::vector<double> rbcc;  // folded boundary (R: cos(mθ)cos(nζ))
    std::vector<double> rbss;  // R: sin(mθ)sin(nζ)
    std::vector<double> zbsc;  // Z: sin(mθ)cos(nζ)
    std::vector<double> zbcs;  // Z: cos(mθ)sin(nζ)
};

inline bool operator==(const InputStage& a, const InputStage& b) {
    return a.ns == b.ns && a.max_iter == b.max_iter && a.ftol == b.ftol;
}

inline bool operator==(const InputParams& a, const InputParams& b) {
    return a.schema == b.schema && a.mpol == b.mpol && a.ntor == b.ntor &&
           a.nfp == b.nfp && a.ntheta == b.ntheta && a.nzeta == b.nzeta &&
           a.ncurr == b.ncurr && a.delt == b.delt && a.phiedge == b.phiedge &&
           a.pres_scale == b.pres_scale &&
           a.adiabatic_index == b.adiabatic_index &&
           a.spres_ped == b.spres_ped && a.bloat == b.bloat &&
           a.curtor == b.curtor && a.tcon0 == b.tcon0 && a.am == b.am &&
           a.ac == b.ac && a.ai == b.ai && a.aphi == b.aphi &&
           a.raxis_c == b.raxis_c && a.zaxis_s == b.zaxis_s &&
           a.stages == b.stages && a.rbc_m == b.rbc_m && a.rbc_n == b.rbc_n &&
           a.rbc_value == b.rbc_value && a.zbs_m == b.zbs_m &&
           a.zbs_n == b.zbs_n && a.zbs_value == b.zbs_value &&
           a.rbcc == b.rbcc && a.rbss == b.rbss && a.zbsc == b.zbsc &&
           a.zbcs == b.zbcs;
}

// The embedded record from a validated problem: the RESOLVED values
// (angular-grid defaults applied, axis padded, folded boundary filled).
inline InputParams make_input_params(const ValidatedProblem& vp) {
    const ProblemSpec& sp = vp.spec();
    const FoldedBoundary& b = vp.boundary();
    InputParams p;
    p.mpol = sp.mpol;
    p.ntor = sp.ntor;
    p.nfp = sp.nfp;
    p.ntheta = sp.angular.ntheta;
    p.nzeta = sp.angular.nzeta;
    p.ncurr = (sp.current_model == CurrentModel::kPrescribedCurrent) ? 1 : 0;
    p.delt = sp.delt;
    p.phiedge = sp.physical.phiedge;
    p.pres_scale = sp.physical.pres_scale;
    p.adiabatic_index = sp.physical.adiabatic_index;
    p.spres_ped = sp.physical.spres_ped;
    p.bloat = sp.physical.bloat;
    p.curtor = sp.physical.curtor;
    p.tcon0 = sp.physical.tcon0;
    p.am = sp.mass.coefficients;
    p.ac = sp.current.coefficients;
    p.ai = sp.iota.coefficients;
    p.aphi = sp.toroidal_flux.coefficients;
    p.raxis_c = sp.raxis_c;
    p.zaxis_s = sp.zaxis_s;
    for (const auto& s : sp.stages) {
        InputStage st;
        st.ns = static_cast<int>(s.radial_surfaces);
        st.max_iter = static_cast<int>(s.max_iterations);
        st.ftol = s.tolerance;
        p.stages.push_back(st);
    }
    for (const auto& h : sp.rbc) {
        p.rbc_m.push_back(h.m);
        p.rbc_n.push_back(h.n);
        p.rbc_value.push_back(h.value);
    }
    for (const auto& h : sp.zbs) {
        p.zbs_m.push_back(h.m);
        p.zbs_n.push_back(h.n);
        p.zbs_value.push_back(h.value);
    }
    p.rbcc = b.rbcc;
    p.rbss = b.rbss;
    p.zbsc = b.zbsc;
    p.zbcs = b.zbcs;
    return p;
}

}  // namespace cumes
