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
// append a fixed-order typed record (io_common.hpp write/read_input_params);
// the NetCDF/HDF5 writers map the fields to native scalar variables,
// datasets, and attributes. The schema tag names the normalized-input layout
// the record corresponds to.
#ifndef CUMES_INCLUDE_CUMES_IO_INPUT_PARAMS_HPP_
#define CUMES_INCLUDE_CUMES_IO_INPUT_PARAMS_HPP_

#include "cumes/config/validated_problem.hpp"

#include <optional>
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
    bool lasym = false;
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
    std::string pmass_type = "power_series";
    std::string piota_type = "power_series";
    std::string pcurr_type = "power_series";
    bool lfreeb = false;
    int nvacskip = 1;
    std::string mgrid_file;
    std::string coils_file;
    std::string makegrid_parameters_file;
    std::optional<MakegridParametersSpec> embedded_makegrid_parameters;
    std::vector<double> extcur;
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
    // Complementary input fields, present only for lasym=true. Raw harmonics
    // retain signed n, ordering, duplicates, and explicit zero coefficients.
    std::vector<double> raxis_s;
    std::vector<double> zaxis_c;
    std::vector<int> rbs_m;
    std::vector<int> rbs_n;
    std::vector<double> rbs_value;
    std::vector<int> zbc_m;
    std::vector<int> zbc_n;
    std::vector<double> zbc_value;
    std::vector<double> rbsc;  // R: sin(mθ)cos(nζ)
    std::vector<double> rbcs;  // R: cos(mθ)sin(nζ)
    std::vector<double> zbcc;  // Z: cos(mθ)cos(nζ)
    std::vector<double> zbss;  // Z: sin(mθ)sin(nζ)
};

inline bool operator==(const InputStage& a, const InputStage& b) {
    return a.ns == b.ns && a.max_iter == b.max_iter && a.ftol == b.ftol;
}

inline bool operator==(const InputParams& a, const InputParams& b) {
    return a.lasym == b.lasym && a.schema == b.schema && a.mpol == b.mpol &&
           a.ntor == b.ntor && a.nfp == b.nfp && a.ntheta == b.ntheta &&
           a.nzeta == b.nzeta && a.ncurr == b.ncurr && a.delt == b.delt &&
           a.phiedge == b.phiedge && a.pres_scale == b.pres_scale &&
           a.adiabatic_index == b.adiabatic_index &&
           a.spres_ped == b.spres_ped && a.bloat == b.bloat &&
           a.curtor == b.curtor && a.tcon0 == b.tcon0 &&
           a.pmass_type == b.pmass_type && a.piota_type == b.piota_type &&
           a.pcurr_type == b.pcurr_type && a.lfreeb == b.lfreeb &&
           a.nvacskip == b.nvacskip && a.mgrid_file == b.mgrid_file &&
           a.coils_file == b.coils_file &&
           a.makegrid_parameters_file == b.makegrid_parameters_file &&
           a.embedded_makegrid_parameters == b.embedded_makegrid_parameters &&
           a.extcur == b.extcur && a.am == b.am && a.ac == b.ac &&
           a.ai == b.ai && a.aphi == b.aphi && a.raxis_c == b.raxis_c &&
           a.zaxis_s == b.zaxis_s && a.stages == b.stages &&
           a.rbc_m == b.rbc_m && a.rbc_n == b.rbc_n &&
           a.rbc_value == b.rbc_value && a.zbs_m == b.zbs_m &&
           a.zbs_n == b.zbs_n && a.zbs_value == b.zbs_value &&
           a.rbcc == b.rbcc && a.rbss == b.rbss && a.zbsc == b.zbsc &&
           a.zbcs == b.zbcs && a.raxis_s == b.raxis_s &&
           a.zaxis_c == b.zaxis_c && a.rbs_m == b.rbs_m && a.rbs_n == b.rbs_n &&
           a.rbs_value == b.rbs_value && a.zbc_m == b.zbc_m &&
           a.zbc_n == b.zbc_n && a.zbc_value == b.zbc_value &&
           a.rbsc == b.rbsc && a.rbcs == b.rbcs && a.zbcc == b.zbcc &&
           a.zbss == b.zbss;
}

// The embedded record from a validated problem: the RESOLVED values
// (angular-grid defaults applied, axis padded, folded boundary filled).
inline InputParams make_input_params(const ValidatedProblem& vp) {
    const ProblemSpec& sp = vp.spec();
    const FoldedBoundary& b = vp.boundary();
    InputParams p;
    p.lasym = sp.lasym;
    p.mpol = sp.mpol;
    p.ntor = sp.ntor;
    p.nfp = sp.nfp;
    p.ntheta = sp.angular.ntheta;
    p.nzeta = sp.angular.nzeta;
    p.ncurr = (sp.current_model == CurrentModel::PRESCRIBED_CURRENT) ? 1 : 0;
    p.delt = sp.delt;
    p.phiedge = sp.physical.phiedge;
    p.pres_scale = sp.physical.pres_scale;
    p.adiabatic_index = sp.physical.adiabatic_index;
    p.spres_ped = sp.physical.spres_ped;
    p.bloat = sp.physical.bloat;
    p.curtor = sp.physical.curtor;
    p.tcon0 = sp.physical.tcon0;
    p.pmass_type = profile_type_to_string(sp.mass.type);
    p.piota_type = profile_type_to_string(sp.iota.type);
    p.pcurr_type = profile_type_to_string(sp.current.type);
    p.lfreeb = sp.free_boundary.lfreeb;
    p.nvacskip = sp.free_boundary.nvacskip;
    p.mgrid_file = sp.free_boundary.mgrid_file;
    p.coils_file = sp.free_boundary.coils_file;
    p.makegrid_parameters_file = sp.free_boundary.makegrid_parameters_file;
    p.embedded_makegrid_parameters =
        sp.free_boundary.embedded_makegrid_parameters;
    p.extcur = sp.free_boundary.extcur;
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
    if (sp.lasym) {
        p.raxis_s = sp.raxis_s;
        p.zaxis_c = sp.zaxis_c;
        for (const auto& h : sp.rbs) {
            p.rbs_m.push_back(h.m);
            p.rbs_n.push_back(h.n);
            p.rbs_value.push_back(h.value);
        }
        for (const auto& h : sp.zbc) {
            p.zbc_m.push_back(h.m);
            p.zbc_n.push_back(h.n);
            p.zbc_value.push_back(h.value);
        }
        p.rbsc = b.rbsc;
        p.rbcs = b.rbcs;
        p.zbcc = b.zbcc;
        p.zbss = b.zbss;
    }
    return p;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_INPUT_PARAMS_HPP_
