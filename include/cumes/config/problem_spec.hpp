// problem_spec.hpp — dynamic user-facing problem description (blueprint §6.1).
//
// ProblemSpec holds values exactly as parsed/requested: no fixed capacities, no
// derived constants, no resolution defaults applied. Validation, folding, and
// resolution are the separate ValidatedProblem stage. The JSON reader
// (json_reader.hpp) maps a vmecpp-style document onto this type; a caller may
// also construct one directly.
#ifndef CUMES_INCLUDE_CUMES_CONFIG_PROBLEM_SPEC_HPP_
#define CUMES_INCLUDE_CUMES_CONFIG_PROBLEM_SPEC_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cumes {

enum class CurrentModel : std::uint8_t {
    FIXED_IOTA = 0,         // ncurr = 0
    PRESCRIBED_CURRENT = 1  // ncurr = 1
};

// The radial-profile parameterizations cuMES supports. `TWO_POWER` is
// vmecpp's "two_power": f(s) = c0·(1 − s^c1)^c2, applicable to the mass
// (pressure) and current profiles only — the iota profile stays a power
// series (see json_reader.cpp).
enum class ProfileType : std::uint8_t { POWER_SERIES = 0, TWO_POWER = 1 };

// One profile coefficient vector (am/ac/ai/aphi) plus its parameterization.
// `coefficients` is the raw coefficient list; for POWER_SERIES its length is
// the polynomial order, for TWO_POWER entries 0..2 are c0, c1, c2.
struct ProfileSpec {
    ProfileType type = ProfileType::POWER_SERIES;
    std::vector<double> coefficients;
};

// Canonical name of a profile parameterization, as it appears in the input
// JSON and in the embedded normalized-input record.
inline const char* profile_type_to_string(ProfileType t) {
    switch (t) {
        case ProfileType::POWER_SERIES:
            return "power_series";
        case ProfileType::TWO_POWER:
            return "two_power";
    }
    return "power_series";
}

struct AngularResolution {
    int ntheta = 0;  // 0 => resolution default (2*mpol+6, made even)
    int nzeta = 0;   // 0 => resolution default (1 if ntor==0 else 2*ntor+4)
};

struct PhysicalScalars {
    double phiedge = 1.0;
    double pres_scale = 1.0;
    double adiabatic_index = 0.0;  // gamma; must stay 0 (unimplemented)
    double spres_ped = 1.0;
    double bloat = 1.0;
    double curtor = 0.0;
    double tcon0 = 1.0;
};

// A raw boundary harmonic with a SIGNED toroidal index (blueprint §4.2: the
// raw input n may be negative; folding happens in ValidatedProblem).
struct BoundaryHarmonic {
    int m = 0;
    int n = 0;
    double value = 0.0;
};

struct StageRequest {
    std::size_t radial_surfaces = 0;  // ns
    std::size_t max_iterations = 0;
    double tolerance = 0.0;  // ftol
};

// Embedded form of a MAKEGRID parameter JSON file (`makegrid_parameters`).
struct MakegridParametersSpec {
    bool normalize_by_currents = false;
    bool assume_stellarator_symmetry = false;
    int number_of_field_periods = 0;
    double r_grid_minimum = 0.0;
    double r_grid_maximum = 0.0;
    int number_of_r_grid_points = 0;
    double z_grid_minimum = 0.0;
    double z_grid_maximum = 0.0;
    int number_of_z_grid_points = 0;
    int number_of_phi_grid_points = 0;
};

inline bool operator==(const MakegridParametersSpec& a,
                       const MakegridParametersSpec& b) {
    return a.normalize_by_currents == b.normalize_by_currents &&
           a.assume_stellarator_symmetry == b.assume_stellarator_symmetry &&
           a.number_of_field_periods == b.number_of_field_periods &&
           a.r_grid_minimum == b.r_grid_minimum &&
           a.r_grid_maximum == b.r_grid_maximum &&
           a.number_of_r_grid_points == b.number_of_r_grid_points &&
           a.z_grid_minimum == b.z_grid_minimum &&
           a.z_grid_maximum == b.z_grid_maximum &&
           a.number_of_z_grid_points == b.number_of_z_grid_points &&
           a.number_of_phi_grid_points == b.number_of_phi_grid_points;
}

// Free-boundary run parameters (vmecpp indata keys plus cuMES's inline
// Makegrid source paths). Parsed for every input; only exercised when lfreeb.
struct FreeBoundarySpec {
    bool lfreeb = false;
    std::string mgrid_file;  // precomputed MAKEGRID-format coil field
    // In-memory alternative: vacuum-field builds the response table directly
    // from these two files at run construction.
    std::string coils_file;
    std::string makegrid_parameters_file;
    std::optional<MakegridParametersSpec> embedded_makegrid_parameters;
    std::vector<double> extcur;  // coil currents (A)
    int nvacskip = 1;            // vacuum full-update cadence (>= 1)
};

struct ProblemSpec {
    int mpol = 6;
    int ntor = 0;
    int nfp = 1;
    AngularResolution angular;
    CurrentModel current_model = CurrentModel::FIXED_IOTA;
    double delt = 0.9;

    ProfileSpec mass;           // am (pressure/mass power series)
    ProfileSpec iota;           // ai (prescribed iota, fixed-iota mode)
    ProfileSpec current;        // ac (prescribed current, ncurr=1)
    ProfileSpec toroidal_flux;  // aphi (toroidal flux; default {1.0})

    std::vector<double> raxis_c;  // R axis cos coefficients (length ntor+1)
    std::vector<double> zaxis_s;  // Z axis sin coefficients (length ntor+1)
    // Presence flags distinguish an absent key (zero-padded to ntor+1) from a
    // present-but-empty array (a malformed input the legacy parser rejects).
    bool has_raxis_c = false;
    bool has_zaxis_s = false;

    std::vector<BoundaryHarmonic> rbc;  // raw, signed-n
    std::vector<BoundaryHarmonic> zbs;  // raw, signed-n

    std::vector<StageRequest> stages;  // default: a single {11, 1000, 1e-16}

    PhysicalScalars physical;
    FreeBoundarySpec free_boundary;
};

// The legacy fixed-capacity defaults are reproduced here as the default single
// stage, matching InputParams ns=11 / max_iter=1000 / ftol=1e-16.
inline StageRequest default_stage() {
    return StageRequest{11, 1000, 1e-16};
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CONFIG_PROBLEM_SPEC_HPP_
