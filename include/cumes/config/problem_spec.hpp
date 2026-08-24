// problem_spec.hpp — dynamic user-facing problem description (blueprint §6.1).
//
// ProblemSpec holds values exactly as parsed/requested: no fixed capacities, no
// derived constants, no resolution defaults applied. Validation, folding, and
// resolution are the separate ValidatedProblem stage. The JSON reader
// (json_reader.hpp) maps a vmecpp-style document onto this type; a caller may
// also construct one directly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cumes {

enum class CurrentModel : std::uint8_t {
    kFixedIota = 0,         // ncurr = 0
    kPrescribedCurrent = 1  // ncurr = 1
};

enum class ProfileType : std::uint8_t { kPowerSeries = 0 };

// One power-series coefficient vector (am/ac/ai/aphi). `coefficients` is the
// raw polynomial list; length is the order.
struct ProfileSpec {
    ProfileType type = ProfileType::kPowerSeries;
    std::vector<double> coefficients;
};

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

// Free-boundary run parameters (vmecpp indata keys lfreeb / mgrid_file /
// extcur / nvacskip). Parsed for every input; only exercised when lfreeb.
struct FreeBoundarySpec {
    bool lfreeb = false;
    std::string mgrid_file;      // MAKEGRID-format coil field
    std::vector<double> extcur;  // coil currents (A)
    int nvacskip = 1;            // vacuum full-update cadence (>= 1)
};

struct ProblemSpec {
    int mpol = 6;
    int ntor = 0;
    int nfp = 1;
    AngularResolution angular;
    CurrentModel current_model = CurrentModel::kFixedIota;
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
inline StageRequest kDefaultStage() {
    return StageRequest{11, 1000, 1e-16};
}

}  // namespace cumes
