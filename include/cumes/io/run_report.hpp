// run_report.hpp — run outcome + full stage history and provenance
// (blueprint §6.11, §6.13). The host-only record a solver emits so the I/O
// layer never needs device state to write provenance.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cumes {

enum class RunStatus : std::uint8_t {
    kConverged = 0,
    kNotConverged = 1,
    kNumericalFailure = 2,
    kValidationFailure = 3,
    kOutputFailure = 4,
};

struct ResidualTriple {
    double fsqr = 0.0;
    double fsqz = 0.0;
    double fsql = 0.0;
};

struct RestartEvent {
    int iteration = 0;
};

struct StageReport {
    int ns = 0;
    int effective_iterations = 0;
    bool converged = false;
    ResidualTriple final_residual;
    std::vector<RestartEvent> restarts;
};

struct BuildProvenance {
    std::string revision;     // git revision or empty
    bool dirty = false;       // working tree dirty at build time
    std::string build_type;   // Release/Debug/...
    std::string scalar_type;  // "double" / "float"
    // Precision-policy provenance (completion plan step 3.1): the named
    // policy and its effective flags that produced the binary.
    std::string precision_policy;  // verify-double|fast-double|mixed-float|debug-double
    std::string compile_flags;     // effective fast-math/device-check flags
};

struct InputProvenance {
    std::string source_path;  // input JSON path
    std::string source_hash;  // hash of the input bytes, or empty
};

struct RuntimeProvenance {
    std::string gpu_name;    // e.g. "NVIDIA TITAN Xp"
    std::string driver;      // driver version
    std::string runtime;     // CUDA runtime version
    std::string toolkit;     // CUDA toolkit version
};

struct RunReport {
    RunStatus status = RunStatus::kNotConverged;
    int total_effective_iterations = 0;
    std::vector<StageReport> stages;
    BuildProvenance build;
    InputProvenance input;
    RuntimeProvenance runtime;
};

}  // namespace cumes
