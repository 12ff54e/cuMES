// run_report.hpp — run outcome + full stage history and provenance
// (blueprint §6.11, §6.13). The host-only record a solver emits so the I/O
// layer never needs device state to write provenance.
#pragma once

#include "cumes/io/input_params.hpp"

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
    std::string
        precision_policy;  // verify-double|fast-double|mixed-float|debug-double
    std::string compile_flags;  // effective fast-math/device-check flags
};

struct InputProvenance {
    std::string source_path;  // input JSON path
    std::string source_hash;  // hash of the input bytes, or empty
};

struct RuntimeProvenance {
    std::string gpu_name;  // e.g. "NVIDIA TITAN Xp"
    std::string driver;    // driver version
    std::string runtime;   // CUDA runtime version
    std::string toolkit;   // CUDA toolkit version
};

struct RunReport {
    RunStatus status = RunStatus::kNotConverged;
    int total_effective_iterations = 0;
    std::vector<StageReport> stages;
    BuildProvenance build;
    InputProvenance input;
    RuntimeProvenance runtime;
    // The embedded normalized-input record (input_params.hpp). Default-empty
    // for containers written before the record was introduced; every current
    // writer serializes it.
    InputParams input_params;
};

// Validate serialized per-stage restart offsets before they are used to index
// a restart-iteration array (completion-plan follow-up §2.2). The readers
// previously cast the serialized ints to size_t and indexed rst_iter[k]
// directly, so a corrupted file with negative, descending, or oversized
// offsets read out of bounds (or attributed restarts to the wrong stage).
// `offsets` must already have nstages entries. Returns an empty string when
// valid, or a human-readable reason.
inline std::string validateRestartOffsets(const std::vector<int>& offsets,
                                          std::size_t nstages,
                                          std::size_t nrestarts) {
    if (offsets.size() != nstages) {
        return "restart_stage_offset length " + std::to_string(offsets.size()) +
               " != nstages " + std::to_string(nstages);
    }
    if (nstages > 0 && offsets[0] != 0) {
        return "first restart_stage_offset is " + std::to_string(offsets[0]) +
               ", expected 0";
    }
    for (std::size_t g = 0; g < nstages; ++g) {
        if (offsets[g] < 0) {
            return "restart_stage_offset[" + std::to_string(g) +
                   "] is negative";
        }
        if (static_cast<std::size_t>(offsets[g]) > nrestarts) {
            return "restart_stage_offset[" + std::to_string(g) + "] (" +
                   std::to_string(offsets[g]) + ") exceeds nrestarts " +
                   std::to_string(nrestarts);
        }
        if (g + 1 < nstages && offsets[g + 1] < offsets[g]) {
            return "restart_stage_offset is not monotonic (" +
                   std::to_string(offsets[g]) + " then " +
                   std::to_string(offsets[g + 1]) + ")";
        }
    }
    return "";
}

}  // namespace cumes
