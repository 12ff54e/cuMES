#pragma once

#include "cumes/webgpu/constraint.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/geometry.hpp"
#include "cumes/webgpu/initialization.hpp"
#include "cumes/webgpu/numerics.hpp"
#include "cumes/webgpu/preconditioner.hpp"
#include "cumes/webgpu/reduction.hpp"
#include "cumes/webgpu/toroidal.hpp"

#include <array>

namespace cumes::webgpu {

struct IterationCase {
    DeviceFields device_state;
    DeviceFields device_r_con0, device_z_con0;
    AxisymmetricStageData stage;
    bool double_single = false;
    bool refresh_preconditioner = false;
    bool reset_reference = false;
    bool zero_m1_z = false;
    bool use_fft = false;
    bool optimized_fft = true;
    bool canonical_zeta = false;
    bool shadow_norms = false;
    bool compact_norms = false;
    bool compact_fields = false;
    // Retain force and projection snapshots for per-operator CPU comparisons.
    bool readback_intermediates = false;
    bool geometry_control = false;
    bool include_lcfs = false;
    bool include_edge_invariant = false;
    AxisymmetricPreconditionerElements elements;
    AxisymmetricPreconditionerMatrix matrix;
    std::vector<float> r_con0, r_con0_lo, z_con0, z_con0_lo, tcon;
};

// Speculative force evaluation only: no controller, checkpoint, or persistent
// constraint state is committed until the host accepts the Jacobian. Results
// retain the original CPU reduction order and validation inputs.
// Forward/residual device handles alias reusable intra-iteration scratch;
// only their host vectors are snapshots, not those transient handles.
struct IterationResult {
    ToroidalInverseResult inverse;
    BaseGeometryResult geometry;
    MagneticFieldResult magnetic;
    AxisymmetricForceResult force;
    std::array<ToroidalForwardResult, 2> forward;
    std::array<ResidualDecompositionResult, 2> residual;
    AxisymmetricPreconditionerElements elements;
    AxisymmetricPreconditionerMatrix matrix;
    AxisymmetricConstraintResult constraint;
    AxisymmetricPreconditionerApplyResult preconditioned;
    std::array<ResidualNormResult, 3> norms;
};

using IterationCallback = std::function<void(std::string, IterationResult)>;
// Resume after the caller accepts the geometry and updates the vacuum force.
using ResumeIteration =
    std::function<void(IterationCase, DeviceFields, IterationCallback)>;
using IterationPrefixCallback =
    std::function<void(std::string, IterationResult, ResumeIteration)>;

std::uint64_t iteration_readback_capacity(const AxisymmetricStageData& stage,
                                          bool readback_intermediates = false);

void enqueue_iteration(const wgpu::Device& device,
                       IterationCase input,
                       const std::shared_ptr<ReadbackBatch>& batch,
                       IterationCallback callback);

void enqueue_iteration_prefix(const wgpu::Device& device,
                              IterationCase input,
                              const std::shared_ptr<ReadbackBatch>& batch,
                              IterationPrefixCallback callback);

}  // namespace cumes::webgpu
