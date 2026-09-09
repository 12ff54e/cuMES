#ifndef CUMES_WEBGPU_VACUUM_HPP_
#define CUMES_WEBGPU_VACUUM_HPP_

#include "cumes/physics/free_boundary_operator.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/initialization.hpp"

#include <span>

namespace cumes::webgpu {

std::unique_ptr<FreeBoundaryOperator<double>> create_vacuum(
    const ValidatedProblem& problem,
    const AxisymmetricStageData& stage,
    const wgpu::Device& device,
    bool use_webgpu,
    bool device_lu = false);
void prepare_vacuum_stage(FreeBoundaryOperator<double>& vacuum,
                          const ValidatedProblem& problem,
                          const AxisymmetricStageData& stage);
void update_vacuum(FreeBoundaryOperator<double>& vacuum,
                   const AxisymmetricStageData& stage,
                   std::span<const float> state_lo,
                   const AxisymmetricForceCase& fields);
void apply_vacuum_force(const wgpu::Device& device,
                        FreeBoundaryOperator<double>& vacuum,
                        const AxisymmetricStageData& stage,
                        const AxisymmetricForceCase& fields,
                        AxisymmetricForceResult& force);
// Enqueue device correction and append diagnostics/validation to the suffix
// readback. Invoke the returned completion after mapping, before accepting any
// controller state. It throws on invalid correction or vacuum outputs.
std::function<void()> enqueue_resident_vacuum_force(
    const wgpu::Device& device,
    FreeBoundaryOperator<double>& vacuum,
    const AxisymmetricStageData& stage,
    const AxisymmetricForceCase& fields,
    const AxisymmetricForceResult& force,
    const std::shared_ptr<ReadbackBatch>& batch);
void decay_vacuum_reference(std::vector<float>& high, std::vector<float>& low);

}  // namespace cumes::webgpu
#endif  // CUMES_WEBGPU_VACUUM_HPP_
