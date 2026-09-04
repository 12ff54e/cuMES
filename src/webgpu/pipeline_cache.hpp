#pragma once

#include <map>
#include <string>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu::detail {

inline const wgpu::ComputePipeline& cached_compute_pipeline(
    const wgpu::Device& device,
    const std::string& key,
    const std::string& shader_text,
    const char* label,
    const char* entry_point = "main") {
    static std::map<std::string, wgpu::ComputePipeline> cache;
    auto [position, inserted] = cache.try_emplace(key);
    if (inserted) {
        wgpu::ShaderSourceWGSL wgsl{};
        wgsl.code = shader_text.c_str();
        wgpu::ShaderModuleDescriptor shader_descriptor{};
        shader_descriptor.label = label;
        shader_descriptor.nextInChain = &wgsl;
        const auto shader = device.CreateShaderModule(&shader_descriptor);
        wgpu::ComputePipelineDescriptor pipeline_descriptor{};
        pipeline_descriptor.label = label;
        pipeline_descriptor.compute.module = shader;
        pipeline_descriptor.compute.entryPoint = entry_point;
        position->second = device.CreateComputePipeline(&pipeline_descriptor);
    }
    return position->second;
}

}  // namespace cumes::webgpu::detail
