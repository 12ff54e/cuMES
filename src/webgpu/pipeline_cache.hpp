#pragma once

#include <cstdint>
#include <map>
#include <string>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu::detail {

struct BufferCacheEntry {
    wgpu::Buffer buffer;
    std::uint64_t capacity = 0;
    wgpu::BufferUsage usage = wgpu::BufferUsage::None;
};

// The browser solver is deliberately single-flight: an operator starts only
// after the previous operator's readback callback has unmapped its buffer.
// That makes one persistent allocation per semantic label safe to reuse.
inline const wgpu::Buffer& cached_buffer(const wgpu::Device& device,
                                         std::uint64_t size,
                                         wgpu::BufferUsage usage,
                                         const char* label) {
    static std::map<std::string, BufferCacheEntry> cache;
    auto [position, inserted] = cache.try_emplace(label);
    auto& entry = position->second;
    if (inserted || entry.capacity < size || entry.usage != usage) {
        if (!inserted) entry.buffer.Destroy();
        wgpu::BufferDescriptor descriptor{};
        descriptor.label = label;
        descriptor.size = size;
        descriptor.usage = usage;
        entry.buffer = device.CreateBuffer(&descriptor);
        entry.capacity = size;
        entry.usage = usage;
    }
    return entry.buffer;
}

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
