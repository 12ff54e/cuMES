#pragma once

#include <functional>
#include <string>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {
void run_vacuum_force_tests(const wgpu::Device& device,
                            std::function<void(std::string)> callback);
}  // namespace cumes::webgpu
