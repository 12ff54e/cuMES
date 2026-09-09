#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include <webgpu_fft/shader.hpp>

namespace cumes::webgpu::detail {

inline std::string fft_shader(int length, bool optimized) {
    auto source = webgpu_fft::shader(length, true, false, optimized);
    // The pinned FFT dependency uses store/load rounding barriers. Firefox's
    // NVIDIA path needs read-modify-write operations to retain the low word.
    constexpr std::string_view ORIGINAL =
        "atomicStore(&rounding[s],bitcast<u32>(v));\n"
        "    return bitcast<f32>(atomicLoad(&rounding[s]));";
    constexpr std::string_view REPLACEMENT =
        "atomicExchange(&rounding[s],bitcast<u32>(v));\n"
        "    return bitcast<f32>(atomicAdd(&rounding[s],0u));";
    const auto position = source.find(ORIGINAL);
    if (position == std::string::npos)
        throw std::runtime_error(
            "FFT rounding helper changed; review Firefox workaround");
    source.replace(position, ORIGINAL.size(), REPLACEMENT);
    return source;
}

}  // namespace cumes::webgpu::detail
