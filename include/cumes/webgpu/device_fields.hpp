#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

// Owning handle to field-major GPU planes. Copies retain the allocation, not
// its contents: the producer may overwrite it on its next enqueue. Consumers
// must be queued first. Low words may follow extra producer-specific planes.
struct DeviceFields {
    wgpu::Buffer buffer;
    std::size_t values = 0;
    std::uint64_t high_offset = 0;
    std::uint64_t low_offset = 0;
    explicit operator bool() const { return static_cast<bool>(buffer); }
};

inline bool field_shape(const std::vector<float>& host,
                        const DeviceFields& device,
                        std::size_t count) {
    return device ? device.values == count : host.size() == count;
}

// A device-to-device copy also handles unaligned sub-plane offsets, which
// cannot be used directly as storage bindings on all WebGPU implementations.
inline void transfer_fields(const wgpu::Device& device,
                            const wgpu::CommandEncoder& encoder,
                            const wgpu::Buffer& destination,
                            const std::vector<float>& host,
                            const DeviceFields& source,
                            bool low = false) {
    if (!source) {
        device.GetQueue().WriteBuffer(destination, 0, host.data(),
                                      host.size() * sizeof(float));
        return;
    }
    if (source.values * sizeof(float) < destination.GetSize()) {
        encoder.ClearBuffer(destination);
    }
    encoder.CopyBufferToBuffer(source.buffer,
                               low ? source.low_offset : source.high_offset,
                               destination, 0, source.values * sizeof(float));
}

inline void transfer_fields(const wgpu::Device& device,
                            const wgpu::Buffer& destination,
                            const std::vector<float>& host,
                            const DeviceFields& source,
                            bool low = false) {
    if (!source) {
        device.GetQueue().WriteBuffer(destination, 0, host.data(),
                                      host.size() * sizeof(float));
        return;
    }
    const auto encoder = device.CreateCommandEncoder();
    transfer_fields(device, encoder, destination, host, source, low);
    const auto commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
}

}  // namespace cumes::webgpu
