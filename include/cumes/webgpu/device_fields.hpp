#pragma once

#include "cumes/webgpu/readback_batch.hpp"

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

// A committed copy, independent of operator scratch. Capture only after the
// candidate has passed its validity gates; rejected candidates leave it intact.
// One allocation per shape, reused for subsequent commits.
class FieldSnapshot {
   public:
    DeviceFields capture(const wgpu::Device& device,
                         const DeviceFields& source,
                         bool paired,
                         const char* label) {
        if (!source) return {};
        const auto bytes = source.values * sizeof(float);
        const auto size = bytes * (paired ? 2 : 1);
        if (!buffer_ || buffer_.GetSize() != size) {
            wgpu::BufferDescriptor descriptor{};
            descriptor.label = label;
            descriptor.size = size;
            descriptor.usage = wgpu::BufferUsage::Storage |
                               wgpu::BufferUsage::CopySrc |
                               wgpu::BufferUsage::CopyDst;
            buffer_ = device.CreateBuffer(&descriptor);
        }
        const auto encoder = device.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(source.buffer, source.high_offset, buffer_,
                                   0, bytes);
        if (paired)
            encoder.CopyBufferToBuffer(source.buffer, source.low_offset,
                                       buffer_, bytes, bytes);
        const auto commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
        return {buffer_, source.values, 0, paired ? bytes : 0};
    }

   private:
    wgpu::Buffer buffer_;
};

inline DeviceFields field_slice(const DeviceFields& fields,
                                std::size_t offset,
                                std::size_t count) {
    return {fields.buffer, count, fields.high_offset + offset * sizeof(float),
            fields.low_offset + offset * sizeof(float)};
}

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
