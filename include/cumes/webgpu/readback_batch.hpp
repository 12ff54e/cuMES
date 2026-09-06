#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

// One single-flight host fence. Each producer snapshots its output into a
// disjoint slice before its scratch can be reused. Decode callbacks must only
// collect results: the completion callback runs after every slice is decoded
// and the buffer is unmapped, and may start the next iteration.
class ReadbackBatch : public std::enable_shared_from_this<ReadbackBatch> {
   public:
    using Decode = std::function<void(std::span<const float>)>;
    using Complete = std::function<void(std::string)>;

    ReadbackBatch(const wgpu::Device& device, std::uint64_t capacity) {
        wgpu::BufferDescriptor descriptor{};
        descriptor.label = "cuMES iteration readback batch";
        descriptor.size = (capacity + 7) & ~std::uint64_t{7};
        descriptor.usage =
            wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        buffer_ = device.CreateBuffer(&descriptor);
    }

    void append(const wgpu::CommandEncoder& encoder,
                const wgpu::Buffer& source,
                std::uint64_t source_offset,
                std::uint64_t bytes,
                Decode decode) {
        if (mapping_ || bytes == 0 || bytes % sizeof(float) != 0 ||
            source_offset % sizeof(float) != 0 || used_ > buffer_.GetSize() ||
            bytes > buffer_.GetSize() - used_) {
            error_ = "invalid iteration readback slice";
            return;
        }
        encoder.CopyBufferToBuffer(source, source_offset, buffer_, used_,
                                   bytes);
        slices_.push_back({used_, bytes, std::move(decode)});
        used_ = (used_ + bytes + 7) & ~std::uint64_t{7};
    }

    void map(Complete complete) {
        if (mapping_) {
            complete("iteration readback is already mapping");
            return;
        }
        if (!error_.empty() || slices_.empty()) {
            auto error = std::exchange(error_, {});
            if (error.empty()) error = "iteration readback has no slices";
            slices_.clear();
            used_ = 0;
            complete(std::move(error));
            return;
        }
        mapping_ = true;
        const auto self = shared_from_this();
        buffer_.MapAsync(
            wgpu::MapMode::Read, 0, used_, wgpu::CallbackMode::AllowSpontaneous,
            [self, complete = std::move(complete)](wgpu::MapAsyncStatus status,
                                                   wgpu::StringView message) {
                std::string error;
                if (status != wgpu::MapAsyncStatus::Success) {
                    error = "iteration readback failed: ";
                    if (message.length)
                        error.append(message.data, message.length);
                } else {
                    const auto* h_values = static_cast<const float*>(
                        self->buffer_.GetConstMappedRange(0, self->used_));
                    if (!h_values) {
                        error = "iteration readback returned a null range";
                    } else {
                        for (const auto& slice : self->slices_)
                            slice.decode(
                                {h_values + slice.offset / sizeof(float),
                                 static_cast<std::size_t>(slice.bytes /
                                                          sizeof(float))});
                    }
                    self->buffer_.Unmap();
                }
                self->slices_.clear();
                if (error.empty()) error = std::move(self->error_);
                self->error_.clear();
                self->used_ = 0;
                self->mapping_ = false;
                complete(std::move(error));
            });
    }

   private:
    struct Slice {
        std::uint64_t offset, bytes;
        Decode decode;
    };
    wgpu::Buffer buffer_;
    std::vector<Slice> slices_;
    std::uint64_t used_ = 0;
    bool mapping_ = false;
    std::string error_;
};

// Batched calls publish device handles synchronously and deliver host values
// to the ordinary callback when the batch is mapped. The ordinary callback
// must only collect values; dependent GPU work belongs in device_ready.
template <typename Result>
struct BatchedReadback {
    std::shared_ptr<ReadbackBatch> batch;
    std::function<void(Result)> device_ready;
    void publish_device(Result result) const {
        if (device_ready) device_ready(std::move(result));
    }
};

}  // namespace cumes::webgpu
