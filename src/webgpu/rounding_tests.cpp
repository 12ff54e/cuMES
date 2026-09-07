#include "rounding_tests.hpp"

#include "cumes/webgpu/readback_batch.hpp"
#include "fft_shader.hpp"
#include "pipeline_cache.hpp"
#include "shader_source.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace cumes::webgpu {
namespace {

struct RoundingCase {
    std::string_view shader;
    std::string_view function;
};
constexpr std::array CASES = {
    RoundingCase{"toroidal_inverse", "rounded"},
    RoundingCase{"toroidal_inverse_double_single", "ff_strict_round"},
    RoundingCase{"toroidal_forward_double_single", "rnd"},
    RoundingCase{"base_geometry_double_single", "ff_strict_round"},
    RoundingCase{"magnetic_field_double_single", "ff_strict_round"},
    RoundingCase{"force_double_single", "rnd"},
    RoundingCase{"constraint_head_double_single", "rnd"},
    RoundingCase{"constraint_tail_double_single", "rnd"},
    RoundingCase{"axisymmetric_descent_double_single", "ff_strict_round"},
    RoundingCase{"residual_decompose_double_single", "rnd"},
    RoundingCase{"residual_norm", "round32"},
    RoundingCase{"geometry_control", "round32"},
    RoundingCase{"fft", "rnd"}};
constexpr std::size_t COUNT = 128;

// Exercise the embedded production function, including its real scratch
// allocation. Copying a second implementation here would miss regressions.
std::string rounding_source(const RoundingCase& test) {
    const auto path =
        std::string("/shaders/") + std::string(test.shader) + ".wgsl";
    const auto source = test.shader == "fft"
                            ? detail::fft_shader(36, true)
                            : detail::cached_shader_source(path.c_str());
    const auto allocation = source.find("var<workgroup> rounding:");
    const auto allocation_end = source.find(';', allocation);
    const auto function =
        source.find(std::string("fn ") + std::string(test.function) + '(');
    const auto function_end = source.find('}', function);
    if (allocation_end == std::string::npos ||
        function_end == std::string::npos)
        throw std::runtime_error(path + ": rounding helper not found");
    return source.substr(allocation, allocation_end + 1 - allocation) + '\n' +
           source.substr(function, function_end + 1 - function) +
           "\nfn r(v:f32,s:u32)->f32 { return " + std::string(test.function) +
           "(v,s); }\n";
}

constexpr std::string_view PROBE = R"(
@group(0) @binding(0) var<storage,read> input:array<vec2f>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
fn probe(i:u32,s:u32) {
    let a=input[i].x;let b=input[i].y;
    let h=r(a+b,s);let bv=r(h-a,s);let av=r(h-bv,s);
    let ae=r(a-av,s);let be=r(b-bv,s);let l=r(ae+be,s);
    let p=r(a*b,s);let e=r(fma(a,b,-p),s);
    let ph=r(p+e,s);let pv=r(ph-p,s);let pl=r(e-pv,s);
    output[i]=vec4f(h,l,ph,pl);
}
@compute @workgroup_size(1)
fn single(@builtin(global_invocation_id) id:vec3u) { probe(id.x,0u); }
@compute @workgroup_size(32)
fn wide(@builtin(global_invocation_id) id:vec3u,
        @builtin(local_invocation_index) lane:u32) { probe(id.x,lane); }
)";

class RoundingTest : public std::enable_shared_from_this<RoundingTest> {
   public:
    wgpu::Device device;
    std::function<void(std::string)> callback;

    void start() {
        // Products and sums of these moderate f32 values are exact in f64.
        // Include cancellation and nonzero low words; no denormal assumptions.
        for (std::size_t i = 0; i < COUNT; ++i) {
            const float a = static_cast<float>((i % 2 ? -1.0 : 1.0) *
                                               (0.3 + double(i) / 171));
            const float b = static_cast<float>(std::sin(double(i) + 0.125));
            input_[2 * i] = a;
            input_[2 * i + 1] = b;
            const double sum = double(a) + b, product = double(a) * b;
            const float h = static_cast<float>(sum);
            const float p = static_cast<float>(product);
            expected_[4 * i] = h;
            expected_[4 * i + 1] = static_cast<float>(sum - h);
            expected_[4 * i + 2] = p;
            expected_[4 * i + 3] = static_cast<float>(product - p);
        }
        input_buffer_ = buffer(sizeof(input_), wgpu::BufferUsage::Storage |
                                                   wgpu::BufferUsage::CopyDst);
        output_buffer_ =
            buffer(sizeof(expected_),
                   wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
        device.GetQueue().WriteBuffer(input_buffer_, 0, input_.data(),
                                      sizeof(input_));
        next();
    }

   private:
    std::array<float, 2 * COUNT> input_{};
    std::array<float, 4 * COUNT> expected_{};
    wgpu::Buffer input_buffer_, output_buffer_;
    std::size_t case_ = 0;
    std::string error_;

    wgpu::Buffer buffer(std::uint64_t bytes, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor descriptor{};
        descriptor.label = "cuMES rounding regression";
        descriptor.size = bytes;
        descriptor.usage = usage;
        return device.CreateBuffer(&descriptor);
    }

    void next() {
        if (case_ == CASES.size()) {
            std::printf(
                "  production rounding barriers: PASS (%zu shaders, "
                "1/32 invocations, %zu exact sum/product cases)\n",
                CASES.size(), COUNT);
            callback({});
            return;
        }
        const auto& test = CASES[case_];
        auto source = rounding_source(test);
        source += PROBE;
        const auto encoder = device.CreateCommandEncoder();
        const auto batch =
            std::make_shared<ReadbackBatch>(device, 2 * sizeof(expected_));
        const auto self = shared_from_this();
        for (const auto entry :
             {std::string_view{"single"}, std::string_view{"wide"}}) {
            const auto key = std::string("rounding-test-") +
                             std::string(test.shader) + '-' +
                             std::string(entry);
            const auto& pipeline = detail::cached_compute_pipeline(
                device, key, source, "cuMES rounding regression", entry.data());
            const std::array<wgpu::BindGroupEntry, 2> entries = {
                wgpu::BindGroupEntry{nullptr, 0, input_buffer_, 0,
                                     sizeof(input_), nullptr, nullptr},
                wgpu::BindGroupEntry{nullptr, 1, output_buffer_, 0,
                                     sizeof(expected_), nullptr, nullptr}};
            wgpu::BindGroupDescriptor descriptor{};
            descriptor.layout = pipeline.GetBindGroupLayout(0);
            descriptor.entryCount = entries.size();
            descriptor.entries = entries.data();
            const auto group = device.CreateBindGroup(&descriptor);
            const auto pass = encoder.BeginComputePass();
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, group);
            pass.DispatchWorkgroups(entry == "single" ? COUNT : COUNT / 32);
            pass.End();
            batch->append(encoder, output_buffer_, 0, sizeof(expected_),
                          [self, key](std::span<const float> values) {
                              for (std::size_t i = 0; i < values.size(); ++i)
                                  if (values[i] != self->expected_[i] &&
                                      self->error_.empty())
                                      self->error_ =
                                          key +
                                          ": incorrect high/low word at " +
                                          std::to_string(i);
                          });
        }
        const auto command = encoder.Finish();
        device.GetQueue().Submit(1, &command);
        batch->map([self](std::string error) {
            if (error.empty()) error = self->error_;
            if (!error.empty()) {
                self->callback(std::move(error));
                return;
            }
            ++self->case_;
            self->next();
        });
    }
};

}  // namespace

void run_rounding_tests(const wgpu::Device& device,
                        std::function<void(std::string)> callback) {
    const auto test = std::make_shared<RoundingTest>();
    test->device = device;
    test->callback = std::move(callback);
    test->start();
}

}  // namespace cumes::webgpu
