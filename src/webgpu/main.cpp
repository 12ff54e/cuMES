#include "cumes/webgpu/prolongation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <emscripten.h>
#include <emscripten/em_js.h>
#include <webgpu/webgpu_cpp.h>

namespace {

EM_JS(void, publish_browser_result, (int success, const char* detail), {
    document.body.dataset.cumesWebgpu = success ? 'pass' : 'fail';
    document.body.dataset.cumesDetail = UTF8ToString(detail);
    clearTimeout(window.cumesDeadline);
    clearInterval(window.cumesKeepAlive);
});

std::string message_text(wgpu::StringView message) {
    return message.length == 0 ? std::string{}
                               : std::string(message.data, message.length);
}

class BrowserSelfTest : public std::enable_shared_from_this<BrowserSelfTest> {
   public:
    void start() {
        instance_ = wgpu::Instance(wgpuCreateInstance(nullptr));
        if (!instance_) {
            finish(false, "wgpuCreateInstance returned null");
            return;
        }
        std::printf("cuMES WebGPU milestone: requesting adapter\n");
        const auto self = shared_from_this();
        instance_.RequestAdapter(
            nullptr, wgpu::CallbackMode::AllowSpontaneous,
            [self](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                   wgpu::StringView message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    self->finish(false, "WebGPU adapter request failed: " +
                                            message_text(message));
                    return;
                }
                self->adapter_ = std::move(adapter);
                self->request_device();
            });
    }

   private:
    void request_device() {
        const auto self = shared_from_this();
        wgpu::DeviceDescriptor descriptor{};
        descriptor.label = "cuMES WebGPU device";
        descriptor.SetUncapturedErrorCallback(
            [](const wgpu::Device&, wgpu::ErrorType type,
               wgpu::StringView message, BrowserSelfTest* test) {
                test->gpu_error_ = "uncaptured WebGPU error (" +
                                   std::to_string(static_cast<unsigned>(type)) +
                                   "): " + message_text(message);
                std::fprintf(stderr, "%s\n", test->gpu_error_.c_str());
            },
            this);
        adapter_.RequestDevice(
            &descriptor, wgpu::CallbackMode::AllowSpontaneous,
            [self](wgpu::RequestDeviceStatus status, wgpu::Device device,
                   wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    self->finish(false, "WebGPU device request failed: " +
                                            message_text(message));
                    return;
                }
                self->device_ = std::move(device);
                self->cases_ = make_cases();
                std::printf(
                    "adapter/device ready; running %zu radial-transfer "
                    "cases\n",
                    self->cases_.size());
                self->run_next();
            });
    }

    static std::vector<cumes::webgpu::ProlongationCase> make_cases() {
        std::vector<cumes::webgpu::ProlongationCase> cases;
        for (const auto interpolation :
             {cumes::webgpu::RadialInterpolation::LINEAR,
              cumes::webgpu::RadialInterpolation::CATMULL_ROM}) {
            cumes::webgpu::ProlongationCase test;
            test.ns_old = 5;
            test.ns_new = 8;
            test.ntor = 1;
            test.mnmax = 6;  // m=0..2, n=0..1 exercises true m parity.
            test.interpolation = interpolation;
            test.state.resize(6 * test.mnmax * test.ns_old);
            for (int family = 0; family < 6; ++family) {
                for (int mode = 0; mode < test.mnmax; ++mode) {
                    const bool odd = ((mode / (test.ntor + 1)) % 2) == 1;
                    for (int j = 0; j < test.ns_old; ++j) {
                        const float base =
                            0.4F * static_cast<float>(family + 1) +
                            0.2F * static_cast<float>(mode);
                        float value = base + 0.08F * static_cast<float>(j) +
                                      0.013F * static_cast<float>(j * j);
                        if (odd) {
                            const float s = static_cast<float>(j) /
                                            static_cast<float>(test.ns_old - 1);
                            const float radial = std::max(
                                std::sqrt(s),
                                std::sqrt(1.0F /
                                          static_cast<float>(test.ns_old - 1)));
                            value *= radial;
                            if (j == 0) value = 0.0F;
                        }
                        const auto index =
                            (static_cast<std::size_t>(family) * test.mnmax +
                             mode) *
                                test.ns_old +
                            j;
                        test.state[index] = value;
                    }
                }
            }
            cases.push_back(std::move(test));
        }
        return cases;
    }

    void run_next() {
        if (case_index_ == cases_.size()) {
            if (!gpu_error_.empty()) {
                finish(false, gpu_error_);
            } else {
                finish(true,
                       "linear and Catmull-Rom GPU/CPU agreement; odd-m axis "
                       "regularity, LCFS preservation, and velocity reset "
                       "verified");
            }
            return;
        }

        const auto self = shared_from_this();
        const auto interpolation = cases_[case_index_].interpolation;
        cumes::webgpu::enqueue_prolongation(
            device_, cases_[case_index_],
            [self, interpolation](std::string error,
                                  cumes::webgpu::ProlongationResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::prolongation_reference(
                    self->cases_[self->case_index_]);
                if (actual.state.size() != expected.state.size() ||
                    actual.velocity.size() != expected.velocity.size()) {
                    self->finish(false, "WebGPU result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                for (std::size_t i = 0; i < actual.state.size(); ++i) {
                    max_error =
                        std::max(max_error,
                                 std::abs(actual.state[i] - expected.state[i]));
                }
                const bool velocity_zero =
                    std::all_of(actual.velocity.begin(), actual.velocity.end(),
                                [](float value) { return value == 0.0F; });
                if (max_error > 4.0e-6F || !velocity_zero) {
                    self->finish(
                        false,
                        "radial-transfer mismatch: max_error=" +
                            std::to_string(max_error) + " velocity_zero=" +
                            std::string(velocity_zero ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  %s: PASS (max |GPU-CPU| = %.3e)\n",
                    interpolation == cumes::webgpu::RadialInterpolation::LINEAR
                        ? "linear"
                        : "Catmull-Rom",
                    static_cast<double>(max_error));
                ++self->case_index_;
                self->run_next();
            });
    }

    static void finish(bool success, const std::string& detail) {
        std::printf("cuMES WebGPU self-test: %s — %s\n",
                    success ? "PASS" : "FAIL", detail.c_str());
        std::fflush(stdout);
        std::fflush(stderr);
        publish_browser_result(success ? 1 : 0, detail.c_str());
        emscripten_force_exit(success ? 0 : 1);
    }

    wgpu::Instance instance_;
    wgpu::Adapter adapter_;
    wgpu::Device device_;
    std::vector<cumes::webgpu::ProlongationCase> cases_;
    std::size_t case_index_ = 0;
    std::string gpu_error_;
};

}  // namespace

int main() {
    std::make_shared<BrowserSelfTest>()->start();
    return 0;
}
