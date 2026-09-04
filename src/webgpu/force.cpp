#include "cumes/webgpu/force.hpp"

#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/geometry.hpp"
#include "pipeline_cache.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

namespace cumes::webgpu {
namespace {
constexpr std::uint32_t WORKGROUP_SIZE = 256;
struct ShaderParams {
    std::uint32_t ns, n_z_n_t, full_points, half_points;
    float delta_s, lamscale;
    std::uint32_t padding[2];
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_case(const AxisymmetricForceCase& in) {
    if (in.ns < 2 || in.ntheta < 2 || in.ntheta % 2 != 0 || in.nzeta < 1 ||
        !(in.delta_s > 0.0F) || !std::isfinite(in.delta_s) ||
        !std::isfinite(in.lamscale))
        return "axisymmetric force has invalid shape or scalars";
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t nf = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t nh = static_cast<std::size_t>(in.ns - 1) * n_z_n_t;
    if (nf > std::numeric_limits<std::uint32_t>::max() ||
        nh > std::numeric_limits<std::uint32_t>::max())
        return "axisymmetric force exceeds WebGPU indexing limits";
    if (in.geometry.size() != GEOMETRY_PARITY_FIELD_COUNT * nf ||
        in.base_geometry.size() != BASE_GEOMETRY_FIELD_COUNT * nh ||
        in.magnetic_field.size() != MAGNETIC_FIELD_COUNT * nh ||
        in.sqrt_s_f.size() != static_cast<std::size_t>(in.ns) ||
        in.sqrt_s_h.size() != static_cast<std::size_t>(in.ns - 1) ||
        in.phip_f.size() != static_cast<std::size_t>(in.ns))
        return "axisymmetric force input shape mismatch";
    return {};
}
std::string load_shader() {
    std::ifstream f("/shaders/axisymmetric_force.wgsl", std::ios::binary);
    if (!f) return {};
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}
wgpu::Buffer buffer(const wgpu::Device& d,
                    std::uint64_t n,
                    wgpu::BufferUsage u,
                    const char* label) {
    return detail::cached_buffer(d, n, u, label);
}
struct Dispatch {
    AxisymmetricForceCallback callback;
    wgpu::Buffer result, readback;
    std::size_t values = 0, bytes = 0;
};
}  // namespace

AxisymmetricForceResult axisymmetric_force_reference(
    const AxisymmetricForceCase& in) {
    if (!validate_case(in).empty()) return {};
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t nf = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t nh = static_cast<std::size_t>(in.ns - 1) * n_z_n_t;
    AxisymmetricForceResult out;
    out.fields.resize(FORCE_FIELD_COUNT * nf);
    auto fg = [&](int f, std::size_t p) { return in.geometry[f * nf + p]; };
    auto hg = [&](int f, std::size_t p) {
        return in.base_geometry[f * nh + p];
    };
    auto bf = [&](int f, std::size_t p) {
        return in.magnetic_field[f * nh + p];
    };
    auto put = [&](int f, std::size_t p, float v) {
        out.fields[f * nf + p] = v;
    };
    for (int j = 0; j < in.ns; ++j) {
        for (std::size_t k = 0; k < n_z_n_t; ++k) {
            const std::size_t p = static_cast<std::size_t>(j) * n_z_n_t + k;
            float r12i = 0, ru12i = 0, zu12i = 0, rsi = 0, zsi = 0, taui = 0,
                  gi = 0, guvi = 0, gvvi = 0, bui = 0, bvi = 0, bdui = 0,
                  bdvi = 0, tpi = 0, shi = 0;
            float r12o = 0, ru12o = 0, zu12o = 0, rso = 0, zso = 0, tauo = 0,
                  go = 0, guvo = 0, gvvo = 0, buo = 0, bvo = 0, bduo = 0,
                  bdvo = 0, tpo = 0, sho = 0;
            if (j > 0) {
                const std::size_t h =
                    static_cast<std::size_t>(j - 1) * n_z_n_t + k;
                r12i = hg(0, h);
                ru12i = hg(1, h);
                zu12i = hg(2, h);
                rsi = hg(3, h);
                zsi = hg(4, h);
                taui = hg(5, h);
                gi = hg(6, h);
                guvi = hg(8, h);
                gvvi = hg(9, h);
                bui = bf(0, h);
                bvi = bf(1, h);
                bdui = bf(2, h);
                bdvi = bf(3, h);
                tpi = bf(4, h);
                shi = in.sqrt_s_h[j - 1];
            }
            if (j < in.ns - 1) {
                const std::size_t h = static_cast<std::size_t>(j) * n_z_n_t + k;
                r12o = hg(0, h);
                ru12o = hg(1, h);
                zu12o = hg(2, h);
                rso = hg(3, h);
                zso = hg(4, h);
                tauo = hg(5, h);
                go = hg(6, h);
                guvo = hg(8, h);
                gvvo = hg(9, h);
                buo = bf(0, h);
                bvo = bf(1, h);
                bduo = bf(2, h);
                bdvo = bf(3, h);
                tpo = bf(4, h);
                sho = in.sqrt_s_h[j];
            }
            const float pi = r12i * tpi, po = r12o * tpo;
            const float zupi = zu12i * pi, zupo = zu12o * po;
            const float rupi = ru12i * pi, rupo = ru12o * po;
            const float rspi = rsi * pi, rspo = rso * po;
            const float zspi = zsi * pi, zspo = zso * po;
            const float taupi = taui * tpi, taupo = tauo * tpo;
            const float gbuui = gi * bui * bui, gbuuo = go * buo * buo;
            const float gbvvi = gi * bvi * bvi, gbvvo = go * bvo * bvo;
            const float gbuv_i = gi * bui * bvi, gbuv_o = go * buo * bvo;
            const float invsi = j > 0 ? 1.0F / shi : 0.0F;
            const float invso = j < in.ns - 1 ? 1.0F / sho : 0.0F;
            const float pav = 0.5F * (po + pi);
            const float pw = 0.5F * (po * invso + pi * invsi);
            const float guav = 0.5F * (gbuuo + gbuui);
            const float guw = 0.5F * (gbuuo * sho + gbuui * shi);
            const float gvav = 0.5F * (gbvvo + gbvvi);
            const float gvw = 0.5F * (gbvvo * sho + gbvvi * shi);
            const float guvav = 0.5F * (gbuv_o + gbuv_i);
            const float guvw = 0.5F * (gbuv_o * sho + gbuv_i * shi);
            const float sf = in.sqrt_s_f[j], s = sf * sf;
            const float re = fg(0, p), ro = fg(6, p), zo = fg(7, p);
            const float rue = fg(3, p), ruo = fg(9, p);
            const float zue = fg(4, p), zuo = fg(10, p);
            const float rve = fg(12, p), rvo = fg(15, p);
            const float zve = fg(13, p), zvo = fg(16, p);
            put(0, p,
                (zupo - zupi) / in.delta_s + 0.5F * (taupo + taupi) -
                    gvav * re - gvw * ro);
            put(1, p,
                (zupo * sho - zupi * shi) / in.delta_s - 0.5F * pw * zue -
                    0.5F * pav * zuo + 0.5F * (taupo * sho + taupi * shi) -
                    gvw * re - gvav * ro * s);
            put(2, p, -(rupo - rupi) / in.delta_s);
            put(3, p,
                -(rupo * sho - rupi * shi) / in.delta_s + 0.5F * pw * rue +
                    0.5F * pav * ruo);
            put(4, p,
                0.5F * (zspo + zspi) + 0.5F * pw * zo - guav * rue - guw * ruo -
                    guvav * rve - guvw * rvo);
            put(5, p,
                0.5F * (zspo * sho + zspi * shi) + 0.5F * pav * zo - guw * rue -
                    guav * ruo * s - guvw * rve - guvav * rvo * s);
            put(6, p,
                -0.5F * (rspo + rspi) - 0.5F * pw * ro - guav * zue -
                    guw * zuo - guvav * zve - guvw * zvo);
            put(7, p,
                -0.5F * (rspo * sho + rspi * shi) - 0.5F * pav * ro -
                    guw * zue - guav * zuo * s - guvw * zve - guvav * zvo * s);
            const float gvi = j > 0 ? gvvi / gi : 0.0F;
            const float gvo = j < in.ns - 1 ? gvvo / go : 0.0F;
            const float guv_bu_i = j > 0 ? guvi * bui : 0.0F;
            const float guv_bu_o = j < in.ns - 1 ? guvo * buo : 0.0F;
            const float lue = in.lamscale * fg(5, p) + in.phip_f[j];
            const float luo = in.lamscale * fg(11, p);
            const float alt = 0.5F * (gvi + gvo) * lue +
                              0.5F * (gvi * shi + gvo * sho) * luo +
                              0.5F * (guv_bu_i + guv_bu_o);
            const float blend = 0.1F * (1.0F - s);
            float lambda = 0.5F * (bdvo + bdvi) * (1.0F - blend) + alt * blend;
            if (j > 0) lambda *= -in.lamscale;
            put(8, p, lambda);
            put(9, p, lambda * sf);
            put(10, p, guvav * rue + guvw * ruo + gvav * rve + gvw * rvo);
            put(11, p,
                guvw * rue + guvav * ruo * s + gvw * rve + gvav * rvo * s);
            put(12, p, guvav * zue + guvw * zuo + gvav * zve + gvw * zvo);
            put(13, p,
                guvw * zue + guvav * zuo * s + gvw * zve + gvav * zvo * s);
            float lambda_toroidal = 0.5F * (bduo + bdui);
            if (j > 0) lambda_toroidal *= -in.lamscale;
            put(14, p, lambda_toroidal);
            put(15, p, lambda_toroidal * sf);
        }
    }
    return out;
}

void enqueue_axisymmetric_force(const wgpu::Device& device,
                                const AxisymmetricForceCase& in,
                                AxisymmetricForceCallback callback) {
    const auto err = validate_case(in);
    if (!err.empty()) {
        callback(err, {});
        return;
    }
    const auto shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/axisymmetric_force.wgsl", {});
        return;
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t nf = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t nh = static_cast<std::size_t>(in.ns - 1) * n_z_n_t;
    std::vector<float> radial;
    radial.insert(radial.end(), in.sqrt_s_f.begin(), in.sqrt_s_f.end());
    radial.insert(radial.end(), in.sqrt_s_h.begin(), in.sqrt_s_h.end());
    radial.insert(radial.end(), in.phip_f.begin(), in.phip_f.end());
    const auto gb = in.geometry.size() * sizeof(float);
    const auto hb = in.base_geometry.size() * sizeof(float);
    const auto bb = in.magnetic_field.size() * sizeof(float);
    const auto rb = radial.size() * sizeof(float);
    const auto values = FORCE_FIELD_COUNT * nf;
    const auto ob = values * sizeof(float);
    auto gbuf = buffer(device, gb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force geometry");
    auto hbuf = buffer(device, hb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force half geometry");
    auto bbuf = buffer(device, bb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force magnetic field");
    auto rbuf = buffer(device, rb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force radial profiles");
    auto obuf = buffer(device, ob,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                       "force output");
    auto read = buffer(device, ob,
                       wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                       "force readback");
    auto pbuf = buffer(device, sizeof(ShaderParams),
                       wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                       "force params");
    const auto& pipeline = detail::cached_compute_pipeline(
        device, "mhd-force", shader_text, "cuMES MHD force pipeline");
    ShaderParams params{static_cast<std::uint32_t>(in.ns),
                        static_cast<std::uint32_t>(n_z_n_t),
                        static_cast<std::uint32_t>(nf),
                        static_cast<std::uint32_t>(nh),
                        in.delta_s,
                        in.lamscale,
                        {0, 0}};
    auto q = device.GetQueue();
    q.WriteBuffer(gbuf, 0, in.geometry.data(), gb);
    q.WriteBuffer(hbuf, 0, in.base_geometry.data(), hb);
    q.WriteBuffer(bbuf, 0, in.magnetic_field.data(), bb);
    q.WriteBuffer(rbuf, 0, radial.data(), rb);
    q.WriteBuffer(pbuf, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, gbuf, 0, gb, nullptr, nullptr},
        {nullptr, 1, hbuf, 0, hb, nullptr, nullptr},
        {nullptr, 2, bbuf, 0, bb, nullptr, nullptr},
        {nullptr, 3, rbuf, 0, rb, nullptr, nullptr},
        {nullptr, 4, obuf, 0, ob, nullptr, nullptr},
        {nullptr, 5, pbuf, 0, sizeof(params), nullptr, nullptr}};
    wgpu::BindGroupDescriptor bd{};
    bd.layout = layout;
    bd.entryCount = std::size(entries);
    bd.entries = entries;
    auto group = device.CreateBindGroup(&bd);
    auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor passd{};
    auto pass = encoder.BeginComputePass(&passd);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(nf) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(obuf, 0, read, 0, ob);
    auto commands = encoder.Finish();
    q.Submit(1, &commands);
    auto d = std::make_shared<Dispatch>();
    d->callback = std::move(callback);
    d->result = obuf;
    d->readback = read;
    d->values = values;
    d->bytes = ob;
    read.MapAsync(
        wgpu::MapMode::Read, 0, ob, wgpu::CallbackMode::AllowSpontaneous,
        [d](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                d->callback("WebGPU force mapping failed: " +
                                std::string(message.data, message.length),
                            {});
                return;
            }
            const auto* v = static_cast<const float*>(
                d->readback.GetConstMappedRange(0, d->bytes));
            if (v == nullptr) {
                d->callback("WebGPU returned a null mapped range", {});
                return;
            }
            AxisymmetricForceResult out;
            out.fields.assign(v, v + d->values);
            d->readback.Unmap();
            d->callback({}, std::move(out));
        });
}
}  // namespace cumes::webgpu
