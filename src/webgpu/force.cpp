#include "cumes/webgpu/force.hpp"

#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/float_float.hpp"
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
    float delta_s, delta_s_lo, lamscale, lamscale_lo;
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
    if (in.double_single &&
        (in.geometry_lo.size() != in.geometry.size() ||
         in.base_geometry_lo.size() != in.base_geometry.size() ||
         in.magnetic_field_lo.size() != in.magnetic_field.size() ||
         in.sqrt_s_f_lo.size() != in.sqrt_s_f.size() ||
         in.sqrt_s_h_lo.size() != in.sqrt_s_h.size() ||
         in.phip_f_lo.size() != in.phip_f.size() ||
         !std::isfinite(in.delta_s_lo) || !std::isfinite(in.lamscale_lo)))
        return "double-single force input shape mismatch";
    return {};
}
std::string load_shader(bool double_single) {
    std::ifstream f(double_single ? "/shaders/force_double_single.wgsl"
                                  : "/shaders/axisymmetric_force.wgsl",
                    std::ios::binary);
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
    bool double_single = false;
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
    if (in.double_single) out.fields_lo.resize(FORCE_FIELD_COUNT * nf);
    const auto paired = [](const std::vector<float>& hi,
                           const std::vector<float>& lo, std::size_t index,
                           bool double_single) {
        return static_cast<double>(hi[index]) +
               (double_single ? static_cast<double>(lo[index]) : 0.0);
    };
    auto fg = [&](int f, std::size_t p) {
        return paired(in.geometry, in.geometry_lo, f * nf + p,
                      in.double_single);
    };
    auto hg = [&](int f, std::size_t p) {
        return paired(in.base_geometry, in.base_geometry_lo, f * nh + p,
                      in.double_single);
    };
    auto bf = [&](int f, std::size_t p) {
        return paired(in.magnetic_field, in.magnetic_field_lo, f * nh + p,
                      in.double_single);
    };
    auto put = [&](int f, std::size_t p, double v) {
        const std::size_t index = f * nf + p;
        const auto value_pair = split(v);
        out.fields[index] = value_pair.hi;
        if (in.double_single) out.fields_lo[index] = value_pair.lo;
    };
    for (int j = 0; j < in.ns; ++j) {
        for (std::size_t k = 0; k < n_z_n_t; ++k) {
            const std::size_t p = static_cast<std::size_t>(j) * n_z_n_t + k;
            double r12i = 0, ru12i = 0, zu12i = 0, rsi = 0, zsi = 0, taui = 0,
                   gi = 0, guvi = 0, gvvi = 0, bui = 0, bvi = 0, bdui = 0,
                   bdvi = 0, tpi = 0, shi = 0;
            double r12o = 0, ru12o = 0, zu12o = 0, rso = 0, zso = 0, tauo = 0,
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
                shi = paired(in.sqrt_s_h, in.sqrt_s_h_lo, j - 1,
                             in.double_single);
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
                sho = paired(in.sqrt_s_h, in.sqrt_s_h_lo, j, in.double_single);
            }
            const double delta_s = static_cast<double>(in.delta_s) +
                                   (in.double_single ? in.delta_s_lo : 0.0);
            const double lamscale = static_cast<double>(in.lamscale) +
                                    (in.double_single ? in.lamscale_lo : 0.0);
            const double pi = r12i * tpi, po = r12o * tpo;
            const double zupi = zu12i * pi, zupo = zu12o * po;
            const double rupi = ru12i * pi, rupo = ru12o * po;
            const double rspi = rsi * pi, rspo = rso * po;
            const double zspi = zsi * pi, zspo = zso * po;
            const double taupi = taui * tpi, taupo = tauo * tpo;
            const double gbuui = gi * bui * bui, gbuuo = go * buo * buo;
            const double gbvvi = gi * bvi * bvi, gbvvo = go * bvo * bvo;
            const double gbuv_i = gi * bui * bvi, gbuv_o = go * buo * bvo;
            const double invsi = j > 0 ? 1.0 / shi : 0.0;
            const double invso = j < in.ns - 1 ? 1.0 / sho : 0.0;
            const double pav = 0.5 * (po + pi);
            const double pw = 0.5 * (po * invso + pi * invsi);
            const double guav = 0.5 * (gbuuo + gbuui);
            const double guw = 0.5 * (gbuuo * sho + gbuui * shi);
            const double gvav = 0.5 * (gbvvo + gbvvi);
            const double gvw = 0.5 * (gbvvo * sho + gbvvi * shi);
            const double guvav = 0.5 * (gbuv_o + gbuv_i);
            const double guvw = 0.5 * (gbuv_o * sho + gbuv_i * shi);
            const double sf = paired(in.sqrt_s_f, in.sqrt_s_f_lo, j,
                                     in.double_single),
                         s = sf * sf;
            const double re = fg(0, p), ro = fg(6, p), zo = fg(7, p);
            const double rue = fg(3, p), ruo = fg(9, p);
            const double zue = fg(4, p), zuo = fg(10, p);
            const double rve = fg(12, p), rvo = fg(15, p);
            const double zve = fg(13, p), zvo = fg(16, p);
            put(0, p,
                (zupo - zupi) / delta_s + 0.5F * (taupo + taupi) - gvav * re -
                    gvw * ro);
            put(1, p,
                (zupo * sho - zupi * shi) / delta_s - 0.5F * pw * zue -
                    0.5F * pav * zuo + 0.5F * (taupo * sho + taupi * shi) -
                    gvw * re - gvav * ro * s);
            put(2, p, -(rupo - rupi) / delta_s);
            put(3, p,
                -(rupo * sho - rupi * shi) / delta_s + 0.5F * pw * rue +
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
            const double gvi = j > 0 ? gvvi / gi : 0.0;
            const double gvo = j < in.ns - 1 ? gvvo / go : 0.0;
            const double guv_bu_i = j > 0 ? guvi * bui : 0.0;
            const double guv_bu_o = j < in.ns - 1 ? guvo * buo : 0.0;
            const double phip =
                paired(in.phip_f, in.phip_f_lo, j, in.double_single);
            const double lue = lamscale * fg(5, p) + phip;
            const double luo = lamscale * fg(11, p);
            const double alt = 0.5F * (gvi + gvo) * lue +
                               0.5F * (gvi * shi + gvo * sho) * luo +
                               0.5F * (guv_bu_i + guv_bu_o);
            const double blend = 0.1F * (1.0F - s);
            double lambda = 0.5F * (bdvo + bdvi) * (1.0F - blend) + alt * blend;
            if (j > 0) lambda *= -lamscale;
            put(8, p, lambda);
            put(9, p, lambda * sf);
            put(10, p, guvav * rue + guvw * ruo + gvav * rve + gvw * rvo);
            put(11, p,
                guvw * rue + guvav * ruo * s + gvw * rve + gvav * rvo * s);
            put(12, p, guvav * zue + guvw * zuo + gvav * zve + gvw * zvo);
            put(13, p,
                guvw * zue + guvav * zuo * s + gvw * zve + gvav * zvo * s);
            double lambda_toroidal = 0.5F * (bduo + bdui);
            if (j > 0) lambda_toroidal *= -lamscale;
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
    const auto shader_text = load_shader(in.double_single);
    if (shader_text.empty()) {
        callback("cannot load embedded WebGPU force shader", {});
        return;
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t nf = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t nh = static_cast<std::size_t>(in.ns - 1) * n_z_n_t;
    std::vector<float> radial;
    radial.insert(radial.end(), in.sqrt_s_f.begin(), in.sqrt_s_f.end());
    radial.insert(radial.end(), in.sqrt_s_h.begin(), in.sqrt_s_h.end());
    radial.insert(radial.end(), in.phip_f.begin(), in.phip_f.end());
    std::vector<float> radial_lo;
    if (in.double_single) {
        radial_lo.insert(radial_lo.end(), in.sqrt_s_f_lo.begin(),
                         in.sqrt_s_f_lo.end());
        radial_lo.insert(radial_lo.end(), in.sqrt_s_h_lo.begin(),
                         in.sqrt_s_h_lo.end());
        radial_lo.insert(radial_lo.end(), in.phip_f_lo.begin(),
                         in.phip_f_lo.end());
        for (std::size_t i = 0; i < in.sqrt_s_h.size(); ++i) {
            const double sqrt_s =
                static_cast<double>(in.sqrt_s_h[i]) + in.sqrt_s_h_lo[i];
            const auto inverse = split(1.0 / sqrt_s);
            radial.push_back(inverse.hi);
            radial_lo.push_back(inverse.lo);
        }
    }
    const auto gb = in.geometry.size() * sizeof(float);
    const auto hb = in.base_geometry.size() * sizeof(float);
    const auto bb = in.magnetic_field.size() * sizeof(float);
    const auto rb = radial.size() * sizeof(float);
    const auto values = FORCE_FIELD_COUNT * nf;
    const auto ob = values * sizeof(float) * (in.double_single ? 2 : 1);
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
    wgpu::Buffer glbuf, hlbuf, blbuf, rlbuf, roundbuf;
    if (in.double_single) {
        glbuf = buffer(device, gb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force geometry low");
        hlbuf = buffer(device, hb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force half geometry low");
        blbuf = buffer(device, bb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force magnetic field low");
        rlbuf = buffer(device, rb,
                       wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                       "force radial profiles low");
        roundbuf =
            buffer(device, nf * sizeof(std::uint32_t),
                   wgpu::BufferUsage::Storage, "force double-single rounding");
    }
    const auto& pipeline = detail::cached_compute_pipeline(
        device, in.double_single ? "mhd-force-double-single" : "mhd-force",
        shader_text,
        in.double_single ? "cuMES double-single MHD force pipeline"
                         : "cuMES MHD force pipeline");
    ShaderParams params{static_cast<std::uint32_t>(in.ns),
                        static_cast<std::uint32_t>(n_z_n_t),
                        static_cast<std::uint32_t>(nf),
                        static_cast<std::uint32_t>(nh),
                        in.delta_s,
                        in.double_single ? in.delta_s_lo : in.lamscale,
                        in.double_single ? in.lamscale : 0.0F,
                        in.double_single ? in.lamscale_lo : 0.0F};
    auto q = device.GetQueue();
    q.WriteBuffer(gbuf, 0, in.geometry.data(), gb);
    q.WriteBuffer(hbuf, 0, in.base_geometry.data(), hb);
    q.WriteBuffer(bbuf, 0, in.magnetic_field.data(), bb);
    q.WriteBuffer(rbuf, 0, radial.data(), rb);
    if (in.double_single) {
        q.WriteBuffer(glbuf, 0, in.geometry_lo.data(), gb);
        q.WriteBuffer(hlbuf, 0, in.base_geometry_lo.data(), hb);
        q.WriteBuffer(blbuf, 0, in.magnetic_field_lo.data(), bb);
        q.WriteBuffer(rlbuf, 0, radial_lo.data(), rb);
    }
    q.WriteBuffer(pbuf, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    std::vector<wgpu::BindGroupEntry> entries = {
        {nullptr, 0, gbuf, 0, gb, nullptr, nullptr},
        {nullptr, 1, hbuf, 0, hb, nullptr, nullptr},
        {nullptr, 2, bbuf, 0, bb, nullptr, nullptr},
        {nullptr, 3, rbuf, 0, rb, nullptr, nullptr},
        {nullptr, 4, obuf, 0, ob, nullptr, nullptr},
        {nullptr, 5, pbuf, 0, sizeof(params), nullptr, nullptr}};
    if (in.double_single) {
        entries.push_back({nullptr, 6, glbuf, 0, gb, nullptr, nullptr});
        entries.push_back({nullptr, 7, hlbuf, 0, hb, nullptr, nullptr});
        entries.push_back({nullptr, 8, blbuf, 0, bb, nullptr, nullptr});
        entries.push_back({nullptr, 9, rlbuf, 0, rb, nullptr, nullptr});
        entries.push_back({nullptr, 10, roundbuf, 0, nf * sizeof(std::uint32_t),
                           nullptr, nullptr});
    }
    wgpu::BindGroupDescriptor bd{};
    bd.layout = layout;
    bd.entryCount = entries.size();
    bd.entries = entries.data();
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
    d->double_single = in.double_single;
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
            if (d->double_single)
                out.fields_lo.assign(v + d->values, v + 2 * d->values);
            d->readback.Unmap();
            d->callback({}, std::move(out));
        });
}
}  // namespace cumes::webgpu
