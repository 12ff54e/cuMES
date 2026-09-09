#include "cumes/webgpu/iteration.hpp"

#include "pipeline_cache.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace cumes::webgpu {
namespace {

class IterationDispatch
    : public std::enable_shared_from_this<IterationDispatch> {
   public:
    wgpu::Device device;
    IterationCase input;
    std::shared_ptr<ReadbackBatch> batch;
    IterationCallback callback;
    IterationResult result;
    std::string error;
    bool prefix_only = false;
    bool probe_only = false;
    IterationProbeResult probe_result;

    void resume(IterationCase next,
                DeviceFields force,
                IterationCallback complete) {
        input = std::move(next);
        callback = std::move(complete);
        prefix_only = false;
        force_.device_fields = std::move(force);
        forward(force_.device_fields, 0);
        map();
    }

    void map() {
        const auto self = shared_from_this();
        batch->map([self](std::string mapping_error) {
            if (self->error.empty()) self->error = std::move(mapping_error);
            self->complete();
        });
    }

    void start() {
        ToroidalInverseCase inverse;
        shape(inverse);
        inverse.nfp = input.stage.nfp;
        inverse.double_single = input.double_single;
        inverse.radius_reference = input.stage.radius_reference;
        inverse.compensated_geometry = input.stage.compensated_geometry;
        inverse.compensated_toroidal_geometry =
            input.stage.compensated_toroidal_geometry;
        inverse.state = input.stage.state;
        inverse.state_lo = input.stage.state_lo;
        inverse.device_state = input.device_state;
        inverse.readback_values = !input.compact_fields;
        inverse.readback = {
            batch, [self = shared_from_this()](ToroidalInverseResult value) {
                self->inverse_ = std::move(value);
                self->geometry();
            }};
        enqueue_toroidal_inverse(device, inverse,
                                 collect(&IterationResult::inverse));
        // All device-ready callbacks above are synchronous. The whole DAG is
        // submitted before this single map; decode callbacks only store values.
        if (!probe_only) map();
    }

   private:
    void complete() {
        auto& geometry = result.geometry;
        if (error.empty() && geometry.control.present &&
            geometry.control.fallback) {
            // Rare ambiguous ordering/range: preserve the complete host gate.
            geometry.control.present = false;
            if (geometry.fields.empty()) {
                const auto fields = geometry.device_fields;
                const auto bytes = fields.values * sizeof(float);
                const auto encoder = device.CreateCommandEncoder();
                const auto self = shared_from_this();
                batch->append(encoder, fields.buffer, fields.high_offset, bytes,
                              [self](std::span<const float> values) {
                                  self->result.geometry.fields.assign(
                                      values.begin(), values.end());
                              });
                if (input.double_single)
                    batch->append(encoder, fields.buffer, fields.low_offset,
                                  bytes, [self](std::span<const float> values) {
                                      self->result.geometry.fields_lo.assign(
                                          values.begin(), values.end());
                                  });
                const auto commands = encoder.Finish();
                device.GetQueue().Submit(1, &commands);
                batch->map([self](std::string message) {
                    self->error = std::move(message);
                    self->complete();
                });
                return;
            }
        }
        auto done = std::move(callback);
        done(std::move(error), std::move(result));
    }

    ToroidalInverseResult inverse_;
    BaseGeometryResult geometry_;
    MagneticFieldResult magnetic_;
    AxisymmetricForceResult force_;
    AxisymmetricPreconditionerElements elements_;
    AxisymmetricPreconditionerMatrix matrix_;

    void snapshot_fields(const DeviceFields& fields,
                         std::vector<float>& high,
                         std::vector<float>& low) {
        if (!input.readback_intermediates) return;
        const auto encoder = device.CreateCommandEncoder();
        const auto bytes = fields.values * sizeof(float);
        for (int word = 0; word < (input.double_single ? 2 : 1); ++word) {
            auto& destination = word ? low : high;
            // Result members remain alive until map() has decoded every slice.
            batch->append(encoder, fields.buffer,
                          word ? fields.low_offset : fields.high_offset, bytes,
                          [&destination](std::span<const float> values) {
                              destination.assign(values.begin(), values.end());
                          });
        }
        const auto commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
    }

    template <typename Case>
    void shape(Case& value) const {
        value.ns = input.stage.ns;
        if constexpr (requires { value.mpol; }) value.mpol = input.stage.mpol;
        if constexpr (requires { value.ntor; }) value.ntor = input.stage.ntor;
        if constexpr (requires { value.ntheta; })
            value.ntheta = input.stage.ntheta;
        if constexpr (requires { value.nzeta; })
            value.nzeta = input.stage.nzeta;
    }

    template <typename Result>
    std::function<void(std::string, Result)> collect(
        Result IterationResult::* member) {
        return [self = shared_from_this(), member](std::string error,
                                                   Result value) {
            if (!error.empty()) self->error = std::move(error);
            self->result.*member = std::move(value);
        };
    }

    void geometry() {
        BaseGeometryCase in;
        in.radius_reference = input.stage.radius_reference;
        in.device_control = input.geometry_control && !probe_only;
        in.axisymmetric = input.stage.ntor == 0;
        in.readback_values = !input.geometry_control || !input.compact_fields ||
                             input.refresh_preconditioner;
        shape(in);
        in.delta_s = input.stage.profiles.delta_s;
        in.double_single = input.double_single;
        in.device_geometry = inverse_.device_geometry;
        in.sqrt_s_f = input.stage.profiles.sqrt_s_f;
        in.sqrt_s_h = input.stage.profiles.sqrt_s_h;
        in.readback = {
            batch, [self = shared_from_this()](BaseGeometryResult value) {
                self->geometry_ = std::move(value);
                if (self->probe_only) {
                    self->probe_result.inverse = self->inverse_.device_geometry;
                    enqueue_geometry_control(
                        self->device, self->geometry_.device_fields,
                        self->input.double_single, true,
                        self->input.stage.ntheta, self->batch,
                        [self](std::string error, GeometryControlResult) {
                            if (!error.empty()) self->error = std::move(error);
                        },
                        [self](DeviceFields fields) {
                            const auto encoder =
                                self->device.CreateCommandEncoder();
                            encoder.CopyBufferToBuffer(
                                fields.buffer, fields.high_offset,
                                self->probe_result.control.buffer,
                                self->probe_result.control.high_offset,
                                8 * sizeof(float));
                            const auto commands = encoder.Finish();
                            self->device.GetQueue().Submit(1, &commands);
                        });
                }
                self->magnetic();
            }};
        enqueue_base_geometry(device, in, collect(&IterationResult::geometry));
    }

    void magnetic() {
        MagneticFieldCase in;
        // CPU force normalization only consumes fields on refresh passes.
        in.readback_values =
            !input.compact_fields || input.refresh_preconditioner;
        shape(in);
        const auto& p = input.stage.profiles;
        in.lamscale = p.lamscale;
        in.lamscale_lo = p.lamscale_lo;
        in.prescribed_current = input.stage.prescribed_current;
        in.double_single = input.double_single;
        in.device_geometry = inverse_.device_geometry;
        in.device_base_geometry = geometry_.device_fields;
        in.sqrt_s_h = p.sqrt_s_h;
        in.sqrt_s_h_lo = p.sqrt_s_h_lo;
        in.phip_f = p.phip_f;
        in.phip_f_lo = p.phip_f_lo;
        in.chip_h = p.chip_h;
        in.chip_h_lo = p.chip_h_lo;
        in.pres_h = p.pres_h;
        in.pres_h_lo = p.pres_h_lo;
        in.curr_h = p.curr_h;
        in.curr_h_lo = p.curr_h_lo;
        in.phip_h = p.phip_h;
        in.phip_h_lo = p.phip_h_lo;
        in.iota_h = p.iota_h;
        in.iota_h_lo = p.iota_h_lo;
        in.readback = {
            batch, [self = shared_from_this()](MagneticFieldResult value) {
                self->magnetic_ = std::move(value);
                if (self->probe_only)
                    self->probe_result.magnetic = self->magnetic_.device_fields;
                self->force();
            }};
        enqueue_magnetic_field(device, in, collect(&IterationResult::magnetic));
    }

    void force() {
        AxisymmetricForceCase in;
        in.radius_reference = input.stage.radius_reference;
        shape(in);
        const auto& p = input.stage.profiles;
        in.delta_s = p.delta_s;
        in.delta_s_lo = p.delta_s_lo;
        in.lamscale = p.lamscale;
        in.lamscale_lo = p.lamscale_lo;
        in.double_single = input.double_single;
        in.readback = false;
        in.device_geometry = inverse_.device_geometry;
        in.device_base_geometry = geometry_.device_fields;
        in.device_magnetic_field = magnetic_.device_fields;
        in.sqrt_s_f = p.sqrt_s_f;
        in.sqrt_s_f_lo = p.sqrt_s_f_lo;
        in.sqrt_s_h = p.sqrt_s_h;
        in.sqrt_s_h_lo = p.sqrt_s_h_lo;
        in.phip_f = p.phip_f;
        in.phip_f_lo = p.phip_f_lo;
        const auto self = shared_from_this();
        if (prefix_only) {
            in.readback_lcfs = input.compact_vacuum;
            in.batched_readback = {batch,
                                   [self](AxisymmetricForceResult value) {
                                       self->force_ = std::move(value);
                                   }};
            enqueue_axisymmetric_force(device, in,
                                       collect(&IterationResult::force));
            return;
        }
        enqueue_axisymmetric_force(
            device, in,
            [self](std::string error, AxisymmetricForceResult value) {
                if (!error.empty()) {
                    self->error = std::move(error);
                    return;
                }
                self->force_ = value;
                self->result.force = value;
                self->snapshot_fields(value.device_fields,
                                      self->result.force.fields,
                                      self->result.force.fields_lo);
                self->forward(value.device_fields, 0);
            });
    }

    void forward(const DeviceFields& fields, int index) {
        ToroidalForwardCase in;
        shape(in);
        in.nfp = input.stage.nfp;
        in.double_single = input.double_single;
        in.use_fft = input.use_fft;
        in.optimized_fft = input.optimized_fft;
        in.canonical_zeta = input.canonical_zeta;
        in.include_lcfs = input.include_lcfs;
        in.readback = false;
        in.device_fields = fields;
        if (index == 1 && input.stage.ntor == 0) {
            // Axisymmetric constraints append four planes after ten forces;
            // the separable projector consumes sixteen force planes first.
            const auto plane = static_cast<std::uint64_t>(input.stage.ns) *
                               input.stage.ntheta * sizeof(float);
            const auto bytes = 20 * plane;
            const auto expanded = detail::cached_buffer(
                device, bytes * (input.double_single ? 2 : 1),
                wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                "axisymmetric constraint projector fields");
            const auto encoder = device.CreateCommandEncoder();
            encoder.ClearBuffer(expanded);
            for (int word = 0; word < (input.double_single ? 2 : 1); ++word) {
                const auto source =
                    word ? fields.low_offset : fields.high_offset;
                encoder.CopyBufferToBuffer(fields.buffer, source, expanded,
                                           word * bytes, 10 * plane);
                encoder.CopyBufferToBuffer(fields.buffer, source + 10 * plane,
                                           expanded, word * bytes + 16 * plane,
                                           4 * plane);
            }
            const auto commands = encoder.Finish();
            device.GetQueue().Submit(1, &commands);
            in.device_fields = {expanded,
                                static_cast<std::size_t>(bytes / sizeof(float)),
                                0, bytes};
        }
        const auto self = shared_from_this();
        enqueue_toroidal_forward(
            device, in,
            [self, index](std::string error, ToroidalForwardResult value) {
                if (!error.empty()) {
                    self->error = std::move(error);
                    return;
                }
                self->result.forward[index] = value;
                self->snapshot_fields(value.device_residual,
                                      self->result.forward[index].residual,
                                      self->result.forward[index].residual_lo);
                self->decompose(value.device_residual, index);
            });
    }

    void decompose(const DeviceFields& fields, int index) {
        ResidualDecompositionCase in;
        shape(in);
        in.double_single = input.double_single;
        in.device_residual = fields;
        in.zero_m1_z = index == 0 || input.zero_m1_z;
        in.include_edge_rz = input.include_edge_invariant;
        in.readback_values = !input.compact_norms;
        in.sqrt_s_f = input.stage.profiles.sqrt_s_f;
        in.sqrt_s_f_lo = input.stage.profiles.sqrt_s_f_lo;
        const auto self = shared_from_this();
        in.readback = {batch, [self, index](ResidualDecompositionResult value) {
                           self->norm(
                               value.device_residual, index,
                               [self, index, fields = value.device_residual] {
                                   if (index == 0)
                                       self->elements();
                                   else
                                       self->apply(fields);
                               });
                       }};
        enqueue_residual_decomposition(
            device, in,
            [self, index](std::string error,
                          ResidualDecompositionResult value) {
                if (!error.empty()) self->error = std::move(error);
                self->result.residual[index] = std::move(value);
            });
    }

    void elements() {
        if (!input.refresh_preconditioner) {
            elements_ = input.elements;
            matrix_ = input.matrix;
            constraint();
            return;
        }
        AxisymmetricPreconditionerElementCase in;
        shape(in);
        in.delta_s = input.stage.profiles.delta_s;
        in.free_boundary = input.include_lcfs;
        in.device_geometry = inverse_.device_geometry;
        in.device_base_geometry = geometry_.device_fields;
        in.device_magnetic_field = magnetic_.device_fields;
        in.sqrt_s_f = input.stage.profiles.sqrt_s_f;
        in.sqrt_s_h = input.stage.profiles.sqrt_s_h;
        in.readback = {batch, [self = shared_from_this()](
                                  AxisymmetricPreconditionerElements value) {
                           self->elements_ = std::move(value);
                           self->matrix();
                       }};
        enqueue_axisymmetric_preconditioner_elements(
            device, in, collect(&IterationResult::elements));
    }

    void matrix() {
        AxisymmetricPreconditionerMatrixCase in;
        shape(in);
        in.nfp = input.stage.nfp;
        in.delta_s = input.stage.profiles.delta_s;
        in.free_boundary = input.include_lcfs;
        in.elements = elements_;
        in.device_base_geometry = geometry_.device_fields;
        in.sqrt_s_f = input.stage.profiles.sqrt_s_f;
        in.phip_h = input.stage.profiles.phip_h;
        in.readback = {batch, [self = shared_from_this()](
                                  AxisymmetricPreconditionerMatrix value) {
                           self->matrix_ = std::move(value);
                           self->constraint();
                       }};
        enqueue_axisymmetric_preconditioner_matrix(
            device, in, collect(&IterationResult::matrix));
    }

    void constraint() {
        AxisymmetricConstraintCase in;
        shape(in);
        in.readback_intermediates = !input.compact_fields;
        in.delta_s = input.stage.profiles.delta_s;
        in.tcon0 = input.stage.tcon0;
        in.double_single = input.double_single;
        in.readback = false;
        in.reset_reference = input.reset_reference;
        in.refresh_preconditioner = input.refresh_preconditioner;
        in.device_geometry = inverse_.device_geometry;
        in.device_r_con = inverse_.device_r_con;
        in.device_z_con = inverse_.device_z_con;
        in.device_force_fields = force_.device_fields;
        if (input.stage.ntor == 0)
            in.device_force_fields =
                field_slice(force_.device_fields, 0,
                            10 * static_cast<std::size_t>(in.ns) * in.ntheta);
        in.device_elements = elements_.device_elements;
        in.device_r_con0 = input.device_r_con0;
        in.device_z_con0 = input.device_z_con0;
        in.ard = elements_.ard;
        in.azd = elements_.azd;
        in.r_con0 = input.r_con0;
        in.r_con0_lo = input.r_con0_lo;
        in.z_con0 = input.z_con0;
        in.z_con0_lo = input.z_con0_lo;
        in.tcon = input.tcon;
        const auto points =
            static_cast<std::size_t>(in.ns) * in.ntheta * in.nzeta;
        if (in.r_con0.empty() && !in.device_r_con0) {
            in.r_con0.assign(points, 0.0F);
            in.z_con0.assign(points, 0.0F);
            in.r_con0_lo.assign(points, 0.0F);
            in.z_con0_lo.assign(points, 0.0F);
            in.tcon.assign(in.ns, 0.0F);
        }
        in.sqrt_s_f = input.stage.profiles.sqrt_s_f;
        in.sqrt_s_f_lo = input.stage.profiles.sqrt_s_f_lo;
        in.batched_readback = {
            batch,
            [self = shared_from_this()](AxisymmetricConstraintResult value) {
                self->snapshot_fields(value.device_fields,
                                      self->result.constraint.fields,
                                      self->result.constraint.fields_lo);
                self->forward(value.device_fields, 1);
            }};
        enqueue_axisymmetric_constraint(device, in,
                                        collect(&IterationResult::constraint));
    }

    void apply(const DeviceFields& residual) {
        AxisymmetricPreconditionerApplyCase in;
        shape(in);
        in.elements = elements_;
        in.matrix = matrix_;
        in.device_residual = residual;
        in.include_lcfs = input.include_lcfs;
        in.readback = {batch, [self = shared_from_this()](
                                  AxisymmetricPreconditionerApplyResult value) {
                           if (self->probe_only)
                               self->probe_result.preconditioned =
                                   value.device_residual;
                           self->norm(value.device_residual, 2, [] {});
                       }};
        enqueue_axisymmetric_preconditioner_apply(
            device, in, collect(&IterationResult::preconditioned));
    }

    void norm(const DeviceFields& fields,
              int index,
              std::function<void()> next) {
        if (!input.shadow_norms && !input.compact_norms) {
            next();
            return;
        }
        ResidualNormCase in;
        in.residual = fields;
        in.ns = input.stage.ns;
        in.paired = index != 2 && input.double_single;
        in.include_edge_rz = index == 2 || input.include_edge_invariant;
        in.readback = {
            batch, [self = shared_from_this(), index,
                    next = std::move(next)](ResidualNormResult value) {
                if (self->probe_only) {
                    const auto encoder = self->device.CreateCommandEncoder();
                    encoder.CopyBufferToBuffer(
                        value.device_norm.buffer, 0,
                        self->probe_result.control.buffer,
                        self->probe_result.control.high_offset +
                            (8 + 9 * index) * sizeof(float),
                        9 * sizeof(float));
                    const auto commands = encoder.Finish();
                    self->device.GetQueue().Submit(1, &commands);
                }
                next();
            }};
        enqueue_residual_norm(
            device, in,
            [self = shared_from_this(), index](std::string error,
                                               ResidualNormResult value) {
                if (!error.empty()) self->error = std::move(error);
                if (self->input.compact_norms) {
                    if (!value.finite)
                        value.raw.fill(
                            std::numeric_limits<double>::quiet_NaN());
                    if (index == 2)
                        self->result.preconditioned.raw_norm = value.raw;
                    else
                        self->result.residual[index].raw_norm = value.raw;
                }
                self->result.norms[index] = std::move(value);
            });
    }
};

}  // namespace

std::uint64_t iteration_readback_capacity(const AxisymmetricStageData& stage,
                                          bool readback_intermediates) {
    const auto points = std::uint64_t(stage.ns) * stage.ntheta * stage.nzeta;
    const auto spectral =
        std::uint64_t(stage.ns) * stage.mpol * (stage.ntor + 1);
    // Paired inverse (40), geometry (20), magnetic (10), constraint (7),
    // two residual snapshots (36 spectral), preconditioner and descent slack.
    // Verification also snapshots paired forces, constraint fields, and both
    // projections before their shared scratch is overwritten.
    const int extra_fields =
        readback_intermediates
            ? 2 * (FORCE_FIELD_COUNT + TOROIDAL_FORWARD_FIELD_COUNT)
            : 0;
    const int extra_spectral = readback_intermediates ? 24 : 0;
    return sizeof(float) *
               (((stage.free_boundary ? 112 : 80) + extra_fields) * points +
                (80 + extra_spectral) * spectral + 32 * stage.ns) +
           256;
}

void enqueue_iteration(const wgpu::Device& device,
                       IterationCase input,
                       const std::shared_ptr<ReadbackBatch>& batch,
                       IterationCallback callback) {
    auto dispatch = std::make_shared<IterationDispatch>();
    dispatch->device = device;
    dispatch->input = std::move(input);
    dispatch->batch = batch;
    dispatch->callback = std::move(callback);
    dispatch->start();
}

void enqueue_iteration_prefix(const wgpu::Device& device,
                              IterationCase input,
                              const std::shared_ptr<ReadbackBatch>& batch,
                              IterationPrefixCallback callback) {
    auto dispatch = std::make_shared<IterationDispatch>();
    dispatch->device = device;
    dispatch->input = std::move(input);
    dispatch->batch = batch;
    dispatch->prefix_only = true;
    // The continuation retains the device views while the host checks the
    // Jacobian and performs NESTOR. It is discarded on a rejected geometry.
    const std::weak_ptr<IterationDispatch> weak = dispatch;
    dispatch->callback = [weak, callback = std::move(callback)](
                             std::string error, IterationResult result) {
        auto pending = weak.lock();
        callback(std::move(error), std::move(result),
                 [pending](IterationCase next, DeviceFields force,
                           IterationCallback complete) {
                     pending->resume(std::move(next), std::move(force),
                                     std::move(complete));
                 });
    };
    dispatch->start();
}

IterationProbeResult enqueue_iteration_probe(const wgpu::Device& device,
                                             IterationCase input,
                                             const DeviceFields& control) {
    if (!input.double_single || input.stage.free_boundary ||
        input.stage.ntor != 0 || input.stage.nzeta != 1 ||
        input.refresh_preconditioner || input.reset_reference ||
        input.include_lcfs || input.include_edge_invariant ||
        !input.device_state || !input.elements.device_elements ||
        !input.matrix.device_matrix || !control || control.values != 35 ||
        control.high_offset % sizeof(float) != 0 ||
        control.high_offset > control.buffer.GetSize() ||
        35 * sizeof(float) > control.buffer.GetSize() - control.high_offset)
        throw std::invalid_argument("invalid frozen Newton probe epoch");
    auto dispatch = std::make_shared<IterationDispatch>();
    dispatch->device = device;
    dispatch->input = std::move(input);
    dispatch->input.compact_fields = true;
    dispatch->input.compact_norms = true;
    dispatch->input.readback_intermediates = false;
    dispatch->probe_only = true;
    dispatch->probe_result.control = control;
    dispatch->batch = std::make_shared<ReadbackBatch>(device, 0, false);
    dispatch->start();
    if (!dispatch->error.empty()) throw std::runtime_error(dispatch->error);
    if (!dispatch->probe_result.preconditioned)
        throw std::runtime_error("Newton probe did not publish device outputs");
    return dispatch->probe_result;
}

}  // namespace cumes::webgpu
