#include "cumes/webgpu/iteration.hpp"

#include <limits>
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

    void start() {
        ToroidalInverseCase inverse;
        shape(inverse);
        inverse.nfp = input.stage.nfp;
        inverse.double_single = input.double_single;
        inverse.state = input.stage.state;
        inverse.state_lo = input.stage.state_lo;
        inverse.device_state = input.device_state;
        inverse.readback = {
            batch, [self = shared_from_this()](ToroidalInverseResult value) {
                self->inverse_ = std::move(value);
                self->geometry();
            }};
        enqueue_toroidal_inverse(device, inverse,
                                 collect(&IterationResult::inverse));
        const auto self = shared_from_this();
        // All device-ready callbacks above are synchronous. The whole DAG is
        // submitted before this single map; decode callbacks only store values.
        batch->map([self](std::string mapping_error) {
            if (self->error.empty()) self->error = std::move(mapping_error);
            self->callback(std::move(self->error), std::move(self->result));
        });
    }

   private:
    ToroidalInverseResult inverse_;
    BaseGeometryResult geometry_;
    MagneticFieldResult magnetic_;
    AxisymmetricForceResult force_;
    AxisymmetricPreconditionerElements elements_;
    AxisymmetricPreconditionerMatrix matrix_;

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
        shape(in);
        in.delta_s = input.stage.profiles.delta_s;
        in.double_single = input.double_single;
        in.device_geometry = inverse_.device_geometry;
        in.sqrt_s_f = input.stage.profiles.sqrt_s_f;
        in.sqrt_s_h = input.stage.profiles.sqrt_s_h;
        in.readback = {batch,
                       [self = shared_from_this()](BaseGeometryResult value) {
                           self->geometry_ = std::move(value);
                           self->magnetic();
                       }};
        enqueue_base_geometry(device, in, collect(&IterationResult::geometry));
    }

    void magnetic() {
        MagneticFieldCase in;
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
        in.readback = {batch,
                       [self = shared_from_this()](MagneticFieldResult value) {
                           self->magnetic_ = std::move(value);
                           self->force();
                       }};
        enqueue_magnetic_field(device, in, collect(&IterationResult::magnetic));
    }

    void force() {
        AxisymmetricForceCase in;
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
        enqueue_axisymmetric_force(
            device, in,
            [self](std::string error, AxisymmetricForceResult value) {
                if (!error.empty()) {
                    self->error = std::move(error);
                    return;
                }
                self->force_ = value;
                self->result.force = value;
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
        in.readback = false;
        in.device_fields = fields;
        const auto self = shared_from_this();
        enqueue_toroidal_forward(
            device, in,
            [self, index](std::string error, ToroidalForwardResult value) {
                if (!error.empty()) {
                    self->error = std::move(error);
                    return;
                }
                self->result.forward[index] = value;
                self->decompose(value.device_residual, index);
            });
    }

    void decompose(const DeviceFields& fields, int index) {
        ResidualDecompositionCase in;
        shape(in);
        in.double_single = input.double_single;
        in.device_residual = fields;
        in.zero_m1_z = index == 0 || input.zero_m1_z;
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
        if (in.r_con0.empty()) {
            in.r_con0.assign(points, 0.0F);
            in.z_con0.assign(points, 0.0F);
            in.r_con0_lo.assign(points, 0.0F);
            in.z_con0_lo.assign(points, 0.0F);
            in.tcon.assign(in.ns, 0.0F);
        }
        in.sqrt_s_f = input.stage.profiles.sqrt_s_f;
        in.sqrt_s_f_lo = input.stage.profiles.sqrt_s_f_lo;
        in.batched_readback = {batch, [self = shared_from_this()](
                                          AxisymmetricConstraintResult value) {
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
        in.readback = {batch, [self = shared_from_this()](
                                  AxisymmetricPreconditionerApplyResult value) {
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
        in.include_edge_rz = index == 2;
        in.readback = {
            batch, [next = std::move(next)](ResidualNormResult) { next(); }};
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

std::uint64_t iteration_readback_capacity(const AxisymmetricStageData& stage) {
    const auto points = std::uint64_t(stage.ns) * stage.ntheta * stage.nzeta;
    const auto spectral =
        std::uint64_t(stage.ns) * stage.mpol * (stage.ntor + 1);
    // Paired inverse (40), geometry (20), magnetic (10), constraint (7),
    // two residual snapshots (36 spectral), preconditioner and descent slack.
    return sizeof(float) * (80 * points + 80 * spectral + 32 * stage.ns) + 256;
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

}  // namespace cumes::webgpu
