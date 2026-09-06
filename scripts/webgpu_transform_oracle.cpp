// Offline, diagnostic-only W7-X oracle for webgpu_transform_capture.js.
// Build with g++-12 -std=c++20 -O2 -ffp-contract=off. Boost binary128
// arithmetic and trigonometry are independent of the WGSL paired arithmetic/FFT
// emitter. Usage: oracle PRIMARY.bin GENERIC.bin DIRECT.bin CANONICAL.bin
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/multiprecision/cpp_bin_float.hpp>

namespace {
using Quad = boost::multiprecision::cpp_bin_float_quad;
constexpr int NS = 99, MPOL = 12, NN = 13, NT = 30, NZ = 36, TR = 16;
constexpr int S = 3, MODES = MPOL * NN;
constexpr std::array<int, S> SURFACES{1, 50, 97};
constexpr int FIELD_VALUES = 20 * S * NT * NZ;
constexpr int BASIS_VALUES = 2 * MPOL * NT + 2 * NN * NZ;
constexpr int RESIDUAL_VALUES = 6 * MODES * NS;
constexpr int PROJECTED_VALUES = 40 * S * TR * NN;
constexpr int PRIMARY_PREFIX = 2 * FIELD_VALUES + 2 * BASIS_VALUES + 12;

std::vector<float> read_values(const std::string& path,
                               int count,
                               bool allow_primary_prefix = false) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const int prefix =
        allow_primary_prefix && input &&
                input.tellg() ==
                    std::streamoff((count + PRIMARY_PREFIX) * sizeof(float))
            ? PRIMARY_PREFIX
            : 0;
    if (!input ||
        input.tellg() != std::streamoff((count + prefix) * sizeof(float)))
        throw std::runtime_error("Unexpected capture size: " + path);
    std::vector<float> values(count);
    input.seekg(prefix * sizeof(float));
    input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    if (!input) throw std::runtime_error("Cannot read capture: " + path);
    for (float value : values)
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite capture");
    return values;
}

int projected_index(int family, int field, int surface, int theta, int n) {
    return (((family * 20 + field) * S + surface) * TR + theta) * NN + n;
}

template <typename T>
std::vector<T> project(std::span<const float> input, bool legacy_zeta) {
    using std::acos;
    using std::cos;
    using std::sin;
    const T pi = acos(T(-1));
    std::array<T, 2 * NN * NZ> roots;
    const int basis = 2 * FIELD_VALUES + 2 * MPOL * NT;
    for (int n = 0; n < NN; ++n)
        for (int z = 0; z < NZ; ++z) {
            const T angle = 2 * pi * n * z / NZ;
            for (int family = 0; family < 2; ++family) {
                const int i = (family * NN + n) * NZ + z;
                roots[i] = legacy_zeta ? T(input[basis + i]) +
                                             T(input[basis + BASIS_VALUES + i])
                           : family == 0 ? T(cos(angle))
                                         : T(sin(angle));
            }
        }
    std::vector<T> output(PROJECTED_VALUES);
    for (int field = 0; field < 20; ++field)
        for (int surface = 0; surface < S; ++surface)
            for (int theta = 0; theta < TR; ++theta)
                for (int n = 0; n < NN; ++n) {
                    T cosine = 0, sine = 0;
                    for (int z = 0; z < NZ; ++z) {
                        const int i =
                            ((field * S + surface) * NZ + z) * NT + theta;
                        const T value =
                            T(input[i]) + T(input[FIELD_VALUES + i]);
                        cosine += value * roots[n * NZ + z];
                        sine += value * roots[(NN + n) * NZ + z];
                    }
                    output[projected_index(0, field, surface, theta, n)] =
                        cosine;
                    output[projected_index(1, field, surface, theta, n)] = sine;
                }
    return output;
}

template <typename T>
std::vector<T> poloidal(std::span<const float> input,
                        const std::vector<T>& projected,
                        bool ideal_basis) {
    using std::acos;
    using std::cos;
    using std::sin;
    using std::sqrt;
    const T pi = acos(T(-1));
    const auto params = input.subspan(PRIMARY_PREFIX - 12, 12);
    const T normalization =
        ideal_basis ? T(1) / (NZ * (TR - 1)) : T(params[8]) + T(params[9]);
    const T sqrt_two =
        ideal_basis ? T(sqrt(T(2))) : T(params[10]) + T(params[11]);
    std::vector<T> output(6 * MODES * S);
    for (int mode = 0; mode < MODES; ++mode) {
        const int m = mode / NN, n = mode % NN, parity = m % 2;
        const int mf = m, nf = n * 5, xmpq = m * (m - 1);
        for (int surface = 0; surface < S; ++surface) {
            std::array<T, 6> sums{};
            for (int theta = 0; theta < TR; ++theta) {
                const auto p = [&](int family, int field) -> const T& {
                    return projected[projected_index(family, field, surface,
                                                     theta, n)];
                };
                T weight = normalization;
                if (theta == 0 || theta + 1 == TR) weight /= 2;
                const int i = 2 * FIELD_VALUES + m * NT + theta;
                const T cm = ideal_basis
                                 ? T(cos(2 * pi * m * theta / NT))
                                 : T(input[i]) + T(input[i + BASIS_VALUES]);
                const T sm = ideal_basis
                                 ? T(sin(2 * pi * m * theta / NT))
                                 : T(input[i + MPOL * NT]) +
                                       T(input[i + MPOL * NT + BASIS_VALUES]);
                const T trc = p(0, parity) + xmpq * p(0, 16 + parity);
                const T trs = p(1, parity) + xmpq * p(1, 16 + parity);
                const T tzc = p(0, 2 + parity) + xmpq * p(0, 18 + parity);
                const T tzs = p(1, 2 + parity) + xmpq * p(1, 18 + parity);
                sums[0] += weight * (trc * cm - mf * p(0, 4 + parity) * sm +
                                     nf * p(1, 10 + parity) * cm);
                sums[3] += weight * (trs * sm + mf * p(1, 4 + parity) * cm -
                                     nf * p(0, 10 + parity) * sm);
                sums[1] += weight * (tzc * sm + mf * p(0, 6 + parity) * cm +
                                     nf * p(1, 12 + parity) * sm);
                sums[4] += weight * (tzs * cm - mf * p(1, 6 + parity) * sm -
                                     nf * p(0, 12 + parity) * cm);
                sums[2] += weight * (mf * p(0, 8 + parity) * cm +
                                     nf * p(1, 14 + parity) * sm);
                sums[5] -= weight * (mf * p(1, 8 + parity) * sm +
                                     nf * p(0, 14 + parity) * cm);
            }
            T scale = 1;
            if (m != 0) scale *= sqrt_two;
            if (n != 0) scale *= sqrt_two;
            for (int c = 0; c < 6; ++c)
                output[(c * MODES + mode) * S + surface] = scale * sums[c];
        }
    }
    return output;
}

struct Metrics {
    long double max_abs = 0, error2 = 0, norm2 = 0;
    std::size_t worst = 0, count = 0;
    void add(long double error, long double reference) {
        error = std::abs(error);
        if (error > max_abs) {
            max_abs = error;
            worst = count;
        }
        error2 += error * error;
        norm2 += reference * reference;
        ++count;
    }
    void print() const {
        std::cout << "{\"max_abs\":" << max_abs << ",\"relative_l2\":"
                  << std::sqrt(error2 / std::max(norm2, 1e-300L))
                  << ",\"worst_index\":" << worst << ",\"count\":" << count
                  << '}';
    }
};

void self_test() {
    std::vector<float> input(PRIMARY_PREFIX);
    const long double pi = std::acos(-1.0L);
    const auto put = [&](int field, int surface, int theta, int zeta,
                         long double value) {
        const int i = ((field * S + surface) * NZ + zeta) * NT + theta;
        input[i] = static_cast<float>(value);
        input[FIELD_VALUES + i] = static_cast<float>(value - input[i]);
    };
    for (int surface = 0; surface < S; ++surface)
        for (int theta = 0; theta < NT; ++theta)
            for (int zeta = 0; zeta < NZ; ++zeta) {
                put(0, surface, theta, zeta, 1.0L);
                put(8, surface, theta, zeta,
                    std::cos(4 * pi * theta / NT) *
                        std::cos(6 * pi * zeta / NZ));
                put(14, surface, theta, zeta,
                    std::sin(4 * pi * theta / NT) *
                        std::sin(6 * pi * zeta / NZ));
            }
    const auto values =
        poloidal<Quad>(input, project<Quad>(input, false), true);
    long double error = 0;
    for (int c = 0; c < 6; ++c)
        for (int mode = 0; mode < MODES; ++mode)
            for (int surface = 0; surface < S; ++surface) {
                // Constant R: unit projection. Lambda m=2,n=3 receives
                // m/2 + n*nfp/2 = 8.5 from its cosine/sine derivative terms.
                const long double expected = c == 0 && mode == 0 ? 1.0L
                                             : c == 2 && mode == 2 * NN + 3
                                                 ? 8.5L
                                                 : 0.0L;
                error = std::max(
                    error,
                    std::abs(static_cast<long double>(
                                 values[(c * MODES + mode) * S + surface]) -
                             expected));
            }
    if (error > 3e-13L)
        throw std::runtime_error("Analytic oracle self-test failed");
    std::cout << "Analytic constant/harmonic oracle PASS: " << error << '\n';
}

void compare(std::span<const float> actual,
             const std::vector<Quad>& projected,
             const std::vector<Quad>& residual,
             const std::vector<Quad>& ideal) {
    Metrics zeta, shared_theta, mathematical;
    for (int i = 0; i < PROJECTED_VALUES; ++i) {
        const Quad value =
            Quad(actual[2 * RESIDUAL_VALUES + i]) +
            Quad(actual[2 * RESIDUAL_VALUES + PROJECTED_VALUES + i]);
        zeta.add(static_cast<long double>(value - projected[i]),
                 static_cast<long double>(projected[i]));
    }
    for (int c = 0; c < 6; ++c)
        for (int mode = 0; mode < MODES; ++mode)
            for (int surface = 0; surface < S; ++surface) {
                const int i = (c * MODES + mode) * NS + SURFACES[surface];
                const int j = (c * MODES + mode) * S + surface;
                const Quad value =
                    Quad(actual[i]) + Quad(actual[RESIDUAL_VALUES + i]);
                shared_theta.add(static_cast<long double>(value - residual[j]),
                                 static_cast<long double>(residual[j]));
                mathematical.add(static_cast<long double>(value - ideal[j]),
                                 static_cast<long double>(ideal[j]));
            }
    std::cout << "{\"zeta\":";
    zeta.print();
    std::cout << ",\"shared_theta_residual\":";
    shared_theta.print();
    std::cout << ",\"mathematical_residual\":";
    mathematical.print();
    std::cout << '}';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc != 5)
            throw std::runtime_error(
                "Pass primary/generic/direct/canonical captures");
        static_assert(std::numeric_limits<Quad>::digits == 113);
        static_assert(std::endian::native == std::endian::little);
        constexpr int OUTPUT_VALUES = 2 * (RESIDUAL_VALUES + PROJECTED_VALUES);
        const auto primary =
            read_values(argv[1], PRIMARY_PREFIX + OUTPUT_VALUES);
        const auto params = std::span(primary).subspan(PRIMARY_PREFIX - 12, 12);
        constexpr std::array<unsigned, 8> SHAPE{NS, MPOL, NN - 1,  NT,
                                                NZ, 5,    NT * NZ, 0};
        for (std::size_t i = 0; i < SHAPE.size(); ++i)
            if (std::bit_cast<unsigned>(params[i]) != SHAPE[i])
                throw std::runtime_error(
                    "Capture shape does not match W7-X oracle");
        const auto projected = project<Quad>(primary, false);
        const auto residual = poloidal<Quad>(primary, projected, false);
        const auto ideal = poloidal<Quad>(primary, projected, true);
        const auto legacy_projected = project<Quad>(primary, true);
        const auto legacy_residual =
            poloidal<Quad>(primary, legacy_projected, false);
        // Recompute independently at native extended precision as an oracle
        // sensitivity check; both have substantially more precision than f32
        // pairs.
        static_assert(std::numeric_limits<long double>::digits >= 64);
        const auto extended = poloidal<long double>(
            primary, project<long double>(primary, false), false);
        Metrics agreement;
        for (std::size_t i = 0; i < residual.size(); ++i)
            agreement.add(
                static_cast<long double>(Quad(extended[i]) - residual[i]),
                static_cast<long double>(residual[i]));
        std::cout << std::setprecision(17)
                  << "{\"oracle_bits\":113,\"extended_agreement\":";
        agreement.print();
        std::cout << ",\"variants\":[";
        for (int variant = 0; variant < 4; ++variant) {
            const auto values = variant == 0 ? std::vector<float>{}
                                             : read_values(argv[variant + 1],
                                                           OUTPUT_VALUES, true);
            const auto actual = variant == 0
                                    ? std::span(primary).subspan(PRIMARY_PREFIX)
                                    : std::span<const float>(values);
            if (variant) std::cout << ',';
            compare(actual, projected, residual, ideal);
        }
        const auto direct = read_values(argv[3], OUTPUT_VALUES, true);
        std::cout << "],\"direct_against_own_roots\":";
        compare(direct, legacy_projected, legacy_residual, ideal);
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
