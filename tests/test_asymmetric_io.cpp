// Typed asymmetric input provenance, including large and empty harmonic lists.
// Manufactured host state keeps this serialization gate independent of a GPU.
#include "cumes/config/validated_problem.hpp"
#include "cumes/io/checkpoint.hpp"
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "cumes_test.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>
#ifdef CUMES_HAVE_HDF5
#include <hdf5.h>
#endif
#ifdef CUMES_HAVE_NETCDF
#include <netcdf.h>
#endif

using namespace cumes;
using test::check;
namespace fs = std::filesystem;

static ProblemSpec make_spec(bool empty) {
    ProblemSpec spec;
    spec.lasym = true;
    spec.mpol = 3;
    spec.ntor = 1;
    spec.nfp = 2;
    spec.toroidal_flux.coefficients = {1.0};
    spec.stages = {{5, 100, 1e-10}};
    spec.rbc = {{0, 0, 4.0}, {1, 0, 1.0}};
    spec.zbs = {{1, 0, 1.0}};
    spec.raxis_c = {4.0};
    spec.raxis_s = {0.0, 0.002};
    spec.zaxis_c = {0.03, -0.001};
    if (!empty) {
        spec.rbs = {{1, 1, 0.01}, {2, -1, 0.002}, {1, 1, -0.003}};
        spec.zbc = {{0, 0, 0.04}, {1, -1, 0.03}, {0, 1, 0.001}};
        // Each value array alone exceeds 64 KiB. Retain accepted duplicates
        // and explicit zeros even though they do not change the folded shape.
        for (int i = 0; i < 10000; ++i) {
            spec.rbs.push_back({2, -1, 0.0});
            spec.zbc.push_back({1, 1, 0.0});
        }
    }
    return spec;
}

static void check_version(const fs::path& path, int expected) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(8);
    std::int32_t version = 0;
    input.read(reinterpret_cast<char*>(&version), sizeof(version));
    check(input.good() && version == expected, "typed input format version");
}

[[maybe_unused]] static void check_rejected(const fs::path& path,
                                            OutputFormat format) {
    RunReport report;
    report.total_effective_iterations = 99;
    auto result = make_reader(format)->read(path.string(), std::ref(report));
    check(!result.has_value(), "partial complementary input is rejected");
    check(report.total_effective_iterations == 99,
          "failed container read preserves the caller's report");
}

static void run_case(const fs::path& directory, bool empty) {
    const auto spec = make_spec(empty);
    auto problem = validate(spec, SolverOptions{});
    check(problem.has_value(), "asymmetric I/O fixture validates");
    if (!problem.has_value()) {
        for (const auto& message : problem.error().errors())
            std::cerr << message << '\n';
        return;
    }
    const auto& vp = problem.value();
    RunReport report;
    report.build.scalar_type = "double";
    report.build.precision_policy = "verify-double";
    report.stages.resize(1);
    report.stages[0].ns = 5;
    report.stages[0].effective_iterations = 3;
    report.stages[0].converged = true;
    report.stages[0].restarts = {{1}};
    report.input_params = make_input_params(vp);
    const auto& params = report.input_params;
    check(params.rbs_m.size() == spec.rbs.size() &&
              params.zbc_m.size() == spec.zbc.size(),
          "input record retains every raw harmonic");
    check(params.raxis_s == spec.raxis_s && params.zaxis_c == spec.zaxis_c,
          "input record retains complementary axes");
    if (!empty) {
        check(params.rbs_n[1] == -1 && params.rbs_n[2] == 1 &&
                  params.rbs_value[2] == -0.003 &&
                  params.rbs_value.back() == 0.0,
              "signed n, duplicate order, and zero coefficients survive");
    }
    check(params.rbsc == vp.boundary().rbsc &&
              params.rbcs == vp.boundary().rbcs &&
              params.zbcc == vp.boundary().zbcc &&
              params.zbss == vp.boundary().zbss,
          "input record carries all complementary folded families");

    EquilibriumSnapshot snapshot;
    snapshot.ns = 5;
    snapshot.mnmax = 6;
    snapshot.families.resize(12);
    for (int c = 0; c < snapshot.components(); ++c) {
        auto& family = snapshot.families[c];
        family.resize(snapshot.family_size());
        for (size_t i = 0; i < family.size(); ++i)
            family[i] = c * 100.0 + static_cast<double>(i) + 0.125;
    }
    const std::string name = empty ? "empty" : "large";
    for (auto format :
         {OutputFormat::BINARY, OutputFormat::NETCDF, OutputFormat::HDF5}) {
        auto writer = make_writer(format);
        if (!writer) continue;
        const auto path =
            directory / (name + std::string(output_suffix(format)));
        OutputSpec output{format, path.string()};
        auto written = writer->write_atomic(snapshot, report, output, vp);
        check(written.has_value(), "write typed asymmetric input");
        if (!written.has_value()) {
            std::cerr << written.error() << '\n';
            continue;
        }
        RunReport restored;
        auto result =
            make_reader(format)->read(path.string(), std::ref(restored));
        check(result.has_value(), "read typed asymmetric input");
        if (result.has_value()) {
            check(result.value().families == snapshot.families,
                  "all twelve state families round trip exactly");
            check(restored.input_params == params,
                  "complete typed input record round trips exactly");
        } else {
            std::cerr << result.error() << '\n';
        }
        if (format == OutputFormat::BINARY) {
            check_version(path, 10);
            const auto obsolete = directory / "obsolete.bin";
            fs::copy_file(path, obsolete, fs::copy_options::overwrite_existing);
            {
                std::fstream file(
                    obsolete, std::ios::binary | std::ios::in | std::ios::out);
                const std::int32_t version = 9;
                file.seekp(8);
                file.write(reinterpret_cast<const char*>(&version),
                           sizeof(version));
            }
            check_rejected(obsolete, format);
            check(make_reader(format)->read(obsolete.string()).has_value(),
                  "old asymmetric state remains readable without JSON "
                  "provenance");
            fs::remove(obsolete);
        }
#ifdef CUMES_HAVE_HDF5
        if (format == OutputFormat::HDF5) {
            const hid_t file =
                H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            check(file >= 0 && H5Aexists(file, "asymmetric_input_json") == 0 &&
                      H5Aexists(file, "lasym") > 0 &&
                      H5Lexists(file, "rbs_value", H5P_DEFAULT) > 0,
                  "HDF5 embeds typed input without a JSON attribute");
            H5Fclose(file);
            const auto corrupt = directory / "corrupt.h5";
            fs::copy_file(path, corrupt, fs::copy_options::overwrite_existing);
            const hid_t edited =
                H5Fopen(corrupt.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            check(H5Ldelete(edited, "zbc_n", H5P_DEFAULT) >= 0,
                  "remove one complementary input dataset");
            H5Fclose(edited);
            check_rejected(corrupt, format);
            fs::remove(corrupt);
        }
#endif
#ifdef CUMES_HAVE_NETCDF
        if (format == OutputFormat::NETCDF) {
            int file = -1, var = -1;
            check(nc_open(path.c_str(), NC_NOWRITE, &file) == NC_NOERR,
                  "open NetCDF typed input");
            check(nc_inq_att(file, NC_GLOBAL, "asymmetric_input_json", nullptr,
                             nullptr) == NC_ENOTATT &&
                      nc_inq_varid(file, "lasym", &var) == NC_NOERR,
                  "NetCDF embeds typed input without a JSON attribute");
            nc_close(file);
            const auto corrupt = directory / "corrupt.nc";
            fs::copy_file(path, corrupt, fs::copy_options::overwrite_existing);
            nc_open(corrupt.c_str(), NC_WRITE, &file);
            nc_redef(file);
            nc_inq_varid(file, "raxis_s", &var);
            check(nc_rename_var(file, var, "missing_raxis_s") == NC_NOERR,
                  "remove complementary axis variable");
            nc_enddef(file);
            nc_close(file);
            check_rejected(corrupt, format);
            fs::remove(corrupt);
        }
#endif
    }

    const auto path = directory / (name + ".ckpt");
    check(write_checkpoint(snapshot, params, path.string()).has_value(),
          "write typed asymmetric checkpoint");
    check_version(path, 8);
    InputParams restored;
    auto result = read_checkpoint(path.string(), std::ref(restored));
    check(result.has_value() && result.value().families == snapshot.families &&
              restored == params,
          "checkpoint state and typed input round trip");
    const auto corrupt = directory / "corrupt.ckpt";
    fs::copy_file(path, corrupt, fs::copy_options::overwrite_existing);
    fs::resize_file(corrupt, fs::file_size(corrupt) - sizeof(double));
    check(!read_checkpoint(corrupt.string(), std::ref(restored)).has_value(),
          "truncated complementary input is rejected");
    check(read_checkpoint(corrupt.string()).has_value(),
          "state-only restart does not require the input record");
    fs::copy_file(path, corrupt, fs::copy_options::overwrite_existing);
    {
        std::fstream file(corrupt,
                          std::ios::binary | std::ios::in | std::ios::out);
        const std::int32_t count = (1 << 20) + 1;
        file.seekp(-static_cast<std::streamoff>(4 + 6 * sizeof(double)),
                   std::ios::end);
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    }
    check(!read_checkpoint(corrupt.string(), std::ref(restored)).has_value(),
          "oversized complementary vector count is rejected");
    fs::copy_file(path, corrupt, fs::copy_options::overwrite_existing);
    {
        std::fstream file(corrupt,
                          std::ios::binary | std::ios::in | std::ios::out);
        const std::int32_t version = 7;
        file.seekp(8);
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    }
    check(!read_checkpoint(corrupt.string(), std::ref(restored)).has_value(),
          "JSON-based checkpoint input version is rejected");
    check(read_checkpoint(corrupt.string()).has_value(),
          "old asymmetric checkpoint state remains restartable");
    fs::remove(corrupt);

    auto symmetric_spec = make_spec(true);
    symmetric_spec.lasym = false;
    symmetric_spec.raxis_s.clear();
    symmetric_spec.zaxis_c.clear();
    auto symmetric = validate(symmetric_spec, SolverOptions{});
    check(symmetric.has_value(), "symmetric input fixture validates");
    if (symmetric.has_value()) {
        const auto input = make_input_params(symmetric.value());
        snapshot.families.resize(6);
        const auto symmetric_path = directory / "symmetric.ckpt";
        check(write_checkpoint(snapshot, input, symmetric_path.string())
                  .has_value(),
              "write symmetric checkpoint after asymmetric input");
        check_version(symmetric_path, 6);
        check(read_checkpoint(symmetric_path.string(), std::ref(restored))
                      .has_value() &&
                  restored == input,
              "reused input record clears complementary fields for symmetric "
              "reads");
    }
}

int main(int argc, char** argv) {
    char temporary[] = "/tmp/cumes_asymmetric_io_XXXXXX";
    const char* path = argc > 1 ? argv[1] : mkdtemp(temporary);
    if (!path) return 1;
    const fs::path directory(path);
    fs::create_directories(directory);
    run_case(directory, false);
    run_case(directory, true);
    if (argc == 1) fs::remove_all(directory);
    return test::summary();
}
