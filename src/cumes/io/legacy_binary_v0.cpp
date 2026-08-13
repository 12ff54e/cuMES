// legacy_binary_v0.cpp — the exact legacy binary v0 state container
// (blueprint §6.13):
//
//   int32 ns
//   int32 mnmax
//   double rmncc[mode][surface]
//   double zmnsc[mode][surface]
//   double lmnsc[mode][surface]
//   double rmnss[mode][surface]
//   double zmncs[mode][surface]
//   double lmncs[mode][surface]
//
// Each family has mnmax*ns values with surface contiguous (mode-major,
// index = surface + mode*ns), native little-endian. On disk the values are
// always double regardless of the computation scalar type. The reader is
// strict: a truncated file or a file with trailing bytes is an error, and the
// header dimensions are validated before any allocation.
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "internal_factories.hpp"
#include "io_common.hpp"

#include <cstdio>
#include <string>

namespace cumes {
namespace {

class LegacyBinaryV0Writer final : public Writer {
 public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report, const OutputSpec& spec) override {
        (void)report;  // the v0 container records only the state
        const std::string tmp = io_detail::tempPathFor(spec.path);
        FILE* fp = fopen(tmp.c_str(), "wb");
        if (!fp) return Status("cannot open " + tmp + " for writing");

        const std::size_t n = snapshot.family_size();
        bool ok = io_detail::write_i32(fp, snapshot.ns) &&
                  io_detail::write_i32(fp, snapshot.mnmax);
        for (const auto& fam : snapshot.families) {
            if (fam.size() != n) ok = false;
            ok = ok && io_detail::write_f64_array(fp, fam.data(), n);
        }
        if (!ok) {
            fclose(fp);
            remove(tmp.c_str());
            return Status("failed to write legacy binary state");
        }
        const std::string err = io_detail::publishAtomic(fp, tmp, spec.path);
        if (!err.empty()) return Status("legacy binary publish: " + err);
        return Status();
    }
};

class LegacyBinaryV0Reader final : public Reader {
 public:
    Result<EquilibriumSnapshot> read(const std::string& path) override {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) return Result<EquilibriumSnapshot>("cannot open " + path);

        std::int32_t ns = 0, mnmax = 0;
        if (!io_detail::read_i32(fp, ns) || !io_detail::read_i32(fp, mnmax)) {
            fclose(fp);
            return Result<EquilibriumSnapshot>("legacy binary state: truncated header");
        }
        if (ns < 1 || mnmax < 1) {
            fclose(fp);
            return Result<EquilibriumSnapshot>(
                "legacy binary state: bad dimensions (ns=" + std::to_string(ns) +
                ", mnmax=" + std::to_string(mnmax) + ")");
        }
        const std::size_t n = static_cast<std::size_t>(ns) * mnmax;
        EquilibriumSnapshot snapshot;
        snapshot.ns = ns;
        snapshot.mnmax = mnmax;
        for (auto& fam : snapshot.families) {
            fam.resize(n);
            if (!io_detail::read_f64_array(fp, fam.data(), n)) {
                fclose(fp);
                return Result<EquilibriumSnapshot>(
                    "legacy binary state: truncated state data");
            }
        }
        // Trailing-data policy: the file must end exactly after the six families.
        if (fgetc(fp) != EOF) {
            fclose(fp);
            return Result<EquilibriumSnapshot>(
                "legacy binary state: trailing data after the six families");
        }
        fclose(fp);
        return snapshot;
    }
};

}  // namespace

std::unique_ptr<Writer> make_writer(OutputFormat format, OutputSchema schema) {
    if (format == OutputFormat::kBinary && schema == OutputSchema::kLegacyV0) {
        return std::make_unique<LegacyBinaryV0Writer>();
    }
    if (format == OutputFormat::kBinary && schema == OutputSchema::kV1) {
        return make_v1_writer();  // defined in versioned_binary.cpp
    }
    return nullptr;  // NetCDF/HDF5 host adapters: deferred to Phase 3.
}

std::unique_ptr<Reader> make_reader(OutputFormat format, OutputSchema schema) {
    if (format == OutputFormat::kBinary && schema == OutputSchema::kLegacyV0) {
        return std::make_unique<LegacyBinaryV0Reader>();
    }
    if (format == OutputFormat::kBinary && schema == OutputSchema::kV1) {
        return make_v1_reader();  // defined in versioned_binary.cpp
    }
    return nullptr;
}

}  // namespace cumes
