// versioned_binary.cpp — schema v1 binary state container (blueprint §6.13).
//
// Layout (little-endian):
//   magic     8 bytes  "CUMES001"
//   version   int32    = 1
//   ns        int32
//   mnmax     int32
//   families  6 * (mnmax*ns) doubles, mode-major (the state payload)
//   ---- provenance trailer (after the state, so a reader can stop early) ----
//   precision int32    (0=double, 1=float; records the computation type)
//   status    int32    (RunStatus)
//   total_iter int32
//   nstages   int32
//   build     revision(str), dirty(u8), build_type(str), scalar_type(str)
//   input     source_path(str), source_hash(str)
//   runtime   gpu_name(str), driver(str), runtime(str), toolkit(str)
//   stages    per stage: ns(i32), iterations(i32), converged(u8),
//             fsqr(f64), fsqz(f64), fsql(f64), nrestarts(i32), restarts(i32...)
//
// The state payload is read and validated independently of the provenance
// trailer, so a reader stays forward-compatible with later v1.x trailers.
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "internal_factories.hpp"
#include "io_common.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace cumes {
namespace {

constexpr char kMagic[9] = "CUMES001";
constexpr std::int32_t kVersion = 1;

class VersionedBinaryWriter final : public Writer {
 public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report, const OutputSpec& spec) override {
        const std::string tmp = io_detail::tempPathFor(spec.path);
        FILE* fp = fopen(tmp.c_str(), "wb");
        if (!fp) return Status("cannot open " + tmp + " for writing");

        auto fail = [&](const std::string& reason) -> Status {
            fclose(fp);
            remove(tmp.c_str());
            return Status(reason);
        };

        const std::size_t n = snapshot.family_size();
        bool ok = io_detail::write_bytes(fp, kMagic, 8) &&
                  io_detail::write_i32(fp, kVersion) &&
                  io_detail::write_i32(fp, snapshot.ns) &&
                  io_detail::write_i32(fp, snapshot.mnmax);
        for (const auto& fam : snapshot.families) {
            if (fam.size() != n) ok = false;
            ok = ok && io_detail::write_f64_array(fp, fam.data(), n);
        }
        if (!ok) return fail("failed to write versioned state payload");

        const int precision = (report.build.scalar_type == "float") ? 1 : 0;
        ok = io_detail::write_i32(fp, precision) &&
             io_detail::write_i32(fp, static_cast<std::int32_t>(report.status)) &&
             io_detail::write_i32(fp, report.total_effective_iterations) &&
             io_detail::write_i32(fp, static_cast<std::int32_t>(report.stages.size()));
        ok = ok && io_detail::write_string(fp, report.build.revision) &&
             io_detail::write_u8(fp, report.build.dirty ? 1 : 0) &&
             io_detail::write_string(fp, report.build.build_type) &&
             io_detail::write_string(fp, report.build.scalar_type) &&
             io_detail::write_string(fp, report.input.source_path) &&
             io_detail::write_string(fp, report.input.source_hash) &&
             io_detail::write_string(fp, report.runtime.gpu_name) &&
             io_detail::write_string(fp, report.runtime.driver) &&
             io_detail::write_string(fp, report.runtime.runtime) &&
             io_detail::write_string(fp, report.runtime.toolkit);
        for (const auto& stage : report.stages) {
            ok = ok && io_detail::write_i32(fp, stage.ns) &&
                 io_detail::write_i32(fp, stage.effective_iterations) &&
                 io_detail::write_u8(fp, stage.converged ? 1 : 0) &&
                 io_detail::write_f64(fp, stage.final_residual.fsqr) &&
                 io_detail::write_f64(fp, stage.final_residual.fsqz) &&
                 io_detail::write_f64(fp, stage.final_residual.fsql) &&
                 io_detail::write_i32(fp, static_cast<std::int32_t>(stage.restarts.size()));
            for (const auto& r : stage.restarts) {
                ok = ok && io_detail::write_i32(fp, r.iteration);
            }
        }
        if (!ok) return fail("failed to write versioned provenance trailer");

        const std::string err = io_detail::publishAtomic(fp, tmp, spec.path);
        if (!err.empty()) return Status("versioned binary publish: " + err);
        return Status();
    }
};

class VersionedBinaryReader final : public Reader {
 public:
    Result<EquilibriumSnapshot> read(const std::string& path) override {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) return Result<EquilibriumSnapshot>("cannot open " + path);

        auto fail = [&](const std::string& reason) -> Result<EquilibriumSnapshot> {
            fclose(fp);
            return Result<EquilibriumSnapshot>(reason);
        };

        char magic[9] = {0};
        std::int32_t version = 0, ns = 0, mnmax = 0;
        if (!io_detail::read_bytes(fp, magic, 8) || !io_detail::read_i32(fp, version) ||
            !io_detail::read_i32(fp, ns) || !io_detail::read_i32(fp, mnmax)) {
            return fail("versioned binary: truncated header");
        }
        if (std::memcmp(magic, kMagic, 8) != 0) {
            return fail("versioned binary: bad magic (not a cumes v1 state file)");
        }
        if (version != kVersion) {
            return fail("versioned binary: unsupported version " + std::to_string(version));
        }
        std::size_t n = 0;
        std::string reason;
        if (!io_detail::checkStateDimensions(fp, ns, mnmax, n, reason)) {
            return fail("versioned binary: " + reason);
        }
        EquilibriumSnapshot snapshot;
        snapshot.ns = ns;
        snapshot.mnmax = mnmax;
        for (auto& fam : snapshot.families) {
            fam.resize(n);
            if (!io_detail::read_f64_array(fp, fam.data(), n)) {
                return fail("versioned binary: truncated state data");
            }
        }
        fclose(fp);
        return snapshot;
    }
};

}  // namespace

std::unique_ptr<Writer> make_v1_writer() { return std::make_unique<VersionedBinaryWriter>(); }
std::unique_ptr<Reader> make_v1_reader() { return std::make_unique<VersionedBinaryReader>(); }

}  // namespace cumes
