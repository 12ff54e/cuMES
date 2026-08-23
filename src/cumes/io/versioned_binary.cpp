// versioned_binary.cpp — the schema v1 binary state container (blueprint
// §6.13) and the host-library binary factories.
//
// Layout (little-endian), version 4 (the current on-disk version):
//   magic     8 bytes  "CUMES001"
//   version   int32    = 3
//   ns        int32
//   mnmax     int32
//   families  6 * (mnmax*ns) doubles, mode-major (the state payload)
//   ---- provenance trailer (after the state, so a reader can stop early) ----
//   precision int32    (0=double, 1=float; records the computation type)
//   status    int32    (RunStatus)
//   total_iter int32
//   nstages   int32
//   build     revision(str), dirty(u8), build_type(str),
//             precision_policy(str), compile_flags(str)
//   input     source_path(str), source_hash(str)
//   runtime   gpu_name(str), driver(str), runtime(str), toolkit(str)
//   stages    per stage: ns(i32), iterations(i32), converged(u8),
//             fsqr(f64), fsqz(f64), fsql(f64), nrestarts(i32), restarts(i32...)
//   params    the embedded normalized-input record (io_common.hpp
//             write_input_params), the LAST trailer element; version 3 and up
//
// All strings are int32-length-prefixed. The scalar_type string is NOT
// serialized: the reader reconstructs it from the precision tag.
//
// Version 1 (historical, still readable): the trailer instead carried a
// scalar_type string between build_type and source_path (the v1 reader
// consumes and discards it). Versions 1 and 2 carry no input record; the
// reader reports a default-empty InputParams for them. Version 3 carries the
// input record WITHOUT the three profile-type strings; version 4 appends
// pmass_type/piota_type/pcurr_type after the schema tag (the reader keeps
// the "power_series" defaults for version-3 records).
//
// The state payload is read and validated independently of the provenance
// trailer, so a reader stays forward-compatible with later v1.x trailers.
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "io_common.hpp"

#include <cstdio>
#include <cstring>
#include <optional>
#include <string>

namespace cumes {
namespace {

constexpr char MAGIC[9] = "CUMES001";
// v2 adds the precision-policy provenance fields to the trailer; v1 files
// are still read (the fields default empty). v3 appends the embedded
// normalized-input record as the last trailer element; v1/v2 files are still
// read (the record defaults empty). v4 appends the three profile-type
// strings to that record; v3 files are still read (the types default to
// "power_series").
constexpr std::int32_t VERSION = 4;
constexpr std::int32_t MIN_READ_VERSION = 1;

// The on-disk precision discriminator of the v1 trailer (0=double, 1=float).
// Typed locally instead of a string compare against
// BuildProvenance::scalar_type so an unknown tag is a write error rather than a
// silent "0=double" record.
enum class PrecisionTag : std::int32_t { DOUBLE = 0, FLOAT = 1 };

std::optional<PrecisionTag> precision_tag_for(const std::string& scalar_type) {
    if (scalar_type == "double") return PrecisionTag::DOUBLE;
    if (scalar_type == "float") return PrecisionTag::FLOAT;
    return std::nullopt;
}

class VersionedBinaryWriter final : public Writer {
   public:
    Status write_atomic(const EquilibriumSnapshot& snapshot,
                        const RunReport& report,
                        const OutputSpec& spec,
                        const ValidatedProblem& problem) override {
        (void)problem;  // v1 binary records report + state only
        const std::string tmp = io_detail::temp_path_for(spec.path);
        FILE* fp = fopen(tmp.c_str(), "wb");
        if (!fp) return Status("cannot open " + tmp + " for writing");

        auto fail = [&](const std::string& reason) -> Status {
            fclose(fp);
            remove(tmp.c_str());
            return Status(reason);
        };

        bool ok = io_detail::write_bytes(fp, MAGIC, 8) &&
                  io_detail::write_i32(fp, VERSION) &&
                  io_detail::write_i32(fp, snapshot.ns) &&
                  io_detail::write_i32(fp, snapshot.mnmax);
        if (!ok) return fail("failed to write versioned state payload");
        // write_state_families aborts on a family-size mismatch before writing
        // (an undersized family must not fall through into an OOB read).
        if (!io_detail::write_state_families(fp, snapshot)) {
            return fail("failed to write versioned state payload");
        }

        const auto precision = precision_tag_for(report.build.scalar_type);
        if (!precision)
            return fail("unknown precision tag '" + report.build.scalar_type +
                        "'");
        ok = io_detail::write_i32(fp, static_cast<std::int32_t>(*precision)) &&
             io_detail::write_i32(fp,
                                  static_cast<std::int32_t>(report.status)) &&
             io_detail::write_i32(fp, report.total_effective_iterations) &&
             io_detail::write_i32(
                 fp, static_cast<std::int32_t>(report.stages.size()));
        ok = ok && io_detail::write_string(fp, report.build.revision) &&
             io_detail::write_u8(fp, report.build.dirty ? 1 : 0) &&
             io_detail::write_string(fp, report.build.build_type) &&
             io_detail::write_string(fp, report.build.precision_policy) &&
             io_detail::write_string(fp, report.build.compile_flags) &&
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
                 io_detail::write_i32(
                     fp, static_cast<std::int32_t>(stage.restarts.size()));
            for (const auto& r : stage.restarts) {
                ok = ok && io_detail::write_i32(fp, r.iteration);
            }
        }
        // The embedded normalized-input record is the LAST trailer element.
        ok = ok && io_detail::write_input_params(fp, report.input_params);
        if (!ok) return fail("failed to write versioned provenance trailer");

        const std::string err = io_detail::publish_atomic(fp, tmp, spec.path);
        if (!err.empty()) return Status("versioned binary publish: " + err);
        return Status();
    }
};

class VersionedBinaryReader final : public Reader {
   public:
    Result<EquilibriumSnapshot> read(
        const std::string& path,
        std::optional<std::reference_wrapper<RunReport>> report) override {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) return Result<EquilibriumSnapshot>("cannot open " + path);

        auto fail =
            [&](const std::string& reason) -> Result<EquilibriumSnapshot> {
            fclose(fp);
            return Result<EquilibriumSnapshot>(reason);
        };

        char magic[9] = {0};
        std::int32_t version = 0, ns = 0, mnmax = 0;
        if (!io_detail::read_bytes(fp, magic, 8) ||
            !io_detail::read_i32(fp, version) || !io_detail::read_i32(fp, ns) ||
            !io_detail::read_i32(fp, mnmax)) {
            return fail("versioned binary: truncated header");
        }
        if (std::memcmp(magic, MAGIC, 8) != 0) {
            return fail(
                "versioned binary: bad magic (not a cumes v1 state file)");
        }
        if (version < MIN_READ_VERSION || version > VERSION) {
            return fail("versioned binary: unsupported version " +
                        std::to_string(version));
        }
        const bool has_policy_fields = (version >= 2);
        const bool has_input_params = (version >= 3);
        std::size_t n = 0;
        std::string reason;
        if (!io_detail::check_state_dimensions(fp, ns, mnmax, n, reason)) {
            return fail("versioned binary: " + reason);
        }
        EquilibriumSnapshot snapshot;
        snapshot.ns = ns;
        snapshot.mnmax = mnmax;
        if (!io_detail::read_state_families(fp, n, snapshot)) {
            return fail("versioned binary: truncated state data");
        }
        // Provenance trailer: parse into the optional RunReport (the schema-v1
        // round-trip contract, completion plan step 2.3). A truncated trailer
        // fails the read when the caller asked for the report.
        if (report) {
            std::int32_t precision = 0, status = 0, total = 0, nstages = 0;
            std::uint8_t dirty = 0;
            if (!io_detail::read_i32(fp, precision) ||
                !io_detail::read_i32(fp, status) ||
                !io_detail::read_i32(fp, total) ||
                !io_detail::read_i32(fp, nstages)) {
                return fail("versioned binary: truncated provenance trailer");
            }
            report->get() = RunReport{};
            report->get().status = static_cast<RunStatus>(status);
            report->get().total_effective_iterations = total;
            report->get().build.scalar_type =
                (precision == 0) ? "double" : "float";
            // The v2 trailer carries precision_policy + compile_flags after
            // build_type; the historical v1 trailer carried a scalar_type
            // string in that slot instead (the precision tag above is
            // authoritative either way, so the v1 string is read and
            // discarded).
            std::string legacy_scalar_type;
            if (!io_detail::read_string(fp, report->get().build.revision) ||
                !io_detail::read_u8(fp, dirty) ||
                !io_detail::read_string(fp, report->get().build.build_type) ||
                (has_policy_fields
                     ? (!io_detail::read_string(
                            fp, report->get().build.precision_policy) ||
                        !io_detail::read_string(
                            fp, report->get().build.compile_flags))
                     : !io_detail::read_string(fp, legacy_scalar_type)) ||
                !io_detail::read_string(fp, report->get().input.source_path) ||
                !io_detail::read_string(fp, report->get().input.source_hash) ||
                !io_detail::read_string(fp, report->get().runtime.gpu_name) ||
                !io_detail::read_string(fp, report->get().runtime.driver) ||
                !io_detail::read_string(fp, report->get().runtime.runtime) ||
                !io_detail::read_string(fp, report->get().runtime.toolkit)) {
                return fail("versioned binary: truncated provenance trailer");
            }
            report->get().build.dirty = (dirty != 0);
            if (nstages < 0) return fail("versioned binary: bad stage count");
            for (int g = 0; g < nstages; ++g) {
                StageReport st;
                std::int32_t ns = 0, iters = 0, nrestarts = 0;
                std::uint8_t conv = 0;
                if (!io_detail::read_i32(fp, ns) ||
                    !io_detail::read_i32(fp, iters) ||
                    !io_detail::read_u8(fp, conv) ||
                    !io_detail::read_f64(fp, st.final_residual.fsqr) ||
                    !io_detail::read_f64(fp, st.final_residual.fsqz) ||
                    !io_detail::read_f64(fp, st.final_residual.fsql) ||
                    !io_detail::read_i32(fp, nrestarts)) {
                    return fail("versioned binary: truncated stage record");
                }
                st.ns = ns;
                st.effective_iterations = iters;
                st.converged = (conv != 0);
                if (nrestarts < 0) {
                    return fail("versioned binary: bad restart count");
                }
                for (int k = 0; k < nrestarts; ++k) {
                    RestartEvent ev;
                    std::int32_t it = 0;
                    if (!io_detail::read_i32(fp, it)) {
                        return fail("versioned binary: truncated restart list");
                    }
                    ev.iteration = it;
                    st.restarts.push_back(ev);
                }
                report->get().stages.push_back(std::move(st));
            }
            // The embedded normalized-input record: required for version 3,
            // absent (default-empty) for versions 1 and 2. The three
            // profile-type strings exist in version 4 records only.
            if (has_input_params) {
                std::string reason;
                if (!io_detail::read_input_params(
                        fp, report->get().input_params, reason, version >= 4)) {
                    return fail("versioned binary: " + reason);
                }
            }
        }
        fclose(fp);
        return snapshot;
    }
};

}  // namespace

// Binary-only factories (completion plan step 2.5): the host I/O library
// needs no backend defines to answer these. The FULL make_writer/make_reader
// dispatch (including the NetCDF/HDF5 adapters) lives in the adapter library
// (src/cumes/io/writer_dispatch.cpp in cumes_io), whose strong references to
// the adapter factories force the linker to extract the adapter TUs.
std::unique_ptr<Writer> make_binary_writer() {
    return std::make_unique<VersionedBinaryWriter>();
}

std::unique_ptr<Reader> make_binary_reader() {
    return std::make_unique<VersionedBinaryReader>();
}

}  // namespace cumes
