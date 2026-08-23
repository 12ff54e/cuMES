// checkpoint.cpp — versioned checkpoint.
//
// Versioned checkpoint layout (little-endian):
//   magic     8 bytes  "CUMECKP1"
//   version   int32    = 3
//   precision int32    (0 = double; the checkpoint is always double on disk)
//   ns        int32
//   mnmax     int32
//   families  6 * (mnmax*ns) doubles, mode-major
//   params    the embedded normalized-input record (io_common.hpp
//             writeInputParams); version 2 and up
//
// Version 3 appends the three profile-type strings to the input record;
// version-2 checkpoints are still read (the types default to
// "power_series"). The restart path reads the state only and ignores the
// record; version-1 checkpoints (no record) remain readable.
#include "cumes/io/checkpoint.hpp"

#include "io_common.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace cumes {
namespace {

constexpr char CHECKPOINT_MAGIC[9] = "CUMECKP1";
constexpr std::int32_t CHECKPOINT_VERSION = 3;
constexpr std::int32_t MIN_CHECKPOINT_VERSION = 1;

}  // namespace

Status write_checkpoint(const EquilibriumSnapshot& snapshot,
                        const InputParams& input_params,
                        const std::string& path) {
    const std::string tmp = io_detail::tempPathFor(path);
    FILE* fp = fopen(tmp.c_str(), "wb");
    if (!fp) return Status("cannot open " + tmp + " for writing");

    auto fail = [&](const std::string& reason) -> Status {
        fclose(fp);
        remove(tmp.c_str());
        return Status(reason);
    };

    bool ok = io_detail::write_bytes(fp, CHECKPOINT_MAGIC, 8) &&
              io_detail::write_i32(fp, CHECKPOINT_VERSION) &&
              io_detail::write_i32(fp, 0 /* precision = double */) &&
              io_detail::write_i32(fp, snapshot.ns) &&
              io_detail::write_i32(fp, snapshot.mnmax);
    if (!ok) return fail("failed to write checkpoint payload");
    // writeStateFamilies aborts on a family-size mismatch before writing (an
    // undersized family must not fall through into an OOB read).
    if (!io_detail::writeStateFamilies(fp, snapshot)) {
        return fail("failed to write checkpoint payload");
    }
    if (!io_detail::writeInputParams(fp, input_params)) {
        return fail("failed to write checkpoint input record");
    }

    const std::string err = io_detail::publishAtomic(fp, tmp, path);
    if (!err.empty()) return Status("checkpoint publish: " + err);
    return Status();
}

Result<EquilibriumSnapshot> read_checkpoint(const std::string& path,
                                            InputParams* input_params) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return Result<EquilibriumSnapshot>("cannot open " + path);

    auto fail = [&](const std::string& reason) -> Result<EquilibriumSnapshot> {
        fclose(fp);
        return Result<EquilibriumSnapshot>(reason);
    };

    char magic[9] = {0};
    std::int32_t version = 0, precision = 0, ns = 0, mnmax = 0;
    if (!io_detail::read_bytes(fp, magic, 8) ||
        !io_detail::read_i32(fp, version) ||
        !io_detail::read_i32(fp, precision) || !io_detail::read_i32(fp, ns) ||
        !io_detail::read_i32(fp, mnmax)) {
        return fail("checkpoint: truncated header");
    }
    if (std::memcmp(magic, CHECKPOINT_MAGIC, 8) != 0) {
        return fail("checkpoint: bad magic (not a cumes checkpoint)");
    }
    if (version < MIN_CHECKPOINT_VERSION || version > CHECKPOINT_VERSION) {
        return fail("checkpoint: unsupported version " +
                    std::to_string(version));
    }
    if (precision != 0) {
        return fail("checkpoint: unsupported precision tag " +
                    std::to_string(precision));
    }
    std::size_t n = 0;
    std::string reason;
    if (!io_detail::checkStateDimensions(fp, ns, mnmax, n, reason)) {
        return fail("checkpoint: " + reason);
    }
    EquilibriumSnapshot snapshot;
    snapshot.ns = ns;
    snapshot.mnmax = mnmax;
    if (!io_detail::readStateFamilies(fp, n, snapshot)) {
        return fail("checkpoint: truncated state data");
    }
    // The version-2 input record rides after the families; the state read
    // above is the whole restart contract, and a reader that does not ask
    // for the record never touches it. The three profile-type strings exist
    // in version-3 checkpoints only.
    if (input_params && version >= 2) {
        if (!io_detail::readInputParams(fp, *input_params, reason,
                                        version >= 3)) {
            return fail("checkpoint: " + reason);
        }
    }
    fclose(fp);
    return snapshot;
}

}  // namespace cumes
