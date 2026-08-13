// checkpoint.cpp — versioned checkpoint + legacy init converter.
//
// Versioned checkpoint layout (little-endian):
//   magic     8 bytes  "CUMECKP1"
//   version   int32    = 1
//   precision int32    (0 = double; the checkpoint is always double on disk)
//   ns        int32
//   mnmax     int32
//   families  6 * (mnmax*ns) doubles, mode-major
//
// convert_legacy_init reads the legacy six-family vmecpp_init.bin payload
// (int32 ns, int32 mnmax, then 6 families of double) — the exact CUMES_LOAD_INIT
// format — and validates the header against the expected shape.
#include "cumes/io/checkpoint.hpp"
#include "io_common.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace cumes {
namespace {

constexpr char kCheckpointMagic[9] = "CUMECKP1";
constexpr std::int32_t kCheckpointVersion = 1;

}  // namespace

Status write_checkpoint(const EquilibriumSnapshot& snapshot,
                        const std::string& path) {
    const std::string tmp = io_detail::tempPathFor(path);
    FILE* fp = fopen(tmp.c_str(), "wb");
    if (!fp) return Status("cannot open " + tmp + " for writing");

    auto fail = [&](const std::string& reason) -> Status {
        fclose(fp);
        remove(tmp.c_str());
        return Status(reason);
    };

    const std::size_t n = snapshot.family_size();
    bool ok = io_detail::write_bytes(fp, kCheckpointMagic, 8) &&
              io_detail::write_i32(fp, kCheckpointVersion) &&
              io_detail::write_i32(fp, 0 /* precision = double */) &&
              io_detail::write_i32(fp, snapshot.ns) &&
              io_detail::write_i32(fp, snapshot.mnmax);
    for (const auto& fam : snapshot.families) {
        if (fam.size() != n) ok = false;
        ok = ok && io_detail::write_f64_array(fp, fam.data(), n);
    }
    if (!ok) return fail("failed to write checkpoint payload");

    const std::string err = io_detail::publishAtomic(fp, tmp, path);
    if (!err.empty()) return Status("checkpoint publish: " + err);
    return Status();
}

Result<EquilibriumSnapshot> read_checkpoint(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return Result<EquilibriumSnapshot>("cannot open " + path);

    auto fail = [&](const std::string& reason) -> Result<EquilibriumSnapshot> {
        fclose(fp);
        return Result<EquilibriumSnapshot>(reason);
    };

    char magic[9] = {0};
    std::int32_t version = 0, precision = 0, ns = 0, mnmax = 0;
    if (!io_detail::read_bytes(fp, magic, 8) || !io_detail::read_i32(fp, version) ||
        !io_detail::read_i32(fp, precision) || !io_detail::read_i32(fp, ns) ||
        !io_detail::read_i32(fp, mnmax)) {
        return fail("checkpoint: truncated header");
    }
    if (std::memcmp(magic, kCheckpointMagic, 8) != 0) {
        return fail("checkpoint: bad magic (not a cumes checkpoint)");
    }
    if (version != kCheckpointVersion) {
        return fail("checkpoint: unsupported version " + std::to_string(version));
    }
    if (precision != 0) {
        return fail("checkpoint: unsupported precision tag " + std::to_string(precision));
    }
    if (ns < 1 || mnmax < 1) {
        return fail("checkpoint: bad dimensions (ns=" + std::to_string(ns) +
                    ", mnmax=" + std::to_string(mnmax) + ")");
    }
    const std::size_t n = static_cast<std::size_t>(ns) * mnmax;
    EquilibriumSnapshot snapshot;
    snapshot.ns = ns;
    snapshot.mnmax = mnmax;
    for (auto& fam : snapshot.families) {
        fam.resize(n);
        if (!io_detail::read_f64_array(fp, fam.data(), n)) {
            return fail("checkpoint: truncated state data");
        }
    }
    fclose(fp);
    return snapshot;
}

Result<EquilibriumSnapshot> convert_legacy_init(const std::string& path,
                                                int expected_ns,
                                                int expected_mnmax) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return Result<EquilibriumSnapshot>("cannot open " + path);

    auto fail = [&](const std::string& reason) -> Result<EquilibriumSnapshot> {
        fclose(fp);
        return Result<EquilibriumSnapshot>(reason);
    };

    std::int32_t ns = 0, mnmax = 0;
    if (!io_detail::read_i32(fp, ns) || !io_detail::read_i32(fp, mnmax)) {
        return fail("legacy init: truncated header");
    }
    if (ns != expected_ns || mnmax != expected_mnmax) {
        return fail("legacy init: header (ns=" + std::to_string(ns) +
                    ", mnmax=" + std::to_string(mnmax) +
                    ") does not match expected (ns=" + std::to_string(expected_ns) +
                    ", mnmax=" + std::to_string(expected_mnmax) + ")");
    }
    const std::size_t n = static_cast<std::size_t>(ns) * mnmax;
    EquilibriumSnapshot snapshot;
    snapshot.ns = ns;
    snapshot.mnmax = mnmax;
    for (auto& fam : snapshot.families) {
        fam.resize(n);
        if (!io_detail::read_f64_array(fp, fam.data(), n)) {
            return fail("legacy init: truncated state data");
        }
    }
    fclose(fp);
    return snapshot;
}

}  // namespace cumes
