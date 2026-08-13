// io_common.hpp — internal helpers shared by the host I/O backends (not
// installed). Atomic publication (temp + fsync + rename, close-safe) and
// little-endian binary field serialization.
#pragma once

#include "cumes/core/checked_size.hpp"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <unistd.h>

namespace cumes {
namespace io_detail {

// A same-directory temp path for `path` (rename() stays on one filesystem).
inline std::string tempPathFor(const std::string& path) {
    return path + ".tmp." + std::to_string(static_cast<long>(getpid()));
}

// Flush + fsync + close `fp`, then atomically rename `tmp` over `path`. On any
// failure removes the temp and leaves `path` untouched; `fp` is closed exactly
// once. Returns an empty string on success, or a human-readable reason.
inline std::string publishAtomic(FILE* fp, const std::string& tmp,
                                 const std::string& path) {
    bool fail = false;
    std::string reason;
    if (fflush(fp) != 0) { reason = "fflush failed"; fail = true; }
    if (!fail && fsync(fileno(fp)) != 0) { reason = "fsync failed"; fail = true; }
    if (fclose(fp) != 0 && !fail) { reason = "fclose failed"; fail = true; }
    if (fail) { remove(tmp.c_str()); return reason; }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        return "rename failed";
    }
    return "";
}

// Current file size in bytes (position-preserving), or nullopt on failure.
inline std::optional<long long> fileSize(FILE* fp) {
    const long long pos = ftell(fp);
    if (pos < 0 || fseek(fp, 0, SEEK_END) != 0) return std::nullopt;
    const long long sz = ftell(fp);
    fseek(fp, static_cast<long>(pos), SEEK_SET);
    if (sz < 0) return std::nullopt;
    return sz;
}

// Validate state dimensions and bound the per-family element count `n` against
// the actual file size, so a corrupt or mismatched header cannot trigger a huge
// allocation (the v0 format has no magic, so a wrong-format file decodes as
// enormous positive dimensions). Returns false and sets `reason` on failure;
// on success sets `n_out`.
inline bool checkStateDimensions(FILE* fp, std::int32_t ns, std::int32_t mnmax,
                                 std::size_t& n_out, std::string& reason) {
    if (ns < 1 || mnmax < 1) {
        reason = "bad dimensions (ns=" + std::to_string(ns) +
                 ", mnmax=" + std::to_string(mnmax) + ")";
        return false;
    }
    auto n = checked_mul(static_cast<std::size_t>(ns),
                         static_cast<std::size_t>(mnmax));
    if (!n) {
        reason = "dimension product overflows size_t";
        return false;
    }
    auto needed = checked_mul(*n, 6 * sizeof(double));
    auto sz = fileSize(fp);
    if (!needed || !sz || static_cast<long long>(*needed) > *sz) {
        reason = "dimensions implausible for file size (truncated or corrupt)";
        return false;
    }
    n_out = *n;
    return true;
}

// ---- little-endian binary field I/O (native on the supported x86 host) ----
inline bool write_u8(FILE* fp, std::uint8_t v) {
    return fwrite(&v, sizeof(v), 1, fp) == 1;
}
inline bool write_i32(FILE* fp, std::int32_t v) {
    return fwrite(&v, sizeof(v), 1, fp) == 1;
}
inline bool write_f64(FILE* fp, double v) {
    return fwrite(&v, sizeof(v), 1, fp) == 1;
}
inline bool write_bytes(FILE* fp, const void* p, std::size_t n) {
    return n == 0 || fwrite(p, 1, n, fp) == n;
}
inline bool write_string(FILE* fp, const std::string& s) {
    if (s.size() > static_cast<std::size_t>(INT32_MAX)) return false;
    return write_i32(fp, static_cast<std::int32_t>(s.size())) &&
           write_bytes(fp, s.data(), s.size());
}
inline bool write_f64_array(FILE* fp, const double* p, std::size_t n) {
    return write_bytes(fp, p, n * sizeof(double));
}

inline bool read_u8(FILE* fp, std::uint8_t& v) {
    return fread(&v, sizeof(v), 1, fp) == 1;
}
inline bool read_i32(FILE* fp, std::int32_t& v) {
    return fread(&v, sizeof(v), 1, fp) == 1;
}
inline bool read_f64(FILE* fp, double& v) {
    return fread(&v, sizeof(v), 1, fp) == 1;
}
inline bool read_bytes(FILE* fp, void* p, std::size_t n) {
    return n == 0 || fread(p, 1, n, fp) == n;
}
inline bool read_string(FILE* fp, std::string& s) {
    std::int32_t n = 0;
    if (!read_i32(fp, n)) return false;
    if (n < 0 || n > (1 << 24)) return false;  // cap a corrupt length prefix
    s.resize(static_cast<std::size_t>(n));
    return read_bytes(fp, s.data(), static_cast<std::size_t>(n));
}
inline bool read_f64_array(FILE* fp, double* p, std::size_t n) {
    return read_bytes(fp, p, n * sizeof(double));
}

}  // namespace io_detail
}  // namespace cumes
