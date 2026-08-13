// io_common.hpp — internal helpers shared by the host I/O backends (not
// installed). Atomic publication (temp + fsync + rename, close-safe) and
// little-endian binary field serialization.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    if (n < 0) return false;
    s.resize(static_cast<std::size_t>(n));
    return read_bytes(fp, s.data(), static_cast<std::size_t>(n));
}
inline bool read_f64_array(FILE* fp, double* p, std::size_t n) {
    return read_bytes(fp, p, n * sizeof(double));
}

}  // namespace io_detail
}  // namespace cumes
