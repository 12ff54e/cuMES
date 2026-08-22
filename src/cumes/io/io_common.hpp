// io_common.hpp — internal helpers shared by the host I/O backends (not
// installed). Little-endian binary field serialization and the shared
// magic-prefixed state payload. The atomic-publication primitives live in the
// installed include/cumes/io/writer_helpers.hpp (one implementation for every
// writer); the shared state-dimension check follows.
#pragma once

#include "cumes/core/checked_size.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/input_params.hpp"
#include "cumes/io/writer_helpers.hpp"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#include <unistd.h>

namespace cumes {
namespace io_detail {

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
// allocation (a wrong-format file decodes as enormous positive dimensions).
// Returns false and sets `reason` on failure; on success sets `n_out`.
inline bool checkStateDimensions(FILE* fp,
                                 std::int32_t ns,
                                 std::int32_t mnmax,
                                 std::size_t& n_out,
                                 std::string& reason) {
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
    // Compare in size_t: the old `(long long)*needed > *sz` cast wrapped
    // negative for byte counts in [2^63, 2^64), silently passing the
    // file-size bound and letting the reader attempt a ~2e17-element
    // allocation (std::bad_alloc -> terminate) instead of the intended
    // "dimensions implausible" error. A negative size (ftell failure) also
    // fails the bound.
    if (!needed || !sz || *sz < 0 || static_cast<std::size_t>(*sz) < *needed) {
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

// ---- shared magic-prefixed state payload (checkpoint.cpp,
// versioned_binary.cpp) ----

// Write the six families of `snapshot`, each exactly family_size() doubles.
// Returns false (before writing anything of the short family) when a family is
// not sized exactly n — a size mismatch must never fall through into an OOB
// read from a short host vector.
inline bool writeStateFamilies(FILE* fp, const EquilibriumSnapshot& snapshot) {
    const std::size_t n = snapshot.family_size();
    for (const auto& fam : snapshot.families) {
        if (fam.size() != n) return false;
        if (!io_detail::write_f64_array(fp, fam.data(), n)) return false;
    }
    return true;
}

// Resize each family to `n` and read it. `n` must come from a successful
// checkStateDimensions call (the bound against the actual file size already
// happened). Returns false on truncation.
inline bool readStateFamilies(FILE* fp,
                              std::size_t n,
                              EquilibriumSnapshot& snapshot) {
    for (auto& fam : snapshot.families) {
        fam.resize(n);
        if (!io_detail::read_f64_array(fp, fam.data(), n)) return false;
    }
    return true;
}

// ---- the embedded normalized-input record (input_params.hpp) ----
// Fixed order: 6*i32 (mpol..ncurr), 8*f64 (delt..tcon0), schema(str), then
// every vector as an i32 count + payload (am, ac, ai, aphi, raxis_c,
// zaxis_s), i32 nstages + per stage (i32 ns, i32 max_iter, f64 ftol),
// i32 nrbc + per (i32 m, i32 n, f64 value), i32 nzbs + same, then the four
// folded vectors (i32 count + f64 payload each). Written as the LAST element
// of the version-3 binary trailer and the version-2 checkpoint; the
// NetCDF/HDF5 writers map the same fields to native variables/datasets/
// attributes instead. A corrupt count must fail before the host allocates:
// every count is bounded by kMaxInputParamsVector.
constexpr std::int32_t kMaxInputParamsVector = 1 << 20;  // elements per vector

inline bool write_i32_vec(FILE* fp, const std::vector<int>& v) {
    if (v.size() > static_cast<std::size_t>(kMaxInputParamsVector))
        return false;
    if (!write_i32(fp, static_cast<std::int32_t>(v.size()))) return false;
    return v.empty() ||
           write_bytes(fp, v.data(), v.size() * sizeof(std::int32_t));
}
inline bool write_f64_vec(FILE* fp, const std::vector<double>& v) {
    if (v.size() > static_cast<std::size_t>(kMaxInputParamsVector))
        return false;
    if (!write_i32(fp, static_cast<std::int32_t>(v.size()))) return false;
    return v.empty() || write_bytes(fp, v.data(), v.size() * sizeof(double));
}
inline bool read_i32_vec(FILE* fp, std::vector<int>& v, std::string& reason) {
    std::int32_t n = 0;
    if (!read_i32(fp, n)) {
        reason = "truncated vector count";
        return false;
    }
    if (n < 0 || n > kMaxInputParamsVector) {
        reason = "vector count out of range";
        return false;
    }
    v.resize(static_cast<std::size_t>(n));
    return n == 0 ||
           read_bytes(fp, v.data(),
                      static_cast<std::size_t>(n) * sizeof(std::int32_t));
}
inline bool read_f64_vec(FILE* fp,
                         std::vector<double>& v,
                         std::string& reason) {
    std::int32_t n = 0;
    if (!read_i32(fp, n)) {
        reason = "truncated vector count";
        return false;
    }
    if (n < 0 || n > kMaxInputParamsVector) {
        reason = "vector count out of range";
        return false;
    }
    v.resize(static_cast<std::size_t>(n));
    return n == 0 || read_bytes(fp, v.data(),
                                static_cast<std::size_t>(n) * sizeof(double));
}

inline bool writeInputParams(FILE* fp, const InputParams& p) {
    bool ok = write_i32(fp, p.mpol) && write_i32(fp, p.ntor) &&
              write_i32(fp, p.nfp) && write_i32(fp, p.ntheta) &&
              write_i32(fp, p.nzeta) && write_i32(fp, p.ncurr) &&
              write_f64(fp, p.delt) && write_f64(fp, p.phiedge) &&
              write_f64(fp, p.pres_scale) && write_f64(fp, p.adiabatic_index) &&
              write_f64(fp, p.spres_ped) && write_f64(fp, p.bloat) &&
              write_f64(fp, p.curtor) && write_f64(fp, p.tcon0) &&
              write_string(fp, p.schema) && write_f64_vec(fp, p.am) &&
              write_f64_vec(fp, p.ac) && write_f64_vec(fp, p.ai) &&
              write_f64_vec(fp, p.aphi) && write_f64_vec(fp, p.raxis_c) &&
              write_f64_vec(fp, p.zaxis_s) &&
              write_i32(fp, static_cast<std::int32_t>(p.stages.size()));
    for (const auto& st : p.stages) {
        ok = ok && write_i32(fp, st.ns) && write_i32(fp, st.max_iter) &&
             write_f64(fp, st.ftol);
    }
    ok = ok && write_i32_vec(fp, p.rbc_m) && write_i32_vec(fp, p.rbc_n) &&
         write_f64_vec(fp, p.rbc_value) && write_i32_vec(fp, p.zbs_m) &&
         write_i32_vec(fp, p.zbs_n) && write_f64_vec(fp, p.zbs_value) &&
         write_f64_vec(fp, p.rbcc) && write_f64_vec(fp, p.rbss) &&
         write_f64_vec(fp, p.zbsc) && write_f64_vec(fp, p.zbcs);
    return ok;
}

inline bool readInputParams(FILE* fp, InputParams& p, std::string& reason) {
    std::int32_t nstages = 0;
    if (!read_i32(fp, p.mpol) || !read_i32(fp, p.ntor) ||
        !read_i32(fp, p.nfp) || !read_i32(fp, p.ntheta) ||
        !read_i32(fp, p.nzeta) || !read_i32(fp, p.ncurr) ||
        !read_f64(fp, p.delt) || !read_f64(fp, p.phiedge) ||
        !read_f64(fp, p.pres_scale) || !read_f64(fp, p.adiabatic_index) ||
        !read_f64(fp, p.spres_ped) || !read_f64(fp, p.bloat) ||
        !read_f64(fp, p.curtor) || !read_f64(fp, p.tcon0) ||
        !read_string(fp, p.schema) || !read_f64_vec(fp, p.am, reason) ||
        !read_f64_vec(fp, p.ac, reason) || !read_f64_vec(fp, p.ai, reason) ||
        !read_f64_vec(fp, p.aphi, reason) ||
        !read_f64_vec(fp, p.raxis_c, reason) ||
        !read_f64_vec(fp, p.zaxis_s, reason)) {
        if (reason.empty()) reason = "truncated input record";
        return false;
    }
    if (!read_i32(fp, nstages)) {
        reason = "truncated input record";
        return false;
    }
    if (nstages < 0 || nstages > kMaxInputParamsVector) {
        reason = "stage count out of range";
        return false;
    }
    for (std::int32_t g = 0; g < nstages; ++g) {
        InputStage st;
        if (!read_i32(fp, st.ns) || !read_i32(fp, st.max_iter) ||
            !read_f64(fp, st.ftol)) {
            reason = "truncated input stage record";
            return false;
        }
        p.stages.push_back(st);
    }
    if (!read_i32_vec(fp, p.rbc_m, reason) ||
        !read_i32_vec(fp, p.rbc_n, reason) ||
        !read_f64_vec(fp, p.rbc_value, reason) ||
        !read_i32_vec(fp, p.zbs_m, reason) ||
        !read_i32_vec(fp, p.zbs_n, reason) ||
        !read_f64_vec(fp, p.zbs_value, reason) ||
        !read_f64_vec(fp, p.rbcc, reason) ||
        !read_f64_vec(fp, p.rbss, reason) ||
        !read_f64_vec(fp, p.zbsc, reason) ||
        !read_f64_vec(fp, p.zbcs, reason)) {
        if (reason.empty()) reason = "truncated input record";
        return false;
    }
    // The parallel boundary vectors must agree in length (a corrupt file
    // could otherwise pair mismatched harmonics).
    if (p.rbc_m.size() != p.rbc_n.size() ||
        p.rbc_m.size() != p.rbc_value.size() ||
        p.zbs_m.size() != p.zbs_n.size() ||
        p.zbs_m.size() != p.zbs_value.size() ||
        p.rbcc.size() != p.rbss.size() || p.rbcc.size() != p.zbsc.size() ||
        p.rbcc.size() != p.zbcs.size()) {
        reason = "boundary vector lengths disagree";
        return false;
    }
    return true;
}

}  // namespace io_detail
}  // namespace cumes
