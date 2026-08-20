// writer_helpers.hpp — shared primitives for the atomic-publishing writers.
//
// One implementation of the protocol every on-disk state writer needs: the
// same-directory temp path + atomic publish (temp + fsync + rename), plus the
// checked family element count and the documented reader-side resource caps.
//
// Consumers: the NetCDF/HDF5 backends (netcdf_writer.cpp / hdf5_writer.cpp)
// and — through src/cumes/io/io_common.hpp — the versioned binary and
// checkpoint writers. Everything here is header-only so the writers never
// drift apart again.
#pragma once

#include "cumes/core/checked_size.hpp"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <string>
#include <unistd.h>

namespace cumes {
namespace io_detail {

// Documented reader-side resource caps. A hostile container must fail with a
// typed error before the host allocates storage proportional to its declared
// dimensions. `ns`/`mnmax` are also bounded by INT_MAX because
// EquilibriumSnapshot stores them as ints.
inline constexpr std::size_t kMaxProvenanceStringBytes = 1u << 20;
inline constexpr std::size_t kMaxStageCount = 1u << 16;
// Per-family element cap: 16,777,216 doubles = 128 MiB. The six-family
// snapshot is therefore bounded at 768 MiB; the HDF5 reader's additional
// transpose slab brings its bounded peak state storage to 896 MiB. This is
// deliberately far above current equilibria while preventing sparse hostile
// containers from requesting multi-gigabyte allocations.
inline constexpr std::size_t kMaxStateElementsPerFamily = 1u << 24;

// A unique same-directory temp path for `path` (rename() stays on one
// filesystem). PID alone can collide across threads writing in one process, so
// an atomic per-process counter is mixed in. Uniqueness is best-effort: the
// writer still creates the file with O_EXCL semantics via fopen's "wx" mode at
// the call sites that need it, and a collision simply fails cleanly.
inline std::string tempPathFor(const std::string& path) {
    static std::atomic<unsigned long> counter{0};
    const unsigned long seq = counter.fetch_add(1, std::memory_order_relaxed);
    return path + ".tmp." + std::to_string(static_cast<long>(getpid())) + "." +
           std::to_string(seq);
}

// fsync the directory containing `path` so the rename itself is durable.
// CHECKED (completion-plan follow-up §3): the old helper ignored both the
// fsync and the close result; a failed directory fsync now propagates to the
// writer and from there to main.
inline std::string fsyncDirectoryOf(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    const std::string dir =
        (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
    const int fd = open(dir.c_str(), O_RDONLY);
    if (fd < 0) {
        return "cannot open directory for fsync";
    }
    const bool fsync_failed = fsync(fd) != 0;
    const int fsync_errno_saved = errno;
    const bool close_failed = close(fd) != 0;
    const int close_errno_saved = errno;
    if (fsync_failed) {
        return "directory fsync failed: " + std::string(strerror(fsync_errno_saved));
    }
    if (close_failed) {
        return "directory close failed: " + std::string(strerror(close_errno_saved));
    }
    return "";
}

// Rename `tmp` over `path`, then fsync the containing directory (durable
// publication protocol — completion plan step 2.4). On a rename failure
// removes the temp and returns a reason (the target `path` is left
// untouched); a directory-fsync failure propagates as an error so the caller
// can report the unproven durability to main.
inline std::string renamePublish(const std::string& tmp, const std::string& path) {
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        return "rename failed";
    }
    return fsyncDirectoryOf(path);
}

// Publish a temp file written and closed through a LIBRARY-managed handle
// (NetCDF/HDF5 own their descriptors, so the FILE* path of publishAtomic
// cannot fsync them — completion-plan follow-up §3):
//   1. reopen the completed same-directory temp file read-only;
//   2. check fsync on it;
//   3. check close;
//   4. atomically rename it over the destination;
//   5. open + fsync + checked-close the containing directory.
// Every pre-rename failure removes the temp and leaves the destination
// untouched.
inline std::string publishLibraryFile(const std::string& tmp,
                                      const std::string& path) {
    const int fd = open(tmp.c_str(), O_RDONLY);
    if (fd < 0) {
        remove(tmp.c_str());
        return "cannot reopen temp for fsync";
    }
    if (fsync(fd) != 0) {
        const std::string reason =
            "fsync failed: " + std::string(strerror(errno));
        close(fd);
        remove(tmp.c_str());
        return reason;
    }
    if (close(fd) != 0) {
        const std::string reason =
            "close failed: " + std::string(strerror(errno));
        remove(tmp.c_str());
        return reason;
    }
    return renamePublish(tmp, path);
}

// Flush + fsync + close `fp`, then atomically rename `tmp` over `path`. On any
// failure removes the temp and leaves `path` untouched; `fp` is closed exactly
// once (a failing close is not re-closed). Returns an empty string on success,
// or a human-readable reason.
inline std::string publishAtomic(FILE* fp, const std::string& tmp,
                                 const std::string& path) {
    bool fail = false;
    std::string reason;
    if (fflush(fp) != 0) { reason = "fflush failed"; fail = true; }
    if (!fail && fsync(fileno(fp)) != 0) { reason = "fsync failed"; fail = true; }
    if (fclose(fp) != 0 && !fail) { reason = "fclose failed"; fail = true; }
    if (fail) { remove(tmp.c_str()); return reason; }
    return renamePublish(tmp, path);
}

// Checked element count for one spectral family (ns x mnmax), per the
// checked_mul mandate (include/cumes/core/checked_size.hpp).
inline std::optional<std::size_t> familyCount(int ns, int mnmax) {
    return checked_mul(static_cast<std::size_t>(ns),
                       static_cast<std::size_t>(mnmax));
}

}  // namespace io_detail
}  // namespace cumes
