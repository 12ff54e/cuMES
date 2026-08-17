// writer_helpers.hpp — shared primitives for the atomic-publishing writers.
//
// One implementation of two protocols every on-disk state writer needs:
//   (a) the same-directory temp path + atomic publish (temp + fsync + rename),
//   (b) the device->host T->double family staging (the on-disk state stays
//       double regardless of the computation scalar type T).
//
// Consumers: the binary writer (src/output.cpp), the netCDF/HDF5 backends
// (src/output_netcdf.cpp / src/output_hdf5.cpp), and — through
// src/cumes/io/io_common.hpp — the checkpoint and versioned-container writers.
// Everything here is header-only so the writers never drift apart again.
//
// The host-only halves (a) and familyCount need no CUDA; the device staging
// (b) is opt-in because the pure host library cumes_io_host (checkpoint +
// versioned containers) is compiled without CUDA include paths. The three
// device-reading writers define CUMES_IO_DEVICE_STAGE before including this.
#pragma once

#include "cumes/core/checked_size.hpp"

#ifdef CUMES_IO_DEVICE_STAGE
#include "cumes/runtime/cuda_status.hpp"
#endif

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

namespace cumes {
namespace io_detail {

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

// fsync the directory containing `path` so the rename itself is durable
// (best-effort: only where the platform exposes a directory fd).
inline void fsyncDirectoryOf(const std::string& path) {
    const std::string dir =
        path.substr(0, path.find_last_of('/') == std::string::npos
                          ? 0
                          : path.find_last_of('/') + 1);
    const int fd = open(dir.empty() ? "." : dir.c_str(), O_RDONLY);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

// Rename `tmp` over `path`, then fsync the containing directory (durable
// publication protocol — completion plan step 2.4). On failure removes the
// temp and returns a reason (the target `path` is left untouched either way).
inline std::string renamePublish(const std::string& tmp, const std::string& path) {
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        return "rename failed";
    }
    fsyncDirectoryOf(path);
    return "";
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

#ifdef CUMES_IO_DEVICE_STAGE

// Device->host family staging for the on-disk-double writers: owns (RAII) the
// T mirror + double conversion buffers, copies `count` elements from the
// device, and converts T -> double. A CUDA copy failure is reported through
// `reason` and returns false — never throws — so the writer can run its own
// fail path (close + remove temp + return false) and the atomic-publish
// contract survives a device fault.
template <class T>
class FamilyStage {
 public:
    explicit FamilyStage(std::size_t count) : buf_(count), dbuf_(count) {}

    bool copy(const T* d_src, const char* tag, std::string& reason) {
        if (!buf_.empty()) {
            const auto bytes = checked_mul(buf_.size(), sizeof(T));
            if (!bytes) {
                reason = std::string(tag) + ": family byte size overflows size_t";
                return false;
            }
            try {
                check_cuda(cudaMemcpy(buf_.data(), d_src, *bytes,
                                      cudaMemcpyDeviceToHost), tag);
            } catch (const CumesError& e) {
                reason = e.what();
                return false;
            }
        }
        for (std::size_t i = 0; i < buf_.size(); ++i) {
            dbuf_[i] = static_cast<double>(buf_[i]);
        }
        return true;
    }

    const double* data() const { return dbuf_.data(); }

 private:
    std::vector<T> buf_;
    std::vector<double> dbuf_;
};

#endif  // CUMES_IO_DEVICE_STAGE

}  // namespace io_detail
}  // namespace cumes
