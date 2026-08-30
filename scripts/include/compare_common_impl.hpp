// Inline implementation keeps each comparison tool directly compilable.

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace cumes::compare {
namespace {

constexpr std::size_t STATE_HEADER_SIZE = 20;
constexpr std::array<std::uint8_t, 8> STATE_MAGIC = {'C', 'U', 'M', 'E',
                                                     'S', '0', '0', '1'};

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("could not open " + path.string());
    }
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error("could not size " + path.string());
    }
    const auto size = static_cast<std::uintmax_t>(end);
    if (size > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("file is too large: " + path.string());
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream) {
        throw std::runtime_error("could not read " + path.string());
    }
    return bytes;
}

std::int32_t read_i32(const std::vector<std::uint8_t>& bytes,
                      std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4) {
        throw std::runtime_error("truncated int32 field");
    }
    std::uint32_t value = 0;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        value |= static_cast<std::uint32_t>(bytes[offset + byte]) << (8 * byte);
    }
    std::int32_t signed_value = 0;
    static_assert(sizeof(signed_value) == sizeof(value));
    std::memcpy(&signed_value, &value, sizeof(value));
    return signed_value;
}

double read_f64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 8) {
        throw std::runtime_error("truncated float64 field");
    }
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(bytes[offset + byte]) << (8 * byte);
    }
    double decoded = 0.0;
    static_assert(sizeof(decoded) == sizeof(value));
    std::memcpy(&decoded, &value, sizeof(value));
    return decoded;
}

std::size_t payload_size(std::int32_t ns, std::int32_t mnmax) {
    if (ns < 1 || mnmax < 1) {
        throw std::runtime_error("invalid state dimensions");
    }
    const auto surfaces = static_cast<std::size_t>(ns);
    const auto modes = static_cast<std::size_t>(mnmax);
    if (modes > std::numeric_limits<std::size_t>::max() / surfaces) {
        throw std::runtime_error("state dimensions overflow");
    }
    const auto values = surfaces * modes;
    constexpr std::size_t bytes_per_value = sizeof(double);
    constexpr std::size_t family_count = FAMILY_NAMES.size();
    if (values > std::numeric_limits<std::size_t>::max() /
                     (family_count * bytes_per_value)) {
        throw std::runtime_error("state payload size overflow");
    }
    return values * family_count * bytes_per_value;
}

bool has_state_magic(const std::vector<std::uint8_t>& bytes) {
    return bytes.size() >= STATE_MAGIC.size() &&
           std::equal(STATE_MAGIC.begin(), STATE_MAGIC.end(), bytes.begin());
}

struct StateLayout {
    std::int32_t ns = 0;
    std::int32_t mnmax = 0;
    std::size_t offset = 0;
};

StateLayout resolve_layout(const std::vector<std::uint8_t>& bytes,
                           bool allow_legacy,
                           const std::filesystem::path& path) {
    if (bytes.size() >= STATE_HEADER_SIZE && has_state_magic(bytes)) {
        const auto version = read_i32(bytes, 8);
        const auto ns = read_i32(bytes, 12);
        const auto mnmax = read_i32(bytes, 16);
        if (version < 1 || version > 8) {
            throw std::runtime_error(path.string() +
                                     " has an unsupported state version");
        }
        const auto size = payload_size(ns, mnmax);
        if (bytes.size() - STATE_HEADER_SIZE < size) {
            throw std::runtime_error(path.string() +
                                     " has a truncated state payload");
        }
        return {ns, mnmax, STATE_HEADER_SIZE};
    }

    if (allow_legacy && bytes.size() >= 8) {
        const auto ns = read_i32(bytes, 0);
        const auto mnmax = read_i32(bytes, 4);
        try {
            const auto size = payload_size(ns, mnmax);
            if (bytes.size() == 8 + size) { return {ns, mnmax, 8}; }
        } catch (const std::runtime_error&) {
            // Fall through to the stable container error below.
        }
    }
    throw std::runtime_error(path.string() + " is not a cuMES state container");
}

}  // namespace

inline State read_state(const std::filesystem::path& path, bool allow_legacy) {
    const auto bytes = read_file(path);
    const auto layout = resolve_layout(bytes, allow_legacy, path);
    State state;
    state.ns = layout.ns;
    state.mnmax = layout.mnmax;
    const auto family_values = static_cast<std::size_t>(state.ns) *
                               static_cast<std::size_t>(state.mnmax);
    std::size_t offset = layout.offset;
    for (auto& family : state.families) {
        family.resize(family_values);
        for (auto& value : family) {
            value = read_f64(bytes, offset);
            offset += sizeof(double);
        }
    }
    return state;
}

inline StatePayload read_state_payload(const std::filesystem::path& path) {
    const auto bytes = read_file(path);
    const auto layout = resolve_layout(bytes, false, path);
    const auto size = payload_size(layout.ns, layout.mnmax);
    StatePayload payload;
    payload.ns = layout.ns;
    payload.mnmax = layout.mnmax;
    payload.bytes.assign(
        bytes.begin() + static_cast<std::ptrdiff_t>(layout.offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(layout.offset + size));
    return payload;
}

inline double relative_difference(double value, double reference) {
    return std::abs(value - reference) / std::max(1.0, std::abs(reference));
}

inline std::string sha256_file(const std::filesystem::path& path) {
    int descriptors[2] = {-1, -1};
    if (pipe(descriptors) != 0) {
        throw std::system_error(errno, std::generic_category(), "pipe");
    }
    const pid_t child = fork();
    if (child < 0) {
        const int error = errno;
        close(descriptors[0]);
        close(descriptors[1]);
        throw std::system_error(error, std::generic_category(), "fork");
    }
    if (child == 0) {
        close(descriptors[0]);
        if (dup2(descriptors[1], STDOUT_FILENO) < 0) { _exit(126); }
        close(descriptors[1]);
        const std::string path_string = path.string();
        execl("/usr/bin/sha256sum", "sha256sum", "--", path_string.c_str(),
              static_cast<char*>(nullptr));
        execl("/bin/sha256sum", "sha256sum", "--", path_string.c_str(),
              static_cast<char*>(nullptr));
        _exit(127);
    }

    close(descriptors[1]);
    std::string output;
    std::array<char, 512> buffer{};
    while (true) {
        const auto count = read(descriptors[0], buffer.data(), buffer.size());
        if (count > 0) {
            output.append(buffer.data(), static_cast<std::size_t>(count));
        } else if (count == 0) {
            break;
        } else if (errno != EINTR) {
            const int error = errno;
            close(descriptors[0]);
            int ignored = 0;
            waitpid(child, &ignored, 0);
            throw std::system_error(error, std::generic_category(),
                                    "read sha256sum output");
        }
    }
    close(descriptors[0]);

    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "waitpid");
        }
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("sha256sum failed for " + path.string());
    }
    if (output.size() < 64 || !std::all_of(output.begin(), output.begin() + 64,
                                           [](unsigned char value) {
                                               return std::isxdigit(value);
                                           })) {
        throw std::runtime_error("invalid sha256sum output for " +
                                 path.string());
    }
    std::string digest = output.substr(0, 64);
    std::transform(digest.begin(), digest.end(), digest.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return digest;
}

inline std::vector<std::filesystem::path> files_with_prefix(
    const std::filesystem::path& directory,
    const std::string& prefix) {
    std::vector<std::filesystem::path> paths;
    std::error_code error;
    for (std::filesystem::directory_iterator it(directory, error), end;
         !error && it != end; it.increment(error)) {
        const auto name = it->path().filename().string();
        if (name.size() >= prefix.size() &&
            name.compare(0, prefix.size(), prefix) == 0) {
            paths.push_back(it->path().filename());
        }
    }
    if (error) {
        throw std::system_error(error, "could not list " + directory.string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

inline std::vector<std::filesystem::path> dump_paths(
    const std::filesystem::path& tree_root) {
    const auto directory = tree_root / "dump" / "cuMES";
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) { return {}; }
    if (error) {
        throw std::system_error(error,
                                "could not inspect " + directory.string());
    }
    std::vector<std::filesystem::path> paths;
    for (std::filesystem::directory_iterator it(directory, error), end;
         !error && it != end; it.increment(error)) {
        if (it->is_regular_file(error)) {
            paths.push_back(std::filesystem::path("dump") / "cuMES" /
                            it->path().filename());
        }
        if (error) break;
    }
    if (error) {
        throw std::system_error(error, "could not list " + directory.string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

}  // namespace cumes::compare
