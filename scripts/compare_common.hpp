#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cumes::compare {

inline constexpr std::array<const char*, 6> FAMILY_NAMES = {
    "rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs"};

struct State {
    std::int32_t ns = 0;
    std::int32_t mnmax = 0;
    std::array<std::vector<double>, FAMILY_NAMES.size()> families;
};

struct StatePayload {
    std::int32_t ns = 0;
    std::int32_t mnmax = 0;
    std::vector<std::uint8_t> bytes;
};

State read_state(const std::filesystem::path& path, bool allow_legacy);
StatePayload read_state_payload(const std::filesystem::path& path);

double relative_difference(double value, double reference);

// Compute a digest through the system sha256sum executable. The child writes
// stdout into a pipe; the parent validates and returns its first 64 hex digits.
std::string sha256_file(const std::filesystem::path& path);

std::vector<std::filesystem::path> files_with_prefix(
    const std::filesystem::path& directory,
    const std::string& prefix);
std::vector<std::filesystem::path> dump_paths(
    const std::filesystem::path& tree_root);

}  // namespace cumes::compare

#include "compare_common_impl.hpp"
