#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

namespace cumes::webgpu::detail {

// Embedded WGSL is immutable for the lifetime of the single-threaded browser
// module. Cache source assembly as well as pipelines; a pipeline cache alone
// still leaves file loading and prelude concatenation in every iteration.
inline const std::string& cached_shader_source(const char* path,
                                               const char* prelude = "") {
    static std::map<std::pair<std::string, std::string>, std::string> cache;
    static const std::string empty;
    const auto key = std::pair{std::string(path), std::string(prelude)};
    if (const auto found = cache.find(key); found != cache.end())
        return found->second;
    const auto read = [](const char* filename) {
        std::ifstream stream(filename, std::ios::binary);
        std::ostringstream text;
        if (stream) text << stream.rdbuf();
        return text.str();
    };
    auto text = read(path);
    if (text.empty()) return empty;  // Failed loads may be retried.
    if (*prelude) {
        const auto prefix = read(prelude);
        if (prefix.empty()) return empty;
        text = prefix + '\n' + text;
    }
    return cache.emplace(key, std::move(text)).first->second;
}

}  // namespace cumes::webgpu::detail
