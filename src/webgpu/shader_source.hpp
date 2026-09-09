#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

namespace cumes::webgpu::detail {

// Embedded WGSL is immutable for the lifetime of the single-threaded browser
// module. Cache source loading as well as pipelines; a pipeline cache alone
// still leaves file loading in every iteration. Templates arrive preprocessed.
inline const std::string& cached_shader_source(const char* path) {
    static std::map<std::string, std::string, std::less<>> cache;
    static const std::string empty;
    if (const auto found = cache.find(path); found != cache.end())
        return found->second;
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream contents;
    if (stream) contents << stream.rdbuf();
    auto text = contents.str();
    if (text.empty()) return empty;  // Failed loads may be retried.
    return cache.emplace(path, std::move(text)).first->second;
}

}  // namespace cumes::webgpu::detail
