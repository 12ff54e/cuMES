#include "../src/webgpu/shader_source.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;  // Caller supplies an isolated scratch directory.
    const std::filesystem::path scratch(argv[1]);
    const auto shader = (scratch / "kernel.wgsl").string();
    const auto prelude = (scratch / "prelude.wgsl").string();
    const auto write = [](const auto& path, const auto& text) {
        std::ofstream output(path, std::ios::binary);
        output << text;
    };
    using cumes::webgpu::detail::cached_shader_source;
    if (!cached_shader_source(shader.c_str()).empty()) return 1;
    write(shader, "kernel\n");
    const auto& plain = cached_shader_source(shader.c_str());
    if (plain != "kernel\n" ||
        !cached_shader_source(shader.c_str(), prelude.c_str()).empty())
        return 1;
    write(prelude, "prelude\n");
    const auto& paired = cached_shader_source(shader.c_str(), prelude.c_str());
    if (paired != "prelude\n\nkernel\n" || plain != "kernel\n") return 1;
    // Module-embedded source is immutable: subsequent loads reuse the same
    // object rather than reading or assembling the text again.
    write(shader, "changed");
    if (&plain != &cached_shader_source(shader.c_str()) ||
        &paired != &cached_shader_source(shader.c_str(), prelude.c_str()) ||
        plain != "kernel\n" || paired != "prelude\n\nkernel\n")
        return 1;
    std::cout << "PASS: shader source identity, prelude assembly, immutable "
                 "cache, and failed-load retry\n";
}
