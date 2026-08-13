// internal_factories.hpp — cross-TU declarations for the v1 backend factories,
// so legacy_binary_v0.cpp's make_writer/make_reader can dispatch to them
// without a public dependency. Not installed.
#pragma once

#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"

#include <memory>

namespace cumes {

std::unique_ptr<Writer> make_v1_writer();
std::unique_ptr<Reader> make_v1_reader();

}  // namespace cumes
