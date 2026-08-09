// input_json.h — JSON-file-driven input (vmecpp indata schema).
//
// initInputParamsFromJson() is defined in src/input_json.cu (the only TU
// that includes JsonParser.h and defines ZQ_JSON_PARSER_IMPLEMENTATION).
// This header stays plain so it can be included from any nvcc-compiled TU.
#pragma once
#include "input.h"

// Parse a vmecpp-style flat JSON input file into InputParams. Throws
// std::runtime_error with a "<path>: <reason>" message on any parse error,
// type mismatch, validation failure, or unsupported-feature request.
InputParams initInputParamsFromJson(const char* json_path);

// Default input file matches the previous hardcoded Solovev default
// (vmecpp/playground/solovev/solovev.json: 5->11->55 multigrid).
inline InputParams initInputParams(const char* json_path = "inputs/solovev.json") {
    return initInputParamsFromJson(json_path);
}
