// json_parser.cpp — the single TU that defines the JsonParser implementation.
//
// JsonParser.h is a header-only JSON library gated on
// ZQ_JSON_PARSER_IMPLEMENTATION. This TU (and only this TU) defines that macro
// so the implementation is compiled once into the cumes_json target, which
// both the legacy parser (src/input_json.cpp) and the new host-model reader
// (src/cumes/config/json_reader.cpp) link against.
#define ZQ_JSON_PARSER_IMPLEMENTATION
#include "JsonParser.h"
