// json_reader.cpp — JSON-file-driven input (vmecpp indata schema).
//
// JsonParser.h is a header-only JSON library (C++20) gated on
// ZQ_JSON_PARSER_IMPLEMENTATION; it is included here WITHOUT the macro — the
// one implementation TU is src/json_parser.cpp (cumes_json), which this
// library PUBLIC-links, so the json:: symbols ride along for every consumer.
// Maps a vmecpp-style flat JSON document onto the dynamic ProblemSpec,
// reproducing the legacy src/input_json.cpp semantics: every key is optional,
// a wrong type is an error, keys for features cuMES does not implement are
// rejected, and unknown keys warn (compatibility) or error (strict).
//
// Difference from the legacy parser: findings are COLLECTED into a
// ValidationReport rather than thrown one-at-a-time, and the fixed capacities
// (8 stages, 16 coefficients, 256 boundary entries, 32 axis entries) are gone
// — the model is dynamic.
#include "cumes/config/json_reader.hpp"

#include "JsonParser.h"

#include <climits>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cumes {

namespace {

const std::set<std::string> SUPPORTED_KEYS = {
    "mpol",
    "ntor",
    "nfp",
    "ntheta",
    "nzeta",
    "ncurr",
    "delt",
    "phiedge",
    "pres_scale",
    "adiabatic_index",
    "gamma",
    "spres_ped",
    "curtor",
    "bloat",
    "tcon0",
    "am",
    "ac",
    "ai",
    "aphi",
    "raxis_c",
    "zaxis_s",
    "ns_array",
    "niter_array",
    "ftol_array",
    "rbc",
    "zbs",
    "lasym",
    "lfreeb",
    "pmass_type",
    "piota_type",
    "pcurr_type",
    "am_aux_s",
    "am_aux_f",
    "ai_aux_s",
    "ai_aux_f",
    "ac_aux_s",
    "ac_aux_f",
    "raxis_s",
    "zaxis_c",
    "rbs",
    "zbc",
    "mgrid_file",
    "coils_file",
    "makegrid_parameters_file",
    "makegrid_parameters",
    "extcur",
    "nvacskip",
};
const std::set<std::string> KNOWN_IGNORED_KEYS = {
    "nstep",     "niter",   "ftolv",      "nsurf",
    "tsw",       "tpot",    "tvac",       "lforbal",
    "lmorebdy",  "lrecon",  "lmove_axis", "lthreed",
    "lpoloidal", "nthreed", "npoloidal",  "nlambda",
    "lspectral", "lcheck",  "lpsplot",    "lwout",
    "lmask",     "nedge",   "nskip",      "free_boundary_method",
};

// Typed getters: on a type/range error record the finding and return the
// fallback so the mapped field keeps a valid default (validate() then reports
// any remaining semantic issue without a misleading cascade).
int get_int(const json::Value& v,
            std::string_view key,
            int fallback,
            ValidationReport& report) {
    if (v.value_category() != json::ValueCategory::NumberInt) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': expected an integer, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    const long long vv = v.as_number<long long>();
    if (vv < INT_MIN || vv > INT_MAX) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': integer value " +
                         std::to_string(vv) + " is out of range");
        return fallback;
    }
    return static_cast<int>(vv);
}

double get_double(const json::Value& v,
                  std::string_view key,
                  double fallback,
                  ValidationReport& report) {
    if (!v.is_number()) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': expected a number, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    return v.as_number<double>();
}

bool get_bool(const json::Value& v,
              std::string_view key,
              bool fallback,
              ValidationReport& report) {
    if (!v.is_boolean()) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': expected a boolean, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    return v.as_boolean();
}

std::string get_string(const json::Value& v,
                       std::string_view key,
                       std::string fallback,
                       ValidationReport& report) {
    if (!v.is_string()) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': expected a string, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    return v.as_string();
}

// Read an array of JSON numbers into a dynamic vector. strict_int requires
// integer literals and range-checks the narrowing to int.
std::vector<int> read_int_array(const json::Value& v,
                                std::string_view key,
                                ValidationReport& report) {
    std::vector<int> out;
    if (!v.is_array()) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': expected an array, got " +
                         json::get_value_category_name(v.value_category()));
        return out;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        const json::Value& e = v[i];
        if (e.value_category() != json::ValueCategory::NumberInt) {
            report.error(std::string(key),
                         "'" + std::string(key) + "[" + std::to_string(i) +
                             "]': expected an integer, got " +
                             json::get_value_category_name(e.value_category()));
            out.push_back(0);
            continue;
        }
        const long long vv = e.as_number<long long>();
        if (vv < INT_MIN || vv > INT_MAX) {
            report.error(std::string(key),
                         "'" + std::string(key) + "[" + std::to_string(i) +
                             "]': integer value " + std::to_string(vv) +
                             " is out of range");
            out.push_back(0);
            continue;
        }
        out.push_back(static_cast<int>(vv));
    }
    return out;
}

std::vector<double> read_double_array(const json::Value& v,
                                      std::string_view key,
                                      ValidationReport& report) {
    std::vector<double> out;
    if (!v.is_array()) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': expected an array, got " +
                         json::get_value_category_name(v.value_category()));
        return out;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        const json::Value& e = v[i];
        if (!e.is_number()) {
            report.error(std::string(key),
                         "'" + std::string(key) + "[" + std::to_string(i) +
                             "]': expected a number, got " +
                             json::get_value_category_name(e.value_category()));
            out.push_back(0.0);
            continue;
        }
        out.push_back(e.as_number<double>());
    }
    return out;
}

// Boundary coefficients: array of {"n","m","value"} objects. Out-of-range
// modes are NOT rejected here (the dynamic model cannot index out of bounds);
// validate() skips them with a warning, matching vmecpp ignore-and-continue.
void read_boundary(const json::Value& v,
                   std::string_view key,
                   std::vector<BoundaryHarmonic>& out,
                   ValidationReport& report) {
    if (!v.is_array()) {
        report.error(std::string(key),
                     "'" + std::string(key) +
                         "': expected an array of "
                         "{\"n\",\"m\",\"value\"} objects, got " +
                         json::get_value_category_name(v.value_category()));
        return;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        const json::Value& e = v[i];
        const std::string where =
            std::string(key) + "[" + std::to_string(i) + "]";
        if (!e.is_object()) {
            report.error(where, "'" + where +
                                    "': expected an object with "
                                    "\"n\", \"m\" and \"value\"");
            continue;
        }
        if (!e.contains("m") || !e.contains("n") || !e.contains("value")) {
            report.error(where,
                         "'" + where + "': missing \"n\", \"m\" or \"value\"");
            continue;
        }
        const int m = get_int(e.at("m"), where + ".m", 0, report);
        const int n = get_int(e.at("n"), where + ".n", 0, report);
        const double value =
            get_double(e.at("value"), where + ".value", 0.0, report);
        out.push_back(BoundaryHarmonic{m, n, value});
    }
}

MakegridParametersSpec read_makegrid_parameters(const json::Value& value,
                                                std::string_view key,
                                                ValidationReport& report) {
    MakegridParametersSpec parameters;
    if (!value.is_object()) {
        report.error(std::string(key),
                     "'" + std::string(key) + "': expected an object, got " +
                         json::get_value_category_name(value.value_category()));
        return parameters;
    }

    const std::set<std::string> required = {
        "normalize_by_currents",   "assume_stellarator_symmetry",
        "number_of_field_periods", "r_grid_minimum",
        "r_grid_maximum",          "number_of_r_grid_points",
        "z_grid_minimum",          "z_grid_maximum",
        "number_of_z_grid_points", "number_of_phi_grid_points",
    };
    for (const std::string& field : required) {
        if (!value.contains(field)) {
            report.error(std::string(key) + "." + field,
                         "'" + std::string(key) +
                             "': missing required field '" + field + "'");
        }
    }
    for (const auto& [field, unused] : value.as_object()) {
        static_cast<void>(unused);
        if (required.count(field) == 0) {
            report.error(
                std::string(key) + "." + field,
                "'" + std::string(key) + "': unknown field '" + field + "'");
        }
    }

    auto if_present = [&](std::string_view field, auto fn) {
        const std::string field_name(field);
        if (value.contains(field_name)) {
            fn(value.at(field_name), std::string(key) + "." + field_name);
        }
    };
    if_present("normalize_by_currents",
               [&](const json::Value& v, std::string_view where) {
                   parameters.normalize_by_currents = get_bool(
                       v, where, parameters.normalize_by_currents, report);
               });
    if_present("assume_stellarator_symmetry", [&](const json::Value& v,
                                                  std::string_view where) {
        parameters.assume_stellarator_symmetry =
            get_bool(v, where, parameters.assume_stellarator_symmetry, report);
    });
    if_present("number_of_field_periods",
               [&](const json::Value& v, std::string_view where) {
                   parameters.number_of_field_periods = get_int(
                       v, where, parameters.number_of_field_periods, report);
               });
    if_present("r_grid_minimum",
               [&](const json::Value& v, std::string_view where) {
                   parameters.r_grid_minimum =
                       get_double(v, where, parameters.r_grid_minimum, report);
               });
    if_present("r_grid_maximum",
               [&](const json::Value& v, std::string_view where) {
                   parameters.r_grid_maximum =
                       get_double(v, where, parameters.r_grid_maximum, report);
               });
    if_present("number_of_r_grid_points",
               [&](const json::Value& v, std::string_view where) {
                   parameters.number_of_r_grid_points = get_int(
                       v, where, parameters.number_of_r_grid_points, report);
               });
    if_present("z_grid_minimum",
               [&](const json::Value& v, std::string_view where) {
                   parameters.z_grid_minimum =
                       get_double(v, where, parameters.z_grid_minimum, report);
               });
    if_present("z_grid_maximum",
               [&](const json::Value& v, std::string_view where) {
                   parameters.z_grid_maximum =
                       get_double(v, where, parameters.z_grid_maximum, report);
               });
    if_present("number_of_z_grid_points",
               [&](const json::Value& v, std::string_view where) {
                   parameters.number_of_z_grid_points = get_int(
                       v, where, parameters.number_of_z_grid_points, report);
               });
    if_present("number_of_phi_grid_points",
               [&](const json::Value& v, std::string_view where) {
                   parameters.number_of_phi_grid_points = get_int(
                       v, where, parameters.number_of_phi_grid_points, report);
               });
    return parameters;
}

}  // namespace

ParsedProblem read_problem_spec(const std::string& path,
                                const SolverOptions& options) {
    ParsedProblem parsed;
    ValidationReport& report = parsed.report;
    ProblemSpec& p = parsed.spec;

    const json::Value root = json::parse_file(path);
    if (!root.is_object()) {
        throw std::runtime_error("top-level JSON value must be an object");
    }

    // Helper: apply fn(v, key) when root has the key.
    auto if_present = [&](std::string_view key, auto fn) {
        if (root.contains(std::string(key))) fn(root.at(std::string(key)), key);
    };

    // ---- scalars ----
    if_present("mpol", [&](const json::Value& v, std::string_view k) {
        p.mpol = get_int(v, k, p.mpol, report);
    });
    if_present("ntor", [&](const json::Value& v, std::string_view k) {
        p.ntor = get_int(v, k, p.ntor, report);
    });
    if_present("nfp", [&](const json::Value& v, std::string_view k) {
        p.nfp = get_int(v, k, p.nfp, report);
    });
    if_present("ntheta", [&](const json::Value& v, std::string_view k) {
        p.angular.ntheta = get_int(v, k, p.angular.ntheta, report);
    });
    if_present("nzeta", [&](const json::Value& v, std::string_view k) {
        p.angular.nzeta = get_int(v, k, p.angular.nzeta, report);
    });
    if_present("ncurr", [&](const json::Value& v, std::string_view k) {
        const int nc = get_int(v, k, 0, report);
        if (nc == 0)
            p.current_model = CurrentModel::FIXED_IOTA;
        else if (nc == 1)
            p.current_model = CurrentModel::PRESCRIBED_CURRENT;
        else
            report.error(std::string(k), "ncurr must be 0 or 1");
    });
    if_present("delt", [&](const json::Value& v, std::string_view k) {
        p.delt = get_double(v, k, p.delt, report);
    });
    if_present("phiedge", [&](const json::Value& v, std::string_view k) {
        p.physical.phiedge = get_double(v, k, p.physical.phiedge, report);
    });
    if_present("pres_scale", [&](const json::Value& v, std::string_view k) {
        p.physical.pres_scale = get_double(v, k, p.physical.pres_scale, report);
    });
    // "adiabatic_index" is the legacy alias for vmecpp's "gamma".
    if (root.contains("adiabatic_index")) {
        p.physical.adiabatic_index =
            get_double(root.at("adiabatic_index"), "adiabatic_index",
                       p.physical.adiabatic_index, report);
    } else if (root.contains("gamma")) {
        p.physical.adiabatic_index = get_double(
            root.at("gamma"), "gamma", p.physical.adiabatic_index, report);
    }
    if_present("spres_ped", [&](const json::Value& v, std::string_view k) {
        p.physical.spres_ped = get_double(v, k, p.physical.spres_ped, report);
    });
    if_present("curtor", [&](const json::Value& v, std::string_view k) {
        p.physical.curtor = get_double(v, k, p.physical.curtor, report);
    });
    if_present("bloat", [&](const json::Value& v, std::string_view k) {
        p.physical.bloat = get_double(v, k, p.physical.bloat, report);
    });
    if_present("tcon0", [&](const json::Value& v, std::string_view k) {
        p.physical.tcon0 = get_double(v, k, p.physical.tcon0, report);
    });

    // ---- free boundary ----
    if_present("lfreeb", [&](const json::Value& v, std::string_view k) {
        p.free_boundary.lfreeb = get_bool(v, k, p.free_boundary.lfreeb, report);
    });
    if_present("mgrid_file", [&](const json::Value& v, std::string_view k) {
        p.free_boundary.mgrid_file =
            get_string(v, k, p.free_boundary.mgrid_file, report);
    });
    if_present("coils_file", [&](const json::Value& v, std::string_view k) {
        p.free_boundary.coils_file =
            get_string(v, k, p.free_boundary.coils_file, report);
    });
    if_present("makegrid_parameters_file",
               [&](const json::Value& v, std::string_view k) {
                   p.free_boundary.makegrid_parameters_file = get_string(
                       v, k, p.free_boundary.makegrid_parameters_file, report);
               });
    const bool has_makegrid_parameters = root.contains("makegrid_parameters");
    if_present("makegrid_parameters",
               [&](const json::Value& v, std::string_view k) {
                   p.free_boundary.embedded_makegrid_parameters =
                       read_makegrid_parameters(v, k, report);
               });
    if (root.contains("makegrid_parameters_file") && has_makegrid_parameters) {
        report.warn("makegrid_parameters",
                    "both makegrid_parameters_file and embedded "
                    "makegrid_parameters are present; using embedded "
                    "makegrid_parameters");
    }
    if_present("extcur", [&](const json::Value& v, std::string_view k) {
        p.free_boundary.extcur = read_double_array(v, k, report);
    });
    if_present("nvacskip", [&](const json::Value& v, std::string_view k) {
        p.free_boundary.nvacskip =
            get_int(v, k, p.free_boundary.nvacskip, report);
    });

    // ---- profile coefficients (power series) ----
    if_present("am", [&](const json::Value& v, std::string_view k) {
        p.mass.coefficients = read_double_array(v, k, report);
    });
    if_present("ac", [&](const json::Value& v, std::string_view k) {
        p.current.coefficients = read_double_array(v, k, report);
    });
    if_present("ai", [&](const json::Value& v, std::string_view k) {
        p.iota.coefficients = read_double_array(v, k, report);
    });
    if (root.contains("aphi")) {
        p.toroidal_flux.coefficients =
            read_double_array(root.at("aphi"), "aphi", report);
    } else {
        // vmecpp default. push_back instead of an initializer_list
        // assignment: g++-13 -O2 -Warray-bounds (warnings-as-errors) mis-
        // diagnoses the list's internal copy as an out-of-bounds memmove.
        p.toroidal_flux.coefficients.clear();
        p.toroidal_flux.coefficients.push_back(1.0);
    }

    // ---- magnetic axis ----
    if (root.contains("raxis_c")) {
        p.has_raxis_c = true;
        p.raxis_c = read_double_array(root.at("raxis_c"), "raxis_c", report);
    }
    if (root.contains("zaxis_s")) {
        p.has_zaxis_s = true;
        p.zaxis_s = read_double_array(root.at("zaxis_s"), "zaxis_s", report);
    }

    // ---- multi-radial-grid sequence (all-or-none) ----
    const bool has_ns = root.contains("ns_array");
    const bool has_niter = root.contains("niter_array");
    const bool has_ftol = root.contains("ftol_array");
    if (has_ns || has_niter || has_ftol) {
        if (!(has_ns && has_niter && has_ftol)) {
            report.error("ns_array",
                         "ns_array, niter_array and ftol_array must be "
                         "provided together");
        } else {
            const auto ns =
                read_int_array(root.at("ns_array"), "ns_array", report);
            const auto niter =
                read_int_array(root.at("niter_array"), "niter_array", report);
            const auto ftol =
                read_double_array(root.at("ftol_array"), "ftol_array", report);
            if (niter.size() != ns.size()) {
                report.error("niter_array",
                             "niter_array length must match ns_array");
            }
            if (ftol.size() != ns.size()) {
                report.error("ftol_array",
                             "ftol_array length must match ns_array");
            }
            const std::size_t ng =
                std::min(ns.size(), std::min(niter.size(), ftol.size()));
            for (std::size_t g = 0; g < ng; ++g) {
                // Clamp negative entries to 0 so a negative int never wraps to
                // SIZE_MAX; validate() then rejects 0 with the legacy message.
                p.stages.push_back(StageRequest{
                    static_cast<std::size_t>(ns[g] < 0 ? 0 : ns[g]),
                    static_cast<std::size_t>(niter[g] < 0 ? 0 : niter[g]),
                    ftol[g]});
            }
        }
    } else {
        p.stages.push_back(default_stage());
    }

    // ---- boundary ----
    if (root.contains("rbc"))
        read_boundary(root.at("rbc"), "rbc", p.rbc, report);
    if (root.contains("zbs"))
        read_boundary(root.at("zbs"), "zbs", p.zbs, report);

    // ---- unsupported features -> errors ----
    if (root.contains("lasym") &&
        get_bool(root.at("lasym"), "lasym", false, report)) {
        report.error(
            "lasym",
            "lasym=true: asymmetric equilibria are not supported by cuMES");
    }
    // "two_power" is supported for the mass (pressure) and current profiles;
    // it is NOT applicable to the iota profile (vmecpp marks it
    // allowedForIota=false), which stays a power series.
    auto read_profile_type = [&](std::string_view key, ProfileType& out,
                                 bool two_power_allowed) {
        if (!root.contains(std::string(key))) return;
        const std::string v =
            get_string(root.at(std::string(key)), key, "power_series", report);
        if (v == "power_series") {
            out = ProfileType::POWER_SERIES;
        } else if (v == "two_power" && two_power_allowed) {
            out = ProfileType::TWO_POWER;
        } else if (v == "two_power") {
            report.error(std::string(key),
                         "'" + std::string(key) +
                             "': \"two_power\" is not applicable to the "
                             "iota profile (power series only)");
        } else {
            report.error(std::string(key),
                         "'" + std::string(key) +
                             "': unsupported profile "
                             "type \"" +
                             v +
                             "\" (supported: \"power_series\", "
                             "\"two_power\")");
        }
    };
    read_profile_type("pmass_type", p.mass.type, true);
    read_profile_type("piota_type", p.iota.type, false);
    read_profile_type("pcurr_type", p.current.type, true);
    // Unsupported-feature keys are TYPE-CHECKED before the semantic support
    // check, so a scalar/object of the wrong type is a hard error rather than
    // silently ignored.
    constexpr std::array<std::string_view, 6> AUX_ARRAYS = {
        "am_aux_s", "am_aux_f", "ai_aux_s", "ai_aux_f", "ac_aux_s", "ac_aux_f"};
    for (std::string_view k : AUX_ARRAYS) {
        if (!root.contains(std::string(k))) continue;
        if (!root.at(std::string(k)).is_array()) {
            report.error(std::string(k),
                         "'" + std::string(k) + "': expected an array, got " +
                             json::get_value_category_name(
                                 root.at(std::string(k)).value_category()));
            continue;
        }
        if (root.at(std::string(k)).size() > 0) {
            report.error(std::string(k),
                         "'" + std::string(k) +
                             "': spline profile coefficients "
                             "are not supported by cuMES (power series only)");
        }
    }
    constexpr std::array<std::string_view, 4> ASYM_ARRAYS = {
        "raxis_s", "zaxis_c", "rbs", "zbc"};
    for (std::string_view k : ASYM_ARRAYS) {
        if (!root.contains(std::string(k))) continue;
        if (!root.at(std::string(k)).is_array()) {
            report.error(std::string(k),
                         "'" + std::string(k) + "': expected an array, got " +
                             json::get_value_category_name(
                                 root.at(std::string(k)).value_category()));
            continue;
        }
        if (root.at(std::string(k)).size() > 0) {
            report.error(std::string(k), "'" + std::string(k) +
                                             "': asymmetric (lasym) input is "
                                             "not supported by cuMES");
        }
    }

    // ---- unknown keys (strict: error; compatibility: warn) ----
    for (const auto& [key, _val] : root.as_object()) {
        if (SUPPORTED_KEYS.count(key) == 0 &&
            KNOWN_IGNORED_KEYS.count(key) == 0) {
            const std::string msg = "unknown input key '" + key + "'";
            if (options.strict_schema) {
                report.error(std::string(key), msg);
            } else {
                report.warn(key, msg);
            }
        }
    }

    return parsed;
}

ValidationResult read_and_validate(const std::string& path,
                                   const SolverOptions& options) {
    ParsedProblem parsed;
    try {
        parsed = read_problem_spec(path, options);
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
    auto vr = validate(std::move(parsed.spec), options);

    if (vr.has_value()) {
        // Success: fold mapping-phase warnings into the model (observability).
        // A mapping-phase ERROR still invalidates the model — it is a type
        // error that validate() cannot see, because the field kept its default.
        for (const auto& issue : parsed.report.issues()) {
            if (issue.severity == Severity::WARNING) {
                vr.value().add_warning(issue.key, issue.message);
            }
        }
        if (parsed.report.has_errors()) {
            return ValidationResult(parsed.report);
        }
        return vr;
    }

    // validate() failed: merge mapping findings into its report.
    ValidationReport combined = std::move(parsed.report);
    for (const auto& issue : vr.error().issues()) {
        if (issue.severity == Severity::ERROR) {
            combined.error(issue.key, issue.message);
        } else {
            combined.warn(issue.key, issue.message);
        }
    }
    return ValidationResult(combined);
}

}  // namespace cumes
