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
    "mpol",       "ntor",      "nfp",         "ntheta",     "nzeta",
    "ncurr",      "delt",      "phiedge",     "pres_scale", "adiabatic_index",
    "gamma",      "spres_ped", "curtor",      "bloat",      "tcon0",
    "am",         "ac",        "ai",          "aphi",       "raxis_c",
    "zaxis_s",    "ns_array",  "niter_array", "ftol_array", "rbc",
    "zbs",        "lasym",     "lfreeb",      "pmass_type", "piota_type",
    "pcurr_type", "am_aux_s",  "am_aux_f",    "ai_aux_s",   "ai_aux_f",
    "ac_aux_s",   "ac_aux_f",  "raxis_s",     "zaxis_c",    "rbs",
    "zbc",
};
const std::set<std::string> KNOWN_IGNORED_KEYS = {
    "nstep",   "niter",      "ftolv",      "nsurf",     "tsw",     "tpot",
    "tvac",    "nvacskip",   "mgrid_file", "extcur",    "lforbal", "lmorebdy",
    "lrecon",  "lmove_axis", "lthreed",    "lpoloidal", "nthreed", "npoloidal",
    "nlambda", "lspectral",  "lcheck",     "lpsplot",   "lwout",   "lmask",
    "nedge",   "nskip",
};

// Typed getters: on a type/range error record the finding and return the
// fallback so the mapped field keeps a valid default (validate() then reports
// any remaining semantic issue without a misleading cascade).
int getInt(const json::Value& v,
           const std::string& key,
           int fallback,
           ValidationReport& report) {
    if (v.value_category() != json::ValueCategory::NumberInt) {
        report.error(key,
                     "'" + key + "': expected an integer, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    const long long vv = v.as_number<long long>();
    if (vv < INT_MIN || vv > INT_MAX) {
        report.error(key, "'" + key + "': integer value " + std::to_string(vv) +
                              " is out of range");
        return fallback;
    }
    return static_cast<int>(vv);
}

double getDouble(const json::Value& v,
                 const std::string& key,
                 double fallback,
                 ValidationReport& report) {
    if (!v.is_number()) {
        report.error(key,
                     "'" + key + "': expected a number, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    return v.as_number<double>();
}

bool getBool(const json::Value& v,
             const std::string& key,
             bool fallback,
             ValidationReport& report) {
    if (!v.is_boolean()) {
        report.error(key,
                     "'" + key + "': expected a boolean, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    return v.as_boolean();
}

std::string getString(const json::Value& v,
                      const std::string& key,
                      std::string fallback,
                      ValidationReport& report) {
    if (!v.is_string()) {
        report.error(key,
                     "'" + key + "': expected a string, got " +
                         json::get_value_category_name(v.value_category()));
        return fallback;
    }
    return v.as_string();
}

// Read an array of JSON numbers into a dynamic vector. strict_int requires
// integer literals and range-checks the narrowing to int.
std::vector<int> readIntArray(const json::Value& v,
                              const std::string& key,
                              ValidationReport& report) {
    std::vector<int> out;
    if (!v.is_array()) {
        report.error(key,
                     "'" + key + "': expected an array, got " +
                         json::get_value_category_name(v.value_category()));
        return out;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        const json::Value& e = v[i];
        if (e.value_category() != json::ValueCategory::NumberInt) {
            report.error(key,
                         "'" + key + "[" + std::to_string(i) +
                             "]': expected an integer, got " +
                             json::get_value_category_name(e.value_category()));
            out.push_back(0);
            continue;
        }
        const long long vv = e.as_number<long long>();
        if (vv < INT_MIN || vv > INT_MAX) {
            report.error(key, "'" + key + "[" + std::to_string(i) +
                                  "]': integer value " + std::to_string(vv) +
                                  " is out of range");
            out.push_back(0);
            continue;
        }
        out.push_back(static_cast<int>(vv));
    }
    return out;
}

std::vector<double> readDoubleArray(const json::Value& v,
                                    const std::string& key,
                                    ValidationReport& report) {
    std::vector<double> out;
    if (!v.is_array()) {
        report.error(key,
                     "'" + key + "': expected an array, got " +
                         json::get_value_category_name(v.value_category()));
        return out;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        const json::Value& e = v[i];
        if (!e.is_number()) {
            report.error(key,
                         "'" + key + "[" + std::to_string(i) +
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
void readBoundary(const json::Value& v,
                  const std::string& key,
                  std::vector<BoundaryHarmonic>& out,
                  ValidationReport& report) {
    if (!v.is_array()) {
        report.error(key,
                     "'" + key +
                         "': expected an array of "
                         "{\"n\",\"m\",\"value\"} objects, got " +
                         json::get_value_category_name(v.value_category()));
        return;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        const json::Value& e = v[i];
        const std::string where = key + "[" + std::to_string(i) + "]";
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
        const int m = getInt(e.at("m"), where + ".m", 0, report);
        const int n = getInt(e.at("n"), where + ".n", 0, report);
        const double value =
            getDouble(e.at("value"), where + ".value", 0.0, report);
        out.push_back(BoundaryHarmonic{m, n, value});
    }
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
    auto ifPresent = [&](const char* key, auto fn) {
        if (root.contains(key)) fn(root.at(key), key);
    };

    // ---- scalars ----
    ifPresent("mpol", [&](const json::Value& v, const char* k) {
        p.mpol = getInt(v, k, p.mpol, report);
    });
    ifPresent("ntor", [&](const json::Value& v, const char* k) {
        p.ntor = getInt(v, k, p.ntor, report);
    });
    ifPresent("nfp", [&](const json::Value& v, const char* k) {
        p.nfp = getInt(v, k, p.nfp, report);
    });
    ifPresent("ntheta", [&](const json::Value& v, const char* k) {
        p.angular.ntheta = getInt(v, k, p.angular.ntheta, report);
    });
    ifPresent("nzeta", [&](const json::Value& v, const char* k) {
        p.angular.nzeta = getInt(v, k, p.angular.nzeta, report);
    });
    ifPresent("ncurr", [&](const json::Value& v, const char* k) {
        const int nc = getInt(v, k, 0, report);
        if (nc == 0)
            p.current_model = CurrentModel::FIXED_IOTA;
        else if (nc == 1)
            p.current_model = CurrentModel::PRESCRIBED_CURRENT;
        else
            report.error(k, "ncurr must be 0 or 1");
    });
    ifPresent("delt", [&](const json::Value& v, const char* k) {
        p.delt = getDouble(v, k, p.delt, report);
    });
    ifPresent("phiedge", [&](const json::Value& v, const char* k) {
        p.physical.phiedge = getDouble(v, k, p.physical.phiedge, report);
    });
    ifPresent("pres_scale", [&](const json::Value& v, const char* k) {
        p.physical.pres_scale = getDouble(v, k, p.physical.pres_scale, report);
    });
    // "adiabatic_index" is the legacy alias for vmecpp's "gamma".
    if (root.contains("adiabatic_index")) {
        p.physical.adiabatic_index =
            getDouble(root.at("adiabatic_index"), "adiabatic_index",
                      p.physical.adiabatic_index, report);
    } else if (root.contains("gamma")) {
        p.physical.adiabatic_index = getDouble(
            root.at("gamma"), "gamma", p.physical.adiabatic_index, report);
    }
    ifPresent("spres_ped", [&](const json::Value& v, const char* k) {
        p.physical.spres_ped = getDouble(v, k, p.physical.spres_ped, report);
    });
    ifPresent("curtor", [&](const json::Value& v, const char* k) {
        p.physical.curtor = getDouble(v, k, p.physical.curtor, report);
    });
    ifPresent("bloat", [&](const json::Value& v, const char* k) {
        p.physical.bloat = getDouble(v, k, p.physical.bloat, report);
    });
    ifPresent("tcon0", [&](const json::Value& v, const char* k) {
        p.physical.tcon0 = getDouble(v, k, p.physical.tcon0, report);
    });

    // ---- profile coefficients (power series) ----
    ifPresent("am", [&](const json::Value& v, const char* k) {
        p.mass.coefficients = readDoubleArray(v, k, report);
    });
    ifPresent("ac", [&](const json::Value& v, const char* k) {
        p.current.coefficients = readDoubleArray(v, k, report);
    });
    ifPresent("ai", [&](const json::Value& v, const char* k) {
        p.iota.coefficients = readDoubleArray(v, k, report);
    });
    if (root.contains("aphi")) {
        p.toroidal_flux.coefficients =
            readDoubleArray(root.at("aphi"), "aphi", report);
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
        p.raxis_c = readDoubleArray(root.at("raxis_c"), "raxis_c", report);
    }
    if (root.contains("zaxis_s")) {
        p.has_zaxis_s = true;
        p.zaxis_s = readDoubleArray(root.at("zaxis_s"), "zaxis_s", report);
    }

    // ---- multi-radial-grid sequence (all-or-none) ----
    const bool hasNs = root.contains("ns_array");
    const bool hasNiter = root.contains("niter_array");
    const bool hasFtol = root.contains("ftol_array");
    if (hasNs || hasNiter || hasFtol) {
        if (!(hasNs && hasNiter && hasFtol)) {
            report.error("ns_array",
                         "ns_array, niter_array and ftol_array must be "
                         "provided together");
        } else {
            const auto ns =
                readIntArray(root.at("ns_array"), "ns_array", report);
            const auto niter =
                readIntArray(root.at("niter_array"), "niter_array", report);
            const auto ftol =
                readDoubleArray(root.at("ftol_array"), "ftol_array", report);
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
        readBoundary(root.at("rbc"), "rbc", p.rbc, report);
    if (root.contains("zbs"))
        readBoundary(root.at("zbs"), "zbs", p.zbs, report);

    // ---- unsupported features -> errors ----
    if (root.contains("lasym") &&
        getBool(root.at("lasym"), "lasym", false, report)) {
        report.error(
            "lasym",
            "lasym=true: asymmetric equilibria are not supported by cuMES");
    }
    if (root.contains("lfreeb") &&
        getBool(root.at("lfreeb"), "lfreeb", false, report)) {
        report.error("lfreeb",
                     "lfreeb=true: free-boundary runs are not supported by "
                     "cuMES (fixed boundary only)");
    }
    // "two_power" is supported for the mass (pressure) and current profiles;
    // it is NOT applicable to the iota profile (vmecpp marks it
    // allowedForIota=false), which stays a power series.
    auto readProfileType = [&](const char* key, ProfileType& out,
                               bool twoPowerAllowed) {
        if (!root.contains(key)) return;
        const std::string v =
            getString(root.at(key), key, "power_series", report);
        if (v == "power_series") {
            out = ProfileType::POWER_SERIES;
        } else if (v == "two_power" && twoPowerAllowed) {
            out = ProfileType::TWO_POWER;
        } else if (v == "two_power") {
            report.error(key, "'" + std::string(key) +
                                  "': \"two_power\" is not applicable to the "
                                  "iota profile (power series only)");
        } else {
            report.error(key, "'" + std::string(key) +
                                  "': unsupported profile "
                                  "type \"" +
                                  v +
                                  "\" (supported: \"power_series\", "
                                  "\"two_power\")");
        }
    };
    readProfileType("pmass_type", p.mass.type, true);
    readProfileType("piota_type", p.iota.type, false);
    readProfileType("pcurr_type", p.current.type, true);
    // Unsupported-feature keys are TYPE-CHECKED before the semantic support
    // check, so a scalar/object of the wrong type is a hard error rather than
    // silently ignored.
    const char* AUX_ARRAYS[] = {"am_aux_s", "am_aux_f", "ai_aux_s",
                                "ai_aux_f", "ac_aux_s", "ac_aux_f"};
    for (const char* k : AUX_ARRAYS) {
        if (!root.contains(k)) continue;
        if (!root.at(k).is_array()) {
            report.error(
                k,
                "'" + std::string(k) + "': expected an array, got " +
                    json::get_value_category_name(root.at(k).value_category()));
            continue;
        }
        if (root.at(k).size() > 0) {
            report.error(k,
                         "'" + std::string(k) +
                             "': spline profile coefficients "
                             "are not supported by cuMES (power series only)");
        }
    }
    const char* ASYM_ARRAYS[] = {"raxis_s", "zaxis_c", "rbs", "zbc"};
    for (const char* k : ASYM_ARRAYS) {
        if (!root.contains(k)) continue;
        if (!root.at(k).is_array()) {
            report.error(
                k,
                "'" + std::string(k) + "': expected an array, got " +
                    json::get_value_category_name(root.at(k).value_category()));
            continue;
        }
        if (root.at(k).size() > 0) {
            report.error(k, "'" + std::string(k) +
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
                report.error(key, msg);
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
