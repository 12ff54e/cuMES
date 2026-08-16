// input_json.cpp — JSON-file-driven input (vmecpp indata schema).
//
// The only TU that defines ZQ_JSON_PARSER_IMPLEMENTATION (JsonParser.h is
// C++20; the whole project builds at C++20, but the implementation is kept
// in this single TU). Maps a vmecpp-style flat JSON document onto
// InputParams, mirroring vmecpp's VmecINDATA::FromJson semantics: every
// key is optional, a wrong type is a hard error, and keys for features
// cuMES does not implement (lasym, free boundary, spline profiles) are
// rejected with a clear message. Unknown keys (outside the supported set
// and the known-but-unimplemented vmecpp set) warn to stderr instead of
// passing silently, so a typo'd key cannot silently change the physics.
//
// The parsed document is only ever read: probing goes through
// Value::contains (no Null-insertion side effect of the non-const
// operator[]), reads through at()/const operator[], so everything below
// takes const Value references.
//
// The JsonParser implementation is compiled once into the cumes_json target
// (src/json_parser.cpp); this TU only declares it.
#include "JsonParser.h"
#include "input_json.h"

#include <cmath>
#include <cstdio>  // fprintf
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

// Recognized input keys. kSupportedKeys is the union of every key the parser
// reads below; kKnownIgnoredKeys are vmecpp indata keys cuMES intentionally
// does not implement (they appear in real vmecpp files, so they must not
// warn). Anything outside both sets is a likely typo and warns (see the
// unknown-key scan at the end of initInputParamsFromJson).
const std::set<std::string> kSupportedKeys = {
    "mpol", "ntor", "nfp", "ntheta", "nzeta", "ncurr", "delt", "phiedge",
    "pres_scale", "adiabatic_index", "gamma", "spres_ped", "curtor", "bloat",
    "tcon0", "am", "ac", "ai", "aphi", "raxis_c", "zaxis_s", "ns_array",
    "niter_array", "ftol_array", "rbc", "zbs", "lasym", "lfreeb",
    "pmass_type", "piota_type", "pcurr_type", "am_aux_s", "am_aux_f",
    "ai_aux_s", "ai_aux_f", "ac_aux_s", "ac_aux_f", "raxis_s", "zaxis_c",
    "rbs", "zbc",
};
const std::set<std::string> kKnownIgnoredKeys = {
    // vmecpp indata keys cuMES does not implement (iteration cap is
    // niter_array here; free-boundary / vacuum / recon inputs are unused).
    "nstep", "niter", "ftolv", "nsurf", "tsw", "tpot", "tvac", "nvacskip",
    "mgrid_file", "extcur", "lforbal", "lmorebdy", "lrecon", "lmove_axis",
    "lthreed", "lpoloidal", "nthreed", "npoloidal", "nlambda", "lspectral",
    "lcheck", "lpsplot", "lwout", "lmask", "nedge", "nskip",
};

// Typed getters with key-named error messages.
int getInt(const json::Value& v, const char* key) {
    if (v.value_category() != json::ValueCategory::NumberInt) {
        fail(std::string("'") + key + "': expected an integer, got " +
             json::get_value_category_name(v.value_category()));
    }
    // The parser stores JSON integers as int64_t; range-check before the
    // narrowing to int so a huge literal can never silently wrap into a
    // valid-looking resolution (e.g. 2^32 + 5 -> 5).
    const long long vv = v.as_number<long long>();
    if (vv < INT_MIN || vv > INT_MAX) {
        fail(std::string("'") + key + "': integer value " + std::to_string(vv) +
             " is out of range");
    }
    return static_cast<int>(vv);
}
double getDouble(const json::Value& v, const char* key) {
    if (!v.is_number()) {
        fail(std::string("'") + key + "': expected a number, got " +
             json::get_value_category_name(v.value_category()));
    }
    return v.as_number<double>();
}
bool getBool(const json::Value& v, const char* key) {
    if (!v.is_boolean()) {
        fail(std::string("'") + key + "': expected a boolean, got " +
             json::get_value_category_name(v.value_category()));
    }
    return v.as_boolean();
}
std::string getString(const json::Value& v, const char* key) {
    if (!v.is_string()) {
        fail(std::string("'") + key + "': expected a string, got " +
             json::get_value_category_name(v.value_category()));
    }
    return v.as_string();
}

// Read an array of JSON numbers into a fixed-size C array; returns the
// element count. `strict_int` requires integer literals (ns_array etc.).
template <typename T>
int readNumberArray(const json::Value& v, const std::string& key, T* out,
                    int cap, bool strict_int) {
    if (!v.is_array()) {
        fail("'" + key + "': expected an array, got " +
             json::get_value_category_name(v.value_category()));
    }
    std::size_t n = v.size();
    if (n > static_cast<std::size_t>(cap)) {
        fail("'" + key + "': " + std::to_string(n) + " entries exceed the " +
             std::to_string(cap) + "-entry capacity");
    }
    for (std::size_t i = 0; i < n; ++i) {
        const json::Value& e = v[i];
        if (!e.is_number() ||
            (strict_int && e.value_category() != json::ValueCategory::NumberInt)) {
            fail("'" + key + "[" + std::to_string(i) + "]': expected " +
                 (strict_int ? "an integer" : "a number") + ", got " +
                 json::get_value_category_name(e.value_category()));
        }
        out[i] = static_cast<T>(e.as_number<double>());
    }
    return static_cast<int>(n);
}

// Boundary coefficients: array of {"n","m","value"} objects. Entries outside
// mpol/ntor are skipped with a warning (vmecpp logs and ignores them).
void readBoundary(const json::Value& v, const std::string& key,
                  BoundaryEntry* out, int& count, int mpol, int ntor) {
    if (!v.is_array()) {
        fail("'" + key + "': expected an array of {\"n\",\"m\",\"value\"} "
             "objects, got " + json::get_value_category_name(v.value_category()));
    }
    std::size_t n = v.size();
    if (n > 256) {
        fail("'" + key + "': " + std::to_string(n) +
             " boundary entries exceed the 256-entry capacity");
    }
    for (std::size_t i = 0; i < n; ++i) {
        const json::Value& e = v[i];
        const std::string where = key + "[" + std::to_string(i) + "]";
        if (!e.is_object()) {
            fail("'" + where + "': expected an object with \"n\", \"m\" "
                 "and \"value\"");
        }
        if (!e.contains("m") || !e.contains("n") || !e.contains("value")) {
            fail("'" + where + "': missing \"n\", \"m\" or \"value\"");
        }
        int m = getInt(e.at("m"), (where + ".m").c_str());
        int nn = getInt(e.at("n"), (where + ".n").c_str());
        double value = getDouble(e.at("value"), (where + ".value").c_str());
        // Reject m < 0 too: foldBoundary indexes the fixed [mpol][ntor+1]
        // tables with e.m, and a negative m would be a host out-of-bounds
        // write (only m >= mpol was checked before).
        if (m < 0 || m >= mpol || std::abs(nn) > ntor) {
            fprintf(stderr,
                    "cuMES: %s: skipping mode m=%d n=%d (outside 0<=m<%d, "
                    "|n|<=%d)\n", key.c_str(), m, nn, mpol, ntor);
            continue;
        }
        out[count++] = {m, nn, value};
    }
}

// Apply fn(v, key) when root has the key.
template <typename Fn>
void ifPresent(const json::Value& root, const char* key, Fn fn) {
    if (root.contains(key)) { fn(root.at(key), key); }
}

}  // namespace

InputParams initInputParamsFromJson(const char* json_path) {
    const std::string path = json_path ? json_path : "inputs/solovev.json";
    try {
        const json::Value root = json::parse_file(path);
        if (!root.is_object()) {
            fail("top-level JSON value must be an object");
        }
        // Member defaults are the baseline; every key is optional.
        InputParams p;

        // ---- scalars ----
        ifPresent(root, "mpol", [&](const json::Value& v, const char* k) { p.mpol = getInt(v, k); });
        ifPresent(root, "ntor", [&](const json::Value& v, const char* k) { p.ntor = getInt(v, k); });
        ifPresent(root, "nfp", [&](const json::Value& v, const char* k) { p.nfp = getInt(v, k); });
        ifPresent(root, "ntheta", [&](const json::Value& v, const char* k) { p.ntheta = getInt(v, k); });
        ifPresent(root, "nzeta", [&](const json::Value& v, const char* k) { p.nzeta = getInt(v, k); });
        ifPresent(root, "ncurr", [&](const json::Value& v, const char* k) { p.ncurr = getInt(v, k); });
        ifPresent(root, "delt", [&](const json::Value& v, const char* k) { p.delt = getDouble(v, k); });
        ifPresent(root, "phiedge", [&](const json::Value& v, const char* k) { p.phiedge = getDouble(v, k); });
        ifPresent(root, "pres_scale", [&](const json::Value& v, const char* k) { p.pres_scale = getDouble(v, k); });
        // "adiabatic_index" is the legacy alias for vmecpp's "gamma".
        if (root.contains("adiabatic_index")) {
            p.adiabatic_index = getDouble(root.at("adiabatic_index"), "adiabatic_index");
        } else if (root.contains("gamma")) {
            p.adiabatic_index = getDouble(root.at("gamma"), "gamma");
        }
        ifPresent(root, "spres_ped", [&](const json::Value& v, const char* k) { p.spres_ped = getDouble(v, k); });
        ifPresent(root, "curtor", [&](const json::Value& v, const char* k) { p.curtor = getDouble(v, k); });
        ifPresent(root, "bloat", [&](const json::Value& v, const char* k) { p.bloat = getDouble(v, k); });
        ifPresent(root, "tcon0", [&](const json::Value& v, const char* k) { p.tcon0 = getDouble(v, k); });

        // ---- consistency checks (vmecpp VmecINDATA::IsConsistent) ----
        if (p.mpol < 2 || p.mpol > 16) fail("mpol must be in [2, 16]");
        if (p.ntor < 0 || p.ntor > 15) fail("ntor must be in [0, 15]");
        if (p.nfp < 1) fail("nfp must be >= 1");
        if (p.ncurr != 0 && p.ncurr != 1) fail("ncurr must be 0 or 1");
        if (p.phiedge == 0.0) fail("phiedge must be nonzero");
        if (p.delt <= 0.0) fail("delt must be positive");
        // Angular extents: 0 = resolution default (applyResolutionDefaults);
        // the caps keep the products ntheta*nzeta within int range and the
        // poloidal-accumulation launch blocks valid (they embed ntheta/2 as
        // blockDim.x, capped at 128 by ntheta <= 256).
        if (p.ntheta < 0 || p.ntheta > 256) fail("ntheta must be in [0, 256] (0 = default)");
        if (p.nzeta < 0 || p.nzeta > 256) fail("nzeta must be in [0, 256] (0 = default)");
        // adiabatic_index (gamma) is accepted by vmecpp but NOT implemented
        // by cuMES: profiles.cu always computes pres = mass (the gamma=0
        // model). Accepting a nonzero value would silently solve a different
        // physics problem than the input advertises.
        if (p.adiabatic_index != 0.0) {
            fail("adiabatic_index (gamma) must be 0: the gamma != 0 model "
                 "(pres = mass/dVds^gamma) is not implemented by cuMES");
        }

        // ---- profile coefficients (power series) ----
        ifPresent(root, "am", [&](const json::Value& v, const char* k) {
            p.am_n = readNumberArray(v, k, p.am, InputParams::kMaxCoeff, false);
        });
        ifPresent(root, "ac", [&](const json::Value& v, const char* k) {
            p.ac_n = readNumberArray(v, k, p.ac, InputParams::kMaxCoeff, false);
        });
        ifPresent(root, "ai", [&](const json::Value& v, const char* k) {
            p.ai_n = readNumberArray(v, k, p.ai, InputParams::kMaxCoeff, false);
        });
        if (root.contains("aphi")) {
            p.aphi_n = readNumberArray(root.at("aphi"), "aphi", p.aphi,
                                       InputParams::kMaxCoeff, false);
        } else {
            p.aphi[0] = 1.0;  // vmecpp default (solovev.json omits aphi)
            p.aphi_n = 1;
        }

        // ---- magnetic axis ----
        if (root.contains("raxis_c")) {
            p.raxis_n = readNumberArray(root.at("raxis_c"), "raxis_c", p.raxis_c, 32, false);
            if (p.raxis_n != p.ntor + 1) fail("raxis_c must have exactly ntor+1 entries");
        }
        if (root.contains("zaxis_s")) {
            int zaxis_n = readNumberArray(root.at("zaxis_s"), "zaxis_s", p.zaxis_s, 32, false);
            if (zaxis_n != p.ntor + 1) fail("zaxis_s must have exactly ntor+1 entries");
        }

        // ---- multi-radial-grid sequence (all-or-none) ----
        const bool hasNs = root.contains("ns_array");
        const bool hasNiter = root.contains("niter_array");
        const bool hasFtol = root.contains("ftol_array");
        if (hasNs || hasNiter || hasFtol) {
            if (!(hasNs && hasNiter && hasFtol)) {
                fail("ns_array, niter_array and ftol_array must be provided together");
            }
            p.n_grids = readNumberArray(root.at("ns_array"), "ns_array", p.ns_array,
                                        InputParams::kMaxGrids, true);
            if (readNumberArray(root.at("niter_array"), "niter_array", p.niter_array,
                                InputParams::kMaxGrids, true) != p.n_grids) {
                fail("niter_array length must match ns_array");
            }
            if (readNumberArray(root.at("ftol_array"), "ftol_array", p.ftol_array,
                                InputParams::kMaxGrids, false) != p.n_grids) {
                fail("ftol_array length must match ns_array");
            }
            // All three arrays present but EMPTY is malformed: main would
            // skip the stage loop entirely and save/print a null state.
            if (p.n_grids < 1) fail("ns_array must contain at least one stage");
            for (int g = 0; g < p.n_grids; ++g) {
                // ns cap: keeps the tridiagonal-solve dynamic shared memory
                // (10*ns*sizeof(T)) within the 48KB default limit and the
                // angular-launch blocks valid.
                if (p.ns_array[g] < 3 || p.ns_array[g] > 512) {
                    fail("ns_array entries must be in [3, 512]");
                }
                if (g > 0 && p.ns_array[g] < p.ns_array[g - 1]) {
                    fail("ns_array must be monotonically non-decreasing");
                }
                if (p.niter_array[g] < 1) fail("niter_array entries must be >= 1");
                if (p.ftol_array[g] <= 0.0) fail("ftol_array entries must be positive");
            }
        } else {
            p.ns_array[0] = p.ns;
            p.niter_array[0] = p.max_iter;
            p.ftol_array[0] = p.ftol;
        }

        // ---- boundary ----
        if (root.contains("rbc")) readBoundary(root.at("rbc"), "rbc", p.rbc, p.rbc_n, p.mpol, p.ntor);
        if (root.contains("zbs")) readBoundary(root.at("zbs"), "zbs", p.zbs, p.zbs_n, p.mpol, p.ntor);
        if (p.rbc_n == 0) fail("rbc: at least one boundary coefficient is required");
        if (p.zbs_n == 0) fail("zbs: at least one boundary coefficient is required");

        // ---- unsupported features -> hard error ----
        if (root.contains("lasym") && getBool(root.at("lasym"), "lasym")) {
            fail("lasym=true: asymmetric equilibria are not supported by cuMES");
        }
        if (root.contains("lfreeb") && getBool(root.at("lfreeb"), "lfreeb")) {
            fail("lfreeb=true: free-boundary runs are not supported by cuMES (fixed boundary only)");
        }
        const char* kProfileTypes[] = {"pmass_type", "piota_type", "pcurr_type"};
        for (const char* t : kProfileTypes) {
            if (root.contains(t)) {
                const std::string v = getString(root.at(t), t);
                if (v != "power_series") {
                    fail(std::string("'") + t + "': only \"power_series\" "
                         "profiles are supported by cuMES, got \"" + v + "\"");
                }
            }
        }
        // Unsupported-feature keys are TYPE-CHECKED before the semantic
        // support check: a scalar or object of the wrong type must be a hard
        // error, not silently ignored (the old `is_array() && size() > 0`
        // condition accepted e.g. "am_aux_s": 5 as if the key were absent).
        const char* kAuxArrays[] = {"am_aux_s", "am_aux_f", "ai_aux_s",
                                    "ai_aux_f", "ac_aux_s", "ac_aux_f"};
        for (const char* k : kAuxArrays) {
            if (!root.contains(k)) continue;
            if (!root.at(k).is_array()) {
                fail(std::string("'") + k + "': expected an array, got " +
                     json::get_value_category_name(root.at(k).value_category()));
            }
            if (root.at(k).size() > 0) {
                fail(std::string("'") + k + "': spline profile coefficients "
                     "are not supported by cuMES (power series only)");
            }
        }
        const char* kAsymArrays[] = {"raxis_s", "zaxis_c", "rbs", "zbc"};
        for (const char* k : kAsymArrays) {
            if (!root.contains(k)) continue;
            if (!root.at(k).is_array()) {
                fail(std::string("'") + k + "': expected an array, got " +
                     json::get_value_category_name(root.at(k).value_category()));
            }
            if (root.at(k).size() > 0) {
                fail(std::string("'") + k + "': asymmetric (lasym) input is "
                     "not supported by cuMES");
            }
        }

        // ---- unknown keys ----
        // A typo'd key (e.g. "n_theta") must not pass silently — it would
        // silently change the physics to the defaults. Keys outside the
        // supported set and the known-vmecpp-but-unimplemented set get a
        // stderr warning (rejection would break real vmecpp files that
        // carry e.g. mgrid_file/extcur/nvacskip).
        for (const auto& [key, _val] : root.as_object()) {
            if (kSupportedKeys.count(key) == 0 && kKnownIgnoredKeys.count(key) == 0) {
                fprintf(stderr,
                        "cuMES: WARNING: unknown input key '%s' ignored\n",
                        key.c_str());
            }
        }

        // ---- finalize ----
        applyResolutionDefaults(p);
        // Stage-0 invariant: DeviceParams and the tests read p.ns/max_iter/ftol
        // directly; they must equal the stage-0 multigrid entries.
        p.ns = p.ns_array[0];
        p.max_iter = p.niter_array[0];
        p.ftol = p.ftol_array[0];
        p.raxis_n = p.ntor + 1;  // vmecpp zeroes raxis_c/zaxis_s to ntor+1
        foldBoundary(p);
        return p;
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
}
