// Byte-exact comparison of two cuMES run trees (Class A gate).

#include "../include/clap.h"
#define CUMES_COMPARE_COMMON_IMPLEMENTATION
#include "include/compare_common.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CommandLine {
    std::string baseline_directory;
    std::string run_directory;
    bool full = false;
    bool verbose = false;
};

int parse_command_line(CommandLine& command, int argc, char** argv) {
    CLAP_BEGIN(CommandLine)
    CLAP_ADD_USAGE("BASELINE_DIR RUN_DIR [--full] [--verbose]")
    CLAP_ADD_DESCRIPTION(
        "Byte-compare cuMES state payloads, trajectories, snapshots, and "
        "dumps.")
    CLAP_REGISTER_ARG(baseline_directory)
    CLAP_REGISTER_ARG(run_directory)
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(full, "--full",
                                          "compare every file under dump/cuMES")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        verbose, "--verbose", "report matching snapshots and family diffs")
    CLAP_END(CommandLine)
    try {
        CLAP<CommandLine>::parse_input(command, argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what();
        return EINVAL;
    }
    return 0;
}

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::map<std::filesystem::path, std::string> parse_manifest(
    const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("could not open " + path.string());
    }
    std::map<std::filesystem::path, std::string> entries;
    std::string line;
    while (std::getline(stream, line)) {
        line = trim(std::move(line));
        if (line.empty() || line.front() == '#') continue;
        const auto separator = line.find_first_of(" \t");
        if (separator == std::string::npos) {
            throw std::runtime_error("malformed SHA-256 manifest: " +
                                     path.string());
        }
        const std::string digest = line.substr(0, separator);
        const auto relative_start = line.find_first_not_of(" \t", separator);
        if (digest.size() != 64 || relative_start == std::string::npos) {
            throw std::runtime_error("malformed SHA-256 manifest: " +
                                     path.string());
        }
        entries[std::filesystem::path("dump") / "cuMES" /
                line.substr(relative_start)] = digest;
    }
    return entries;
}

void report(const std::string& label,
            bool pass,
            const std::string& detail = {}) {
    std::cout << "  [" << (pass ? "OK " : "DIFF") << "] " << label;
    if (!detail.empty()) std::cout << "  (" << detail << ')';
    std::cout << '\n';
}

std::string path_list(const std::vector<std::filesystem::path>& paths,
                      std::size_t limit) {
    std::ostringstream output;
    output << '[';
    const auto count = std::min(limit, paths.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (index) output << ", ";
        output << paths[index].generic_string();
    }
    if (paths.size() > count) output << " ...";
    output << ']';
    return output.str();
}

bool same_hash(const std::filesystem::path& a, const std::filesystem::path& b) {
    return cumes::compare::sha256_file(a) == cumes::compare::sha256_file(b);
}

std::vector<std::filesystem::path> set_difference(
    const std::vector<std::filesystem::path>& a,
    const std::vector<std::filesystem::path>& b) {
    std::vector<std::filesystem::path> difference;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(difference));
    return difference;
}

}  // namespace

int main(int argc, char** argv) {
    CommandLine command;
    if (const int status = parse_command_line(command, argc, argv); status) {
        return status;
    }

    namespace fs = std::filesystem;
    const fs::path baseline = command.baseline_directory;
    const fs::path run = command.run_directory;
    if (!fs::is_directory(baseline)) {
        std::cerr << "error: baseline dir not found: " << baseline << '\n';
        return 2;
    }
    if (!fs::is_directory(run)) {
        std::cerr << "error: run dir not found: " << run << '\n';
        return 2;
    }

    try {
        int failures = 0;

        const fs::path baseline_state = baseline / "cumes_state.bin";
        const fs::path run_state = run / "cumes_state.bin";
        if (!fs::is_regular_file(baseline_state) ||
            !fs::is_regular_file(run_state)) {
            std::cerr
                << "error: cumes_state.bin missing from baseline or run\n";
            return 2;
        }
        try {
            const auto left =
                cumes::compare::read_state_payload(baseline_state);
            const auto right = cumes::compare::read_state_payload(run_state);
            if (left.bytes != right.bytes) {
                report("cumes_state.bin", false, "state payload byte mismatch");
                ++failures;
                if (command.verbose && left.ns == right.ns &&
                    left.mnmax == right.mnmax) {
                    const auto a =
                        cumes::compare::read_state(baseline_state, false);
                    const auto b = cumes::compare::read_state(run_state, false);
                    for (std::size_t family = 0;
                         family < cumes::compare::FAMILY_NAMES.size();
                         ++family) {
                        double largest = 0.0;
                        for (std::size_t index = 0;
                             index < a.families[family].size(); ++index) {
                            largest = std::max(
                                largest, std::abs(a.families[family][index] -
                                                  b.families[family][index]));
                        }
                        std::cout << "      " << std::left << std::setw(6)
                                  << cumes::compare::FAMILY_NAMES[family]
                                  << std::right
                                  << ": max abs diff = " << std::scientific
                                  << std::setprecision(3) << largest << '\n';
                    }
                }
            } else {
                report("cumes_state.bin", true);
            }
        } catch (const std::exception&) {
            report("cumes_state.bin", false, "not a valid v1 container");
            ++failures;
        }

        const fs::path trajectory = "per_iter_residuals_cumes.bin";
        const fs::path baseline_trajectory = baseline / trajectory;
        const fs::path run_trajectory = run / trajectory;
        if (!fs::is_regular_file(baseline_trajectory) ||
            !fs::is_regular_file(run_trajectory)) {
            std::cerr << "error: " << trajectory.string()
                      << " missing from baseline or run\n";
            return 2;
        }
        if (!same_hash(baseline_trajectory, run_trajectory)) {
            report(trajectory.string(), false, "byte mismatch");
            ++failures;
        } else {
            report(trajectory.string(), true);
        }

        const auto baseline_initial =
            cumes::compare::files_with_prefix(baseline, "init_");
        const auto run_initial =
            cumes::compare::files_with_prefix(run, "init_");
        if (baseline_initial != run_initial) {
            report("init set", false,
                   "name sets differ: baseline " +
                       path_list(baseline_initial, 10) + " vs run " +
                       path_list(run_initial, 10));
            ++failures;
        } else {
            int initial_failures = 0;
            for (const auto& name : baseline_initial) {
                if (!same_hash(baseline / name, run / name)) {
                    report(name.string(), false, "byte mismatch");
                    ++failures;
                    ++initial_failures;
                } else if (command.verbose) {
                    report(name.string(), true);
                }
            }
            if (initial_failures == 0) {
                report("init set (" + std::to_string(baseline_initial.size()) +
                           " files)",
                       true);
            }
        }

        const fs::path baseline_manifest = baseline / "dump_manifest.sha256";
        if (fs::is_regular_file(baseline_manifest)) {
            const auto expected = parse_manifest(baseline_manifest);
            const auto run_dumps = cumes::compare::dump_paths(run);
            const std::set<fs::path> run_set(run_dumps.begin(),
                                             run_dumps.end());
            std::vector<fs::path> missing;
            for (const auto& [relative, digest] : expected) {
                (void)digest;
                if (run_set.find(relative) == run_set.end()) {
                    missing.push_back(relative);
                }
            }
            if (!missing.empty()) {
                report("dump manifest", false,
                       std::to_string(missing.size()) +
                           " expected dump files missing from run: " +
                           path_list(missing, 5));
                ++failures;
            } else {
                std::vector<fs::path> mismatched;
                for (const auto& [relative, digest] : expected) {
                    if (cumes::compare::sha256_file(run / relative) != digest) {
                        mismatched.push_back(relative);
                    }
                }
                if (!mismatched.empty()) {
                    report("dump manifest", false,
                           std::to_string(mismatched.size()) + "/" +
                               std::to_string(expected.size()) +
                               " files differ from baseline checksums: " +
                               path_list(mismatched, 5));
                    ++failures;
                } else {
                    report("dump manifest (" + std::to_string(expected.size()) +
                               " files)",
                           true);
                }
            }
        } else if (command.full) {
            std::cerr << "  (no baseline dump_manifest.sha256; --full needs "
                         "both trees to keep the full dump set)\n";
        }

        if (command.full) {
            const auto baseline_dumps = cumes::compare::dump_paths(baseline);
            const auto run_dumps = cumes::compare::dump_paths(run);
            if (baseline_dumps != run_dumps) {
                const auto only_baseline =
                    set_difference(baseline_dumps, run_dumps);
                const auto only_run = set_difference(run_dumps, baseline_dumps);
                report("full dump set", false,
                       "set differs (only baseline: " +
                           path_list(only_baseline, 3) +
                           "; only run: " + path_list(only_run, 3) + ')');
                ++failures;
            } else {
                int dump_failures = 0;
                for (const auto& relative : baseline_dumps) {
                    if (!same_hash(baseline / relative, run / relative)) {
                        report(relative.generic_string(), false,
                               "byte mismatch");
                        ++failures;
                        ++dump_failures;
                    }
                }
                if (dump_failures == 0) {
                    report("full dump set (" +
                               std::to_string(baseline_dumps.size()) +
                               " files)",
                           true);
                }
            }
        }

        std::cout << '\n';
        if (failures) {
            std::cout << "FAIL: " << failures << " difference(s) found\n";
            return EXIT_FAILURE;
        }
        std::cout << "PASS: byte-identical\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
