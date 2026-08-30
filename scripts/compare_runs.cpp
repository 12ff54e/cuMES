// Compare two cuMES trajectories and, optionally, their converged states.

#include "../include/clap.h"
#include "include/compare_common.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct CommandLine {
    std::string log_a;
    std::string state_a;
    std::string log_b;
    std::string state_b;
    bool no_state = false;
    double tolerance = 1.0e-8;
    double log_tolerance = 5.0e-3;
    int max_iteration_delta = 10;
};

struct WithStateArguments {
    std::string log_a;
    std::string state_a;
    std::string log_b;
    std::string state_b;
    bool no_state = false;
    double tolerance = 1.0e-8;
    double log_tolerance = 5.0e-3;
    int max_iteration_delta = 10;
};

struct NoStateArguments {
    std::string log_a;
    std::string log_b;
    bool no_state = false;
    double tolerance = 1.0e-8;
    double log_tolerance = 5.0e-3;
    int max_iteration_delta = 10;
};

int parse_with_state(WithStateArguments& input, int argc, char** argv) {
    CLAP_BEGIN(WithStateArguments)
    CLAP_ADD_USAGE(
        "LOG_A STATE_A LOG_B STATE_B [--tol X] [--log-tol X] "
        "[--max-iter-delta N]")
    CLAP_ADD_DESCRIPTION(
        "Compare residual logs, restart decisions, convergence, and state.")
    CLAP_REGISTER_ARG(log_a)
    CLAP_REGISTER_ARG(state_a)
    CLAP_REGISTER_ARG(log_b)
    CLAP_REGISTER_ARG(state_b)
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        no_state, "--no-state", "compare two logs without state files")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        tolerance, "--tol", "state relative tolerance (default 1e-8)")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        log_tolerance, "--log-tol", "printed residual tolerance (default 5e-3)")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        max_iteration_delta, "--max-iter-delta",
        "maximum convergence-iteration difference (default 10)")
    CLAP_END(WithStateArguments)
    try {
        CLAP<WithStateArguments>::parse_input(input, argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what();
        return EINVAL;
    }
    return 0;
}

int parse_without_state(NoStateArguments& input, int argc, char** argv) {
    CLAP_BEGIN(NoStateArguments)
    CLAP_ADD_USAGE("--no-state LOG_A LOG_B [--log-tol X] [--max-iter-delta N]")
    CLAP_ADD_DESCRIPTION(
        "Compare residual logs, restart decisions, and convergence.")
    CLAP_REGISTER_ARG(log_a)
    CLAP_REGISTER_ARG(log_b)
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        no_state, "--no-state", "compare two logs without state files")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        tolerance, "--tol", "unused state tolerance (accepted for parity)")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        log_tolerance, "--log-tol", "printed residual tolerance (default 5e-3)")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        max_iteration_delta, "--max-iter-delta",
        "maximum convergence-iteration difference (default 10)")
    CLAP_END(NoStateArguments)
    try {
        CLAP<NoStateArguments>::parse_input(input, argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what();
        return EINVAL;
    }
    return 0;
}

int parse_command_line(CommandLine& command, int argc, char** argv) {
    bool requested_no_state = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == "--no-state") {
            requested_no_state = true;
            break;
        }
    }
    if (requested_no_state) {
        NoStateArguments input;
        if (const int status = parse_without_state(input, argc, argv); status) {
            return status;
        }
        command.log_a = std::move(input.log_a);
        command.log_b = std::move(input.log_b);
        command.no_state = true;
        command.tolerance = input.tolerance;
        command.log_tolerance = input.log_tolerance;
        command.max_iteration_delta = input.max_iteration_delta;
    } else {
        WithStateArguments input;
        if (const int status = parse_with_state(input, argc, argv); status) {
            return status;
        }
        command.log_a = std::move(input.log_a);
        command.state_a = std::move(input.state_a);
        command.log_b = std::move(input.log_b);
        command.state_b = std::move(input.state_b);
        command.no_state = false;
        command.tolerance = input.tolerance;
        command.log_tolerance = input.log_tolerance;
        command.max_iteration_delta = input.max_iteration_delta;
    }
    if (!(command.tolerance > 0.0) || !std::isfinite(command.tolerance) ||
        !(command.log_tolerance > 0.0) ||
        !std::isfinite(command.log_tolerance) ||
        command.max_iteration_delta < 0) {
        std::cerr << "error: tolerances must be finite and positive and "
                     "--max-iter-delta must be nonnegative\n";
        return EINVAL;
    }
    return 0;
}

using Residual = std::array<double, 4>;

struct Restart {
    std::string kind;
    std::optional<int> iteration;
    double delt = 0.0;
};

struct LogData {
    std::map<int, Residual> rows;
    std::vector<Restart> restarts;
    std::optional<int> converged_iteration;
    std::string status;
    std::optional<int> iterations;
    std::map<std::string, double> summary;
};

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

std::string after_colon(const std::string& line) {
    const auto colon = line.find(':');
    return colon == std::string::npos ? std::string{}
                                      : trim(line.substr(colon + 1));
}

LogData parse_log(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("could not open " + path);

    const std::regex iteration(
        R"(^\s*(\d+)\s*\|\s*([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s*\|\s*([0-9.eE+\-]+))");
    const std::regex bad_jacobian(
        R"(BAD JACOBIAN \(iter2=(\d+)\) delt=([0-9.eE+\-]+))");
    const std::regex bad_progress(
        R"(BAD PROGRESS \(iter2=(\d+)\) delt=([0-9.eE+\-]+))");
    const std::regex reset(
        R"(CONVERGENCE PROBLEM: RESETTING DELT to ([0-9.eE+\-]+) \(ijacob=(\d+)\))");
    const std::regex converged(R"(CONVERGED at iter (\d+))");

    LogData result;
    std::string line;
    std::smatch match;
    while (std::getline(stream, line)) {
        if (std::regex_search(line, match, iteration)) {
            const int index = std::stoi(match[1].str());
            result.rows[index] = {
                std::stod(match[2].str()), std::stod(match[3].str()),
                std::stod(match[4].str()), std::stod(match[5].str())};
        } else if (std::regex_search(line, match, bad_jacobian)) {
            result.restarts.push_back(
                {"BADJ", std::stoi(match[1].str()), std::stod(match[2].str())});
        } else if (std::regex_search(line, match, bad_progress)) {
            result.restarts.push_back(
                {"BADP", std::stoi(match[1].str()), std::stod(match[2].str())});
        } else if (std::regex_search(line, match, reset)) {
            result.restarts.push_back(
                {"RESET", std::nullopt, std::stod(match[1].str())});
        } else if (std::regex_search(line, match, converged)) {
            result.converged_iteration = std::stoi(match[1].str());
        } else if (starts_with(line, "  Status:")) {
            result.status = after_colon(line);
        } else if (starts_with(line, "  Iterations:")) {
            result.iterations = std::stoi(after_colon(line));
        } else if (starts_with(line, "  FSQR:")) {
            result.summary["FSQR"] = std::stod(after_colon(line));
        } else if (starts_with(line, "  FSQZ:")) {
            result.summary["FSQZ"] = std::stod(after_colon(line));
        } else if (starts_with(line, "  FSQL:")) {
            result.summary["FSQL"] = std::stod(after_colon(line));
        }
    }
    return result;
}

struct RowComparison {
    bool pass = false;
    std::array<double, 4> worst{};
    std::array<std::optional<int>, 4> worst_at{};
    std::size_t common_count = 0;
    std::vector<int> only_a;
    std::vector<int> only_b;
};

RowComparison compare_rows(const std::map<int, Residual>& a,
                           const std::map<int, Residual>& b,
                           double tolerance) {
    RowComparison result;
    for (const auto& [iteration, row] : a) {
        const auto found = b.find(iteration);
        if (found == b.end()) {
            result.only_a.push_back(iteration);
            continue;
        }
        ++result.common_count;
        for (std::size_t column = 0; column < result.worst.size(); ++column) {
            const double difference = cumes::compare::relative_difference(
                row[column], found->second[column]);
            if (difference > result.worst[column]) {
                result.worst[column] = difference;
                result.worst_at[column] = iteration;
            }
        }
    }
    for (const auto& [iteration, unused] : b) {
        (void)unused;
        if (a.find(iteration) == a.end()) {
            result.only_b.push_back(iteration);
        }
    }
    result.pass =
        result.only_a.empty() && result.only_b.empty() &&
        std::all_of(result.worst.begin(), result.worst.end(),
                    [tolerance](double value) { return value < tolerance; });
    return result;
}

void print_iteration_list(const std::vector<int>& values) {
    std::cout << '[';
    const auto shown = std::min<std::size_t>(10, values.size());
    for (std::size_t index = 0; index < shown; ++index) {
        if (index) std::cout << ", ";
        std::cout << values[index];
    }
    if (values.size() > shown) std::cout << " ...";
    std::cout << ']';
}

bool same_restart_sequence(const std::vector<Restart>& a,
                           const std::vector<Restart>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t index = 0; index < a.size(); ++index) {
        if (a[index].kind != b[index].kind ||
            a[index].iteration != b[index].iteration) {
            return false;
        }
    }
    return true;
}

void print_restart_sequence(const std::vector<Restart>& events) {
    std::cout << '[';
    for (std::size_t index = 0; index < events.size(); ++index) {
        if (index) std::cout << ", ";
        std::cout << '(' << events[index].kind << ", ";
        if (events[index].iteration) {
            std::cout << *events[index].iteration;
        } else {
            std::cout << "None";
        }
        std::cout << ')';
    }
    std::cout << ']';
}

struct StateComparison {
    bool pass = false;
    std::string dimensions;
    std::array<double, cumes::compare::FAMILY_NAMES.size()> worst{};
};

StateComparison compare_states(const cumes::compare::State& a,
                               const cumes::compare::State& b,
                               double tolerance) {
    StateComparison result;
    if (a.ns != b.ns || a.mnmax != b.mnmax) {
        std::ostringstream message;
        message << "size mismatch: (" << a.ns << ", " << a.mnmax << ") vs ("
                << b.ns << ", " << b.mnmax << ')';
        result.dimensions = message.str();
        return result;
    }
    std::ostringstream message;
    message << "ns=" << a.ns << " mnmax=" << a.mnmax << " (interior j=1.."
            << a.ns - 1 << ')';
    result.dimensions = message.str();
    result.pass = true;
    for (std::size_t family = 0; family < result.worst.size(); ++family) {
        for (std::int32_t mode = 0; mode < a.mnmax; ++mode) {
            for (std::int32_t surface = 1; surface < a.ns; ++surface) {
                const auto index = static_cast<std::size_t>(mode) *
                                       static_cast<std::size_t>(a.ns) +
                                   static_cast<std::size_t>(surface);
                result.worst[family] = std::max(
                    result.worst[family],
                    cumes::compare::relative_difference(
                        a.families[family][index], b.families[family][index]));
            }
        }
        result.pass = result.pass && result.worst[family] < tolerance;
    }
    return result;
}

void append_failure(std::vector<std::string>& failures,
                    const std::string& failure) {
    failures.push_back(failure);
}

std::string optional_iteration(const std::optional<int>& value) {
    return value ? std::to_string(*value) : "None";
}

}  // namespace

int main(int argc, char** argv) {
    CommandLine command;
    if (const int status = parse_command_line(command, argc, argv); status) {
        return status;
    }

    try {
        const auto a = parse_log(command.log_a);
        const auto b = parse_log(command.log_b);
        std::vector<std::string> failures;

        const auto rows = compare_rows(a.rows, b.rows, command.log_tolerance);
        constexpr std::array<const char*, 4> columns = {"fsqr", "fsqz", "fsql",
                                                        "delt"};
        std::cout << "residual rows: " << rows.common_count
                  << " iterations compared (worst rel diff per column)\n";
        for (std::size_t column = 0; column < columns.size(); ++column) {
            std::cout << "  " << std::left << std::setw(4) << columns[column]
                      << std::right << ": " << std::scientific
                      << std::setprecision(3) << rows.worst[column]
                      << " @ iter "
                      << optional_iteration(rows.worst_at[column]);
            if (rows.worst[column] >= command.log_tolerance) {
                std::cout << "  <-- EXCEEDS TOL";
            }
            std::cout << '\n';
        }
        if (!rows.only_a.empty()) {
            std::cout << "  rows only in A: ";
            print_iteration_list(rows.only_a);
            std::cout << '\n';
        }
        if (!rows.only_b.empty()) {
            std::cout << "  rows only in B: ";
            print_iteration_list(rows.only_b);
            std::cout << '\n';
        }
        if (!rows.pass) append_failure(failures, "per-iteration residuals");

        std::cout << "restarts: A=" << a.restarts.size()
                  << " events, B=" << b.restarts.size() << " events\n";
        if (!same_restart_sequence(a.restarts, b.restarts)) {
            append_failure(failures, "restart sequence");
            std::cout << "  A: ";
            print_restart_sequence(a.restarts);
            std::cout << "\n  B: ";
            print_restart_sequence(b.restarts);
            std::cout << '\n';
        } else {
            std::cout << "  identical: ";
            print_restart_sequence(a.restarts);
            std::cout << '\n';
            for (std::size_t index = 0; index < a.restarts.size(); ++index) {
                const auto& left = a.restarts[index];
                const auto& right = b.restarts[index];
                if (cumes::compare::relative_difference(
                        left.delt, right.delt) >= command.log_tolerance) {
                    if (left.kind == "RESET") {
                        std::cout << "  RESET delt drift: " << std::scientific
                                  << std::setprecision(3) << left.delt << " vs "
                                  << right.delt << '\n';
                        append_failure(failures, "reset delt");
                    } else {
                        std::cout << "  " << left.kind << "@iter"
                                  << *left.iteration
                                  << " delt drift: " << std::scientific
                                  << std::setprecision(3) << left.delt << " vs "
                                  << right.delt << '\n';
                        append_failure(failures, "restart delt");
                    }
                }
            }
        }

        std::cout << "converged: A=";
        if (a.converged_iteration) {
            std::cout << "iter " << *a.converged_iteration;
        } else {
            std::cout << "NO";
        }
        std::cout << "  B=";
        if (b.converged_iteration) {
            std::cout << "iter " << *b.converged_iteration;
        } else {
            std::cout << "NO";
        }
        std::cout << "  (status " << (a.status.empty() ? "None" : a.status)
                  << " / " << (b.status.empty() ? "None" : b.status) << ")\n";
        if (!a.converged_iteration || !b.converged_iteration) {
            append_failure(failures, "one or both runs did not converge");
        } else {
            const int difference =
                std::abs(*a.converged_iteration - *b.converged_iteration);
            std::cout << "  converged-iter delta: " << difference << " (window "
                      << command.max_iteration_delta << ")\n";
            if (difference > command.max_iteration_delta) {
                append_failure(failures, "converged iteration count");
            }
        }
        for (const char* key : {"FSQR", "FSQZ", "FSQL"}) {
            const auto left = a.summary.find(key);
            const auto right = b.summary.find(key);
            if (left == a.summary.end() || right == b.summary.end()) continue;
            const double difference = cumes::compare::relative_difference(
                left->second, right->second);
            std::cout << "  final " << key << ": A=" << std::scientific
                      << std::setprecision(3) << left->second
                      << " B=" << right->second
                      << " rel=" << std::setprecision(2) << difference << '\n';
            if (difference >= command.log_tolerance) {
                append_failure(failures, std::string("final ") + key);
            }
        }

        if (!command.no_state) {
            const auto left = cumes::compare::read_state(command.state_a, true);
            const auto right =
                cumes::compare::read_state(command.state_b, true);
            const auto state = compare_states(left, right, command.tolerance);
            std::cout << "state: " << state.dimensions << '\n';
            if (left.ns == right.ns && left.mnmax == right.mnmax) {
                for (std::size_t family = 0; family < state.worst.size();
                     ++family) {
                    std::cout
                        << "  " << std::left << std::setw(6)
                        << cumes::compare::FAMILY_NAMES[family] << std::right
                        << ": max rel diff (interior) = " << std::scientific
                        << std::setprecision(3) << state.worst[family];
                    if (state.worst[family] >= command.tolerance) {
                        std::cout << "  <-- EXCEEDS TOL";
                    }
                    std::cout << '\n';
                }
            }
            if (!state.pass) append_failure(failures, "state");
        }

        if (!failures.empty()) {
            std::cout << "FAIL: ";
            for (std::size_t index = 0; index < failures.size(); ++index) {
                if (index) std::cout << ", ";
                std::cout << failures[index];
            }
            std::cout << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
