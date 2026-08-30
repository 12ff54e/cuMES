// Compare two cuMES converged-state binary payloads.

#include "../include/clap.h"
#include "include/compare_common.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CommandLine {
    std::string state_a;
    std::string state_b;
};

int parse_command_line(CommandLine& command, int argc, char** argv) {
    CLAP_BEGIN(CommandLine)
    CLAP_ADD_USAGE("STATE_A STATE_B")
    CLAP_ADD_DESCRIPTION(
        "Compare two cuMES state payloads, skipping the extrapolated axis row.")
    CLAP_REGISTER_ARG(state_a)
    CLAP_REGISTER_ARG(state_b)
    CLAP_END(CommandLine)
    try {
        CLAP<CommandLine>::parse_input(command, argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what();
        return EINVAL;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CommandLine command;
    if (const int status = parse_command_line(command, argc, argv); status) {
        return status;
    }

    try {
        const auto a = cumes::compare::read_state(command.state_a, false);
        const auto b = cumes::compare::read_state(command.state_b, false);
        if (a.ns != b.ns || a.mnmax != b.mnmax) {
            std::cerr << "error: size mismatch: (" << a.ns << ", " << a.mnmax
                      << ") vs (" << b.ns << ", " << b.mnmax << ")\n";
            return 2;
        }

        std::cout << "ns=" << a.ns << " mnmax=" << a.mnmax
                  << " (comparing interior j=1.." << a.ns - 1 << ")\n";
        double worst = 0.0;
        for (std::size_t family = 0;
             family < cumes::compare::FAMILY_NAMES.size(); ++family) {
            double family_worst = 0.0;
            for (std::int32_t mode = 0; mode < a.mnmax; ++mode) {
                for (std::int32_t surface = 1; surface < a.ns; ++surface) {
                    const auto index = static_cast<std::size_t>(mode) *
                                           static_cast<std::size_t>(a.ns) +
                                       static_cast<std::size_t>(surface);
                    family_worst = std::max(family_worst,
                                            cumes::compare::relative_difference(
                                                a.families[family][index],
                                                b.families[family][index]));
                }
            }
            worst = std::max(worst, family_worst);
            std::cout << "  " << std::left << std::setw(6)
                      << cumes::compare::FAMILY_NAMES[family] << std::right
                      << ": max rel diff (interior) = " << std::scientific
                      << std::setprecision(3) << family_worst << '\n';
        }
        std::cout << "worst over all families: " << std::scientific
                  << std::setprecision(3) << worst << '\n';
        constexpr double tolerance = 1.0e-8;
        const bool pass = worst < tolerance;
        std::cout << (pass ? "PASS\n" : "FAIL\n");
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
