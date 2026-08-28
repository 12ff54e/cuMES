#include "cumes/io/magnetic_coordinate_bridge.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_zero_copy_mapping() {
    cumes::InputParams input;
    input.mpol = 2;
    input.ntor = 1;
    input.nfp = 5;
    input.ntheta = 4;
    input.nzeta = 2;
    input.ncurr = 1;
    input.phiedge = -2.0;
    input.aphi = {1.0, 0.25};

    cumes::EquilibriumSnapshot snapshot;
    snapshot.ns = 3;
    snapshot.mnmax = 4;
    snapshot.ntheta = input.ntheta;
    snapshot.nzeta = input.nzeta;
    for (auto& family : snapshot.families) {
        family.assign(snapshot.family_size(), 1.0);
    }
    for (auto& field : snapshot.half_fields) {
        field.assign(snapshot.half_field_size(), 2.0);
    }
    for (auto& field : snapshot.full_fields) {
        field.assign(snapshot.full_field_size(), 3.0);
    }

    const auto view = cumes::make_magnetic_coordinate_view(snapshot, input);
    expect(view.format_version == 8 && view.ns == snapshot.ns &&
               view.mpol == input.mpol && view.nfp == input.nfp,
           "bridge metadata mismatch");
    expect(view.aphi.data() == input.aphi.data(), "bridge copied aphi");
    expect(view.families[magnetic_coordinate::CumesEquilibrium::RMNCC].data() ==
               snapshot.families[cumes::EquilibriumSnapshot::RMNCC].data(),
           "bridge copied spectral state");
    expect(
        view.half_fields[magnetic_coordinate::CumesEquilibrium::BSUBV].data() ==
            snapshot.half_fields[cumes::EquilibriumSnapshot::BSUBV].data(),
        "bridge copied magnetic fields");
}

void test_metadata_rejection() {
    cumes::InputParams input;
    input.mpol = 2;
    input.ntor = 1;
    input.ntheta = 4;
    input.nzeta = 2;
    cumes::EquilibriumSnapshot snapshot;
    snapshot.ns = 3;
    snapshot.mnmax = 5;
    snapshot.ntheta = input.ntheta;
    snapshot.nzeta = input.nzeta;

    try {
        static_cast<void>(
            cumes::make_magnetic_coordinate_view(snapshot, input));
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("bridge accepted inconsistent mode metadata");
}

}  // namespace

int main() {
    try {
        test_zero_copy_mapping();
        test_metadata_rejection();
    } catch (const std::exception& error) {
        std::cerr << "test_magnetic_coordinate_bridge: " << error.what()
                  << '\n';
        return 1;
    }
    std::cout << "test_magnetic_coordinate_bridge: PASS\n";
    return 0;
}
