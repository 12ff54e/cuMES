# Browser coil presets

Only coil geometry and equilibrium/grid configuration are shipped. The existing
vacuum-field MAKEGRID generator creates response tables in WebAssembly memory.
No precomputed field grids or NetCDF runtime are required.

Solovev and cth_like coils are copied at build time from vacuum-field/tests/data.
Solovev uses inputs/free_bdy/solovev_free_bdy_coils.json. The cth_like configuration
comes from vmecpp/src/vmecpp/cpp/vmecpp/test_data/cth_like_free_bdy.json.
W7-X geometry comes from vmecpp/src/vmecpp/cpp/vmecpp_large_cpp_tests/test_data/coils.w7x;
its currents and initial boundary are from examples/data/w7x_free_bdy_vac.json.
The W7-X preset is the vacuum case (zero pressure and plasma current).

Source: https://github.com/proximafusion/vmecpp at
335ef66441d82980331d0062ab8d0398eff50818; MIT license in LICENSE.vmecpp.
The browser uses paired-f32 or scalar-f32 equilibrium tolerances and a larger
cth_like iteration budget. Users can edit currents, grid bounds/resolution,
and equilibrium input before running.
