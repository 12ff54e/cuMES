# CumesOptions.cmake — cuMES user-facing build options.
#
# Every option is a cache variable so CMakePresets.json / -D<VAR> on the
# command line can set them. They are plain toggles; the value-added policy
# (precision floors, fast-math attribution, tolerance validation) is the
# named CUMES_PRECISION_POLICY below (completion plan step 3.1).

# Named precision policies (completion plan step 3.1). The policy maps to
# target-scoped flags in CMakeLists.txt and is recorded in the build
# provenance of every v1 output:
#   verify-double  precise double math (the verification configuration);
#   fast-double    selected, attributable fast intrinsics (--use_fast_math);
#   mixed-float    float state/FFT with the documented double reductions
#                  (implies CUMES_USE_FLOAT=ON);
#   debug-double   precise double math plus device debug checks (-G).
set(CUMES_PRECISION_POLICY "verify-double" CACHE STRING
    "Named precision policy: verify-double|fast-double|mixed-float|debug-double")
set_property(CACHE CUMES_PRECISION_POLICY PROPERTY STRINGS
             verify-double fast-double mixed-float debug-double)

# Precision: all computation is templated on T; the CLI's `Real` alias
# (vmec_types.h) is the compile-time switch. OFF = double (verified default),
# ON = float (experimental; needs relaxed ftol, see main.cu). Set directly,
# or implied by CUMES_PRECISION_POLICY=mixed-float.
option(CUMES_USE_FLOAT "Build the cuMES CLI in single precision (float)" OFF)

# Optional state-output backends. Each is compiled in only if the option is ON
# and the library is found (see CumesDependencies.cmake); a known suffix whose
# backend is not linked is rejected by the preflight at startup.
option(CUMES_USE_NETCDF "Enable NetCDF output of the solved state (.nc)" ON)
option(CUMES_USE_HDF5  "Enable HDF5  output of the solved state (.h5/.hdf5)" ON)

# Optional Compute Sanitizer pass over the kernel-driving unit tests (memcheck).
# Requires compute-sanitizer on PATH.
option(CUMES_ENABLE_SANITIZER_TESTS "Register compute-sanitizer memcheck test variants" OFF)

# Additional Compute Sanitizer tools (racecheck + synccheck) beyond memcheck
# (blueprint §10.5). Racecheck instruments every memory access and is much
# slower than memcheck, so these are NOT part of the default verify gate —
# enable them in the dedicated `sanitizer` preset.
option(CUMES_ENABLE_EXTRA_SANITIZER_TOOLS "Register racecheck/synccheck compute-sanitizer variants" OFF)

# Host AddressSanitizer + UBSan for the host-only targets (the config/core/IO
# libraries and their tests — blueprint §10.5's "host ASan and UBSan for
# config, controller, and I/O"). OFF by default; enable in the `sanitizer`
# preset. CUDA (.cu) targets are untouched: nvcc and ASan do not mix.
option(CUMES_HOST_SANITIZERS "Build host-only targets with ASan+UBSan" OFF)

# Compile the CUMES_DUMP-gated verification dump machinery into the solver
# (completion plan step 3.3). ON for the verify/sanitizer/float presets (the
# trajectory-verification recipe needs it); production-style builds (e.g. the
# fast preset) turn it OFF to compile the machinery out entirely.
option(CUMES_ENABLE_VERIFY_DUMP "Compile the CUMES_DUMP-gated dump machinery" ON)

# Promote warnings to errors for project sources (excluding vendored deps).
option(CUMES_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
