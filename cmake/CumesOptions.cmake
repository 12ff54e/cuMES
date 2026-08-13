# CumesOptions.cmake — cuMES user-facing build options.
#
# Every option is a cache variable so CMakePresets.json / -D<VAR> on the
# command line can set them. They are plain toggles; the value-added policy
# (precision floors, fast-math attribution, tolerance validation) is a Phase 2
# `PrecisionPolicy` concern and is not wired here yet.

# Precision: all computation is templated on T; the CLI's `Real` alias
# (vmec_types.h) is the compile-time switch. OFF = double (verified default),
# ON = float (experimental; needs relaxed ftol, see main.cu).
option(CUMES_USE_FLOAT "Build the cuMES CLI in single precision (float)" OFF)

# Optional state-output backends. Each is compiled in only if the option is ON
# and the library is found (see CumesDependencies.cmake); otherwise the run
# falls back to binary cumes_state.bin.
option(CUMES_USE_NETCDF "Enable NetCDF output of the solved state (.nc)" ON)
option(CUMES_USE_HDF5  "Enable HDF5  output of the solved state (.h5/.hdf5)" ON)

# Optional Compute Sanitizer pass over the kernel-driving unit tests (memcheck).
# Requires compute-sanitizer on PATH.
option(CUMES_ENABLE_SANITIZER_TESTS "Register compute-sanitizer memcheck test variants" OFF)

# Promote warnings to errors for project sources (excluding vendored deps).
option(CUMES_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
