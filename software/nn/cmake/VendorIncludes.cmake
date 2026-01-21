#+#+#+#+-----------------------------------------------------------------------
# VendorIncludes.cmake
#
# Aggregates vendored/FetchContent/ExternalProject dependencies in one place so
# the rest of the project can `include()` a single module.
#
# This file should remain declarative: it delegates to `Vendor*.cmake` modules
# that each own the details for a specific dependency.
#+#+#+#+-----------------------------------------------------------------------

# Cnpy integration
include(cmake/VendorCnpy.cmake)

# Argparse integration
include(cmake/VendorArgparse.cmake)

# Vendored FFTW handling
include(cmake/VendorFFTW.cmake)

# Vendored NFFT3 handling
include(cmake/VendorNFFT3.cmake)

# ImGui integration
include(cmake/VendorImgui.cmake)

# Sets up GoogleTest (downloads, 
# configures, exposes GTest::GTest 
# and GTest::Main)
include(cmake/VendorGTest.cmake)

# Vendored matio handling (sets 
# MATIO_ROOT_DIR, MATIO_INCLUDE_DIR, 
# MATIO_LIBRARY, exposes MATIO::MATIO)
include(cmake/VendorMatio.cmake)

# Implot integration
include(cmake/VendorImplot.cmake)

# Workarounds / shims for matio-cpp 
# configure/export behavior
include(cmake/VendorMatioCppShim.cmake)

# Matplotlib-cpp integration
include(cmake/VendorMatplotlibCpp.cmake)

# YAML-cpp integration
include(cmake/VendorYaml.cmake)

# Eigen parallelization settings
include(cmake/VendorEigenParallel.cmake)