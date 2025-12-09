# Sets up GoogleTest (downloads, configures, exposes GTest::GTest and GTest::Main)
include(cmake/VendorGTest.cmake)

# Cnpy integration
include(cmake/VendorCnpy.cmake)

# ImGui integration
include(cmake/VendorImgui.cmake)

# Implot integration
include(cmake/VendorImplot.cmake)

# Eigen parallelization settings
include(cmake/VendorEigenParallel.cmake)

# Vendored matio handling (sets MATIO_ROOT_DIR, MATIO_INCLUDE_DIR, MATIO_LIBRARY, exposes MATIO::MATIO)
include(cmake/VendorMatio.cmake)

# Vendored FFTW handling
include(cmake/VendorFFTW.cmake)

# Vendored NFFT3 handling
include(cmake/VendorNFFT3.cmake)

# Workarounds / shims for matio-cpp configure/export behavior
include(cmake/VendorMatioCppShim.cmake)