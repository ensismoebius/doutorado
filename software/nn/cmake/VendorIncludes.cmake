# Cnpy integration
include(cmake/Cnpy.cmake)

# ImGui integration
include(cmake/Imgui.cmake)

# Implot integration
include(cmake/Implot.cmake)

# Eigen parallelization settings
include(cmake/EigenParallel.cmake)

# Vendored matio handling (sets MATIO_ROOT_DIR, MATIO_INCLUDE_DIR, MATIO_LIBRARY, exposes MATIO::MATIO)
include(cmake/VendorMatio.cmake)

# Vendored FFTW handling
include(cmake/VendorFFTW.cmake)

# Vendored NFFT3 handling
include(cmake/VendorNFFT3.cmake)

# Workarounds / shims for matio-cpp configure/export behavior
include(cmake/VendorMatioCppShim.cmake)