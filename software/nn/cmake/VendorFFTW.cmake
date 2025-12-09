# VendorFFTW.cmake
# Configure vendored lib/fftw presence and make FFTW::FFTW available

include(FetchContent)

# Find OpenMP before FFTW so FFTW can use it
find_package(OpenMP REQUIRED)

FetchContent_Declare(
    fftw
    URL            https://fftw.org/fftw-3.3.10.tar.gz
    URL_HASH       SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Enable OpenMP in FFTW
set(ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP in FFTW" FORCE)
set(ENABLE_THREADS ON CACHE BOOL "Enable THREADS in FFTW" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Disable FFTW tests" FORCE)

FetchContent_MakeAvailable(fftw)

# target_include_directories(FFTW3::fftw3 SYSTEM
#   PUBLIC
#     "${fftw_SOURCE_DIR}/api"
# )

# # Provide a lightweight imported "fftw" target if the subproject doesn't
# # create it early enough. This helps export/install steps in other vendored
# # CMakeLists succeed.
# if(TARGET FFTW3::fftw3)
#   add_library(FFTW::FFTW ALIAS FFTW3::fftw3)
# endif()
