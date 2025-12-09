# VendorFFTW.cmake
# Configure vendored lib/fftw presence and make FFTW::FFTW available

include(ExternalProject)

# Find OpenMP before FFTW so FFTW can use it
find_package(OpenMP REQUIRED)

set(FFTW_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/fftw-install")
set(FFTW_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/fftw-src")

# Create directories
file(MAKE_DIRECTORY "${FFTW_INSTALL_DIR}")
file(MAKE_DIRECTORY "${FFTW_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${FFTW_INSTALL_DIR}/lib")

# Use ExternalProject to compile and install FFTW with configure
ExternalProject_Add(fftw_build
    URL            https://fftw.org/fftw-3.3.10.tar.gz
    URL_HASH       SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    DOWNLOAD_DIR   "${CMAKE_BINARY_DIR}/_deps"
    INSTALL_DIR    "${FFTW_INSTALL_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    BUILD_IN_SOURCE TRUE
    # Make Ninja/CMake aware of which files this ExternalProject will produce
    BUILD_BYPRODUCTS
        "${FFTW_INSTALL_DIR}/lib/libfftw3.so"
        "${FFTW_INSTALL_DIR}/lib/libfftw3f.so"

    CONFIGURE_COMMAND "<SOURCE_DIR>/configure"
        --prefix=<INSTALL_DIR>
        --enable-shared
        --disable-static
        --enable-openmp
        --disable-fortran
    BUILD_COMMAND "make" "-j4"
    INSTALL_COMMAND "make" "install"
    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
)

# Create an imported target for FFTW
add_library(FFTW::FFTW SHARED IMPORTED GLOBAL)
set_target_properties(FFTW::FFTW PROPERTIES
    IMPORTED_LOCATION "${FFTW_INSTALL_DIR}/lib/libfftw3.so"
    IMPORTED_NO_SONAME TRUE
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INSTALL_DIR}/include"
)
add_dependencies(FFTW::FFTW fftw_build)
