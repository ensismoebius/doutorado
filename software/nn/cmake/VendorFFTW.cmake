# VendorFFTW.cmake
# Configure vendored lib/fftw presence and make FFTW::FFTW available

include(FetchContent)
include(ExternalProject)

# Find OpenMP before FFTW so FFTW can use it
find_package(OpenMP REQUIRED)

# Use ExternalProject to compile and install FFTW with configure
ExternalProject_Add(fftw_build
    URL            https://fftw.org/fftw-3.3.10.tar.gz
    URL_HASH       SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    DOWNLOAD_DIR   "${CMAKE_BINARY_DIR}/_deps/fftw-src"
    SOURCE_DIR     "${CMAKE_BINARY_DIR}/_deps/fftw-src"
    INSTALL_DIR    "${CMAKE_BINARY_DIR}/_deps/fftw-install"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE

    CONFIGURE_COMMAND "${CMAKE_BINARY_DIR}/_deps/fftw-src/configure" 
        --prefix=<INSTALL_DIR>
        --enable-shared
        --disable-static
        --enable-openmp
        --disable-fortran
    BUILD_COMMAND "make" "-j4"
    INSTALL_COMMAND "make" "install"
)

# Create an imported target for FFTW
add_library(FFTW::FFTW SHARED IMPORTED)
set_target_properties(FFTW::FFTW PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/_deps/fftw-install/lib/libfftw3.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/_deps/fftw-install/include"
)
add_dependencies(FFTW::FFTW fftw_build)
