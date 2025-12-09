# VendorNFFT3.cmake
# Configure vendored lib/nfft3 presence and make NFFT::NFFT available

include(FetchContent)
include(ExternalProject)

# Find OpenMP for NFFT3 (consistency with VendorFFTW.cmake)
find_package(OpenMP REQUIRED)

set(FFTW_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/fftw-install")
set(NFFT3_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/nfft3-install")
file(MAKE_DIRECTORY "${NFFT3_INSTALL_DIR}")


ExternalProject_Add(nfft3
    URL            https://github.com/NFFT/nfft/releases/download/3.5.3/nfft-3.5.3.tar.gz
    URL_HASH       SHA256=caf1b3b3e5bf8c33a6bfd7eca811d954efce896605ecfd0144d47d0bebdf4371
    DOWNLOAD_DIR   "${CMAKE_BINARY_DIR}/_deps"
    INSTALL_DIR    "${NFFT3_INSTALL_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    BUILD_IN_SOURCE TRUE

    # Make Ninja/CMake aware of which files this ExternalProject will produce
    BUILD_BYPRODUCTS "${NFFT3_INSTALL_DIR}/lib/libnfft3.so"

    # Ensure FFTW is built and installed before configuring NFFT3
    DEPENDS fftw_build
    
    CONFIGURE_COMMAND
        <SOURCE>/configure --prefix=${NFFT3_INSTALL_DIR} --disable-examples --disable-applications --enable-openmp --enable-shared --with-fftw3=${FFTW_INSTALL_DIR}

    BUILD_COMMAND
        make -j4

    INSTALL_COMMAND
        make install

    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
)

add_library(NFFT::NFFT SHARED IMPORTED GLOBAL)

set_target_properties(NFFT::NFFT
    PROPERTIES
        IMPORTED_LOCATION "${NFFT3_INSTALL_DIR}/lib/libnfft3.so"
        INTERFACE_INCLUDE_DIRECTORIES "${NFFT3_INSTALL_DIR}/include"
        INTERFACE_LINK_LIBRARIES "OpenMP::OpenMP_C"
)

add_dependencies(NFFT::NFFT nfft3 fftw_build)