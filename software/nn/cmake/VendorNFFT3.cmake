# VendorNFFT3.cmake
# Configure vendored lib/nfft3 presence and make NFFT::NFFT available

include(FetchContent)
include(ExternalProject)

# Find OpenMP for NFFT3 (consistency with VendorFFTW.cmake)
find_package(OpenMP REQUIRED)

set(FFTW_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/fftw-install")

ExternalProject_Add(nfft3
    URL            https://github.com/NFFT/nfft/releases/download/3.5.3/nfft-3.5.3.tar.gz
    URL_HASH       SHA256=caf1b3b3e5bf8c33a6bfd7eca811d954efce896605ecfd0144d47d0bebdf4371
    DOWNLOAD_DIR   "${CMAKE_BINARY_DIR}/_deps/nfft3-src"
    SOURCE_DIR     "${CMAKE_BINARY_DIR}/_deps/nfft3-src"
    INSTALL_DIR    "${CMAKE_BINARY_DIR}/_deps/nfft3-install"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE

    # Ensure FFTW is built and installed before configuring NFFT3
    DEPENDS fftw_build

    PATCH_COMMAND
        "bash" "-c" "cd <SOURCE_DIR> && ./bootstrap.sh && autoreconf --install --force"
    
    CONFIGURE_COMMAND
        "bash" "-c" "cd <SOURCE_DIR> && ./configure --prefix=<INSTALL_DIR> --disable-examples --disable-applications --enable-openmp --enable-shared --with-fftw3=${FFTW_INSTALL_DIR}"

    BUILD_COMMAND
        "bash" "-c" "cd <SOURCE_DIR> && make -j4"

    INSTALL_COMMAND
        "bash" "-c" "cd <SOURCE_DIR> && make install"
)

add_library(NFFT::NFFT SHARED IMPORTED GLOBAL)

set_target_properties(NFFT::NFFT
    PROPERTIES
        IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/_deps/nfft3-install/lib/libnfft3.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/_deps/nfft3-install/include"
        INTERFACE_LINK_LIBRARIES "OpenMP::OpenMP_C"
)

add_dependencies(NFFT::NFFT nfft3 fftw_build)
