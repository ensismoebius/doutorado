##
## VendorNFFT3.cmake
##
## Purpose
## - Provide NFFT3 (Non-uniform FFT library) as a vendored dependency.
## - If not found under `build/_deps/nfft3-install`, build it via autotools using
##   `ExternalProject_Add`.
##
## What it provides
## - Imported target: `NFFT::NFFT`.
##
## Dependencies / ordering
## - Links against `FFTW::FFTW` and `OpenMP::OpenMP_C`.
## - This module expects FFTW to be configured first (via `VendorFFTW.cmake`) so it can
##   pass `--with-fftw3=<prefix>` to NFFT3's configure.
##
## Notes
## - This is an autotools build embedded in a CMake configure. Keep changes minimal
##   and prefer pinning versions + hashes for reproducibility.
##

# VendorNFFT3.cmake
# Configure vendored lib/nfft3 presence and make NFFT::NFFT available

include(FetchContent)
include(ExternalProject)

# Find OpenMP for NFFT3 (consistency with VendorFFTW.cmake)
find_package(OpenMP REQUIRED)

set(FFTW_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/fftw-install")

# Set install directory and check if NFFT3 is already built
set(NFFT3_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/nfft3-install")

# Define configure flags based on build type for autotools
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(NFFT3_ENV_FLAGS "CFLAGS=-g -O0" "CXXFLAGS=-g -O0")
else() # Release or other types, default to Release
  set(NFFT3_ENV_FLAGS "CFLAGS=-O3 -DNDEBUG" "CXXFLAGS=-O3 -DNDEBUG")
endif()

find_library(NFFT3_LIBRARY nfft3 HINTS "${NFFT3_INSTALL_DIR}/lib")
find_path(NFFT3_INCLUDE_DIR nfft3.h HINTS "${NFFT3_INSTALL_DIR}/include")

# Determine the correct FFTW path for NFFT3's configure script
if(FFTW_LIBRARY)
    get_filename_component(NFFT3_FFTW_CONFIG_PATH "${FFTW_LIBRARY}" DIRECTORY)
    get_filename_component(NFFT3_FFTW_CONFIG_PATH "${NFFT3_FFTW_CONFIG_PATH}" DIRECTORY) # Get parent directory (e.g., /usr from /usr/lib)
else()
    set(NFFT3_FFTW_CONFIG_PATH "${CMAKE_BINARY_DIR}/_deps/fftw-install")
endif()

# If NFFT3 is not found, build it from source
if(NOT NFFT3_LIBRARY OR NOT NFFT3_INCLUDE_DIR)
    message(STATUS "NFFT3 library not found, building from source.")

    # Check if fftw_build target exists to add dependency
    set(NFFT3_DEPENDS "")
    if(TARGET fftw_build)
        list(APPEND NFFT3_DEPENDS fftw_build)
    endif()

    message(STATUS "Starting ExternalProject_Add for nfft3...")
    ExternalProject_Add(nfft3
        URL            https://github.com/NFFT/nfft/releases/download/3.5.3/nfft-3.5.3.tar.gz
        URL_HASH       SHA256=caf1b3b3e5bf8c33a6bfd7eca811d954efce896605ecfd0144d47d0bebdf4371
        DOWNLOAD_DIR   "${CMAKE_BINARY_DIR}/_deps"
        INSTALL_DIR    "${NFFT3_INSTALL_DIR}"
        UPDATE_DISCONNECTED 1
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE

        BUILD_BYPRODUCTS "${NFFT3_INSTALL_DIR}/lib/libnfft3.so"
        DEPENDS ${NFFT3_DEPENDS}

        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${NFFT3_ENV_FLAGS}
            <SOURCE_DIR>/configure
            --prefix=<INSTALL_DIR>
            --disable-examples
            --disable-applications
            --enable-openmp
            --enable-shared
            --with-fftw3=${NFFT3_FFTW_CONFIG_PATH}

        BUILD_COMMAND make -j4
        INSTALL_COMMAND make install
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1
    )
    message(STATUS "Finished ExternalProject_Add for nfft3.")

    add_library(NFFT::NFFT SHARED IMPORTED GLOBAL)
    set_target_properties(NFFT::NFFT PROPERTIES
        IMPORTED_LOCATION             "${NFFT3_INSTALL_DIR}/lib/libnfft3.so"
        INTERFACE_INCLUDE_DIRECTORIES "${NFFT3_INSTALL_DIR}/include"
        INTERFACE_LINK_LIBRARIES      "FFTW::FFTW;OpenMP::OpenMP_C"
    )
    set_property(TARGET NFFT::NFFT PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${NFFT3_INSTALL_DIR}/include")
    add_dependencies(NFFT::NFFT nfft3)

else()
    message(STATUS "Found pre-built NFFT3 at: ${NFFT3_LIBRARY}")
    add_library(NFFT::NFFT SHARED IMPORTED GLOBAL)
    set_target_properties(NFFT::NFFT PROPERTIES
        IMPORTED_LOCATION             "${NFFT3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NFFT3_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES      "FFTW::FFTW;OpenMP::OpenMP_C"
    )
    set_property(TARGET NFFT::NFFT PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${NFFT3_INCLUDE_DIR}")
    
    if(TARGET fftw_build)
        add_dependencies(NFFT::NFFT fftw_build)
    endif()
endif()
