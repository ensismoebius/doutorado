##
## VendorFFTW.cmake
##
## Purpose
## - Provide FFTW3 (double or float) either from a prebuilt install under build/_deps
##   or by building from source via `ExternalProject_Add`.
##
## What it provides
## - Imported target: `FFTW::FFTW`.
## - Optional build target: `fftw_build` when FFTW is found prebuilt (no-op target).
##
## Important details
## - OpenMP is located before configuring FFTW so FFTW can be built with `--enable-openmp`.
## - The import location uses a generator expression to switch between `libfftw3.so`
##   and `libfftw3f.so` based on `USE_FFTWF`.
## - Headers are exposed as SYSTEM includes to reduce warnings from vendor code.
##

# VendorFFTW.cmake
# Configure vendored lib/fftw presence and make FFTW::FFTW available

option(USE_FFTWF "Use single-precision FFTW (fftw3f) instead of double-precision (fftw3)" OFF)

include(ExternalProject)

# Find OpenMP before FFTW so FFTW can use it
find_package(OpenMP REQUIRED)

set(FFTW_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/fftw-install")
set(FFTW_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/fftw-src")

# Library file name is platform-dependent (.so on Linux, .dylib on macOS)
set(FFTW_LIBRARY_FILE_NAME "fftw3f")
if(NOT USE_FFTWF)
    set(FFTW_LIBRARY_FILE_NAME "fftw3")
endif()
set(FFTW_IMPORTED_LIBRARY_PATH "${FFTW_INSTALL_DIR}/lib/lib${FFTW_LIBRARY_FILE_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Try to find pre-built FFTW
find_library(FFTW_LIBRARY fftw3 HINTS "${FFTW_INSTALL_DIR}/lib" "${FFTW_INSTALL_DIR}/lib64")
find_path(FFTW_INCLUDE_DIR fftw3.h HINTS "${FFTW_INSTALL_DIR}/include")

# Create an imported target for FFTW (will be defined differently based on whether it's built or found)
add_library(FFTW::FFTW SHARED IMPORTED GLOBAL)

if(NOT FFTW_LIBRARY OR NOT FFTW_INCLUDE_DIR)
    message(STATUS "FFTW library not found, building from source.")

    # Create directories (only if building)
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
            "${FFTW_IMPORTED_LIBRARY_PATH}"

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

    set_target_properties(FFTW::FFTW PROPERTIES
        IMPORTED_LOCATION "${FFTW_IMPORTED_LIBRARY_PATH}"
        IMPORTED_NO_SONAME TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INSTALL_DIR}/include"
    )
    set_property(TARGET FFTW::FFTW PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${FFTW_INSTALL_DIR}/include")

    add_dependencies(FFTW::FFTW fftw_build)
else()
    message(STATUS "Found pre-built FFTW at: ${FFTW_LIBRARY}")
    # Determine the correct library to link based on USE_FFTWF
    if(USE_FFTWF)
        set(FFTW_SPECIFIC_LIBRARY_NAME "fftw3f")
    else()
        set(FFTW_SPECIFIC_LIBRARY_NAME "fftw3")
    endif()

    find_library(FFTW_LIBRARY_ACTUAL "${FFTW_SPECIFIC_LIBRARY_NAME}" HINTS "${FFTW_INSTALL_DIR}/lib" "${FFTW_INSTALL_DIR}/lib64")
    if(NOT FFTW_LIBRARY_ACTUAL)
        message(FATAL_ERROR "Could not find FFTW library for ${FFTW_SPECIFIC_LIBRARY_NAME}")
    endif()

    set_target_properties(FFTW::FFTW PROPERTIES
        IMPORTED_LOCATION "${FFTW_LIBRARY_ACTUAL}"
        IMPORTED_NO_SONAME TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}"
    )
    set_property(TARGET FFTW::FFTW PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}")
    add_custom_target(fftw_build)
endif()

# Single-precision FFTW (fftw3f): consumed by the wave feature pipeline as a
# separate library. A bare `fftw3f` link name only resolves on Linux via
# /usr/lib; resolve the full path for portability (macOS: /opt/homebrew/lib).
find_library(FFTW_F_LIBRARY fftw3f HINTS "${FFTW_INSTALL_DIR}/lib" "${FFTW_INSTALL_DIR}/lib64")
if(FFTW_F_LIBRARY)
    add_library(FFTW::FFTWF SHARED IMPORTED GLOBAL)
    set_target_properties(FFTW::FFTWF PROPERTIES
        IMPORTED_LOCATION "${FFTW_F_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}")
    set_property(TARGET FFTW::FFTWF PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}")
    if(TARGET fftw_build)
        add_dependencies(FFTW::FFTWF fftw_build)
    endif()
else()
    message(WARNING
        "libfftw3f not found; targets linking FFTW::FFTWF (e.g. waveCoreLib) will fail to link")
endif()
