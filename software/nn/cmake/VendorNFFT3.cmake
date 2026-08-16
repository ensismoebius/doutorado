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

# Library file name is platform-dependent (.so on Linux, .dylib on macOS)
set(NFFT3_LIBRARY_FILE "libnfft3${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Define configure flags based on build type for autotools
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(NFFT3_ENV_FLAGS "CFLAGS=-g -O0" "CXXFLAGS=-g -O0")
else() # Release or other types, default to Release
  set(NFFT3_ENV_FLAGS "CFLAGS=-O3 -DNDEBUG" "CXXFLAGS=-O3 -DNDEBUG")
endif()

# NFFT3's bundled AX_OPENMP only probes GCC/ICC-style flags (-fopenmp, ...) and
# its libtool rules propagate a raw `-fopenmp` into the final dylib link.
# AppleClang has no OpenMP support at all, so on macOS we drive the configure
# and build through a small CC/CXX wrapper that translates `-fopenmp` into the
# Homebrew libomp recipe (`-Xpreprocessor -fopenmp` + `-lomp`).
#
# Homebrew GCC is NOT an alternative: gcc-15/gcc-16 hit an internal compiler
# error (`make_ssa_name_fn` in ompexp) compiling nfft.c's OpenMP loops with
# VLAs on aarch64-darwin, while AppleClang handles the same pragmas fine.
set(NFFT3_OPENMP_FLAG "--enable-openmp")
set(NFFT3_CONFIGURE_CMD ${CMAKE_COMMAND} -E env ${NFFT3_ENV_FLAGS})
if(APPLE)
    if(NOT NN_BREW_LIBOMP_INCLUDE OR NOT NN_BREW_LIBOMP_PREFIX)
        message(FATAL_ERROR
            "libomp not configured: cmake/MacOSXDependencies.cmake must run "
            "before VendorNFFT3.cmake.")
    endif()

    set(_nn_nfft3_cc_wrapper "${CMAKE_BINARY_DIR}/nfft3_cc.sh")
    file(WRITE "${_nn_nfft3_cc_wrapper}"
        "#!/bin/sh\n"
        "args=\"\"\n"
        "for a in \"$@\"; do\n"
        "  if [ \"$a\" = \"-fopenmp\" ]; then\n"
        "    args=\"$args -Xpreprocessor -fopenmp\"\n"
        "  else\n"
        "    args=\"$args $a\"\n"
        "  fi\n"
        "done\n"
        "exec /usr/bin/clang $args -L${NN_BREW_LIBOMP_PREFIX}/lib -lomp\n")
    file(CHMOD "${_nn_nfft3_cc_wrapper}"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

    # Reuse the build-type base CFLAGS/CXXFLAGS computed above, plus the libomp
    # include dir so <omp.h> is found (the `-Xpreprocessor -fopenmp` itself is
    # injected by the wrapper for every `-fopenmp` it sees).
    set(_nn_nfft3_base_cflags "")
    set(_nn_nfft3_base_cxxflags "")
    foreach(_nn_flag IN LISTS NFFT3_ENV_FLAGS)
        if(_nn_flag MATCHES "^CFLAGS=(.*)")
            set(_nn_nfft3_base_cflags "${CMAKE_MATCH_1}")
        elseif(_nn_flag MATCHES "^CXXFLAGS=(.*)")
            set(_nn_nfft3_base_cxxflags "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    set(NFFT3_CONFIGURE_CMD ${CMAKE_COMMAND} -E env
        "CC=${_nn_nfft3_cc_wrapper}"
        "CXX=${_nn_nfft3_cc_wrapper}"
        "CFLAGS=${_nn_nfft3_base_cflags} -I${NN_BREW_LIBOMP_INCLUDE}"
        "CXXFLAGS=${_nn_nfft3_base_cxxflags} -I${NN_BREW_LIBOMP_INCLUDE}")
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

        BUILD_BYPRODUCTS "${NFFT3_INSTALL_DIR}/lib/${NFFT3_LIBRARY_FILE}"
        DEPENDS ${NFFT3_DEPENDS}

        CONFIGURE_COMMAND ${NFFT3_CONFIGURE_CMD}
            <SOURCE_DIR>/configure
            --prefix=<INSTALL_DIR>
            --disable-examples
            --disable-applications
            ${NFFT3_OPENMP_FLAG}
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
        IMPORTED_LOCATION             "${NFFT3_INSTALL_DIR}/lib/${NFFT3_LIBRARY_FILE}"
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
