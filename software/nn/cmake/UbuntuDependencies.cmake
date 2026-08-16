#+#+#+#+-----------------------------------------------------------------------
# UbuntuDependencies.cmake
#
# Ubuntu / Debian-family bootstrap for the system dependencies.
#
# Purpose
# - Install every package required to configure/build/test this project, using
#   `apt-get` for packages from the distribution repositories (the Debian family
#   has no Homebrew/yay/AUR analogue for build deps, so apt alone suffices).
# - Unlike Homebrew on macOS, apt packages install into the system prefix
#   (/usr), so headers, libraries and pkg-config modules are already on the
#   default search paths: no include-dir or PKG_CONFIG_PATH surgery is needed
#   here. OpenMP on Linux is also handled by FindOpenMP out of the box (GCC
#   ships libgomp; `libomp-dev` provides the LLVM runtime for the clang
#   presets), so there is no AppleClang-style libomp pre-seeding.
#
# Platform policy
# - This module is a no-op on every non-Ubuntu-family platform (`return()`), so
#   macOS / Arch / Fedora / Docker CI builds are completely unaffected.
# - It is included from the top-level CMakeLists.txt inside an `if(UNIX AND NOT
#   APPLE)` guard; the /etc/os-release check below keeps non-Ubuntu configure
#   paths byte-for-byte identical. The ID/ID_LIKE match is intentionally broad
#   so Ubuntu flavours (Kubuntu, Xubuntu), Linux Mint, Pop!_OS and elementary
#   OS all hit the same bootstrap.
#+#+#+#+-----------------------------------------------------------------------

# Only run on Ubuntu-family systems. Reject everything else before doing
# anything. Ubuntu proper has `ID=ubuntu`; flavours/derivatives (Mint, Pop!_OS,
# elementary) set `ID_LIKE=ubuntu`.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
endif()
file(READ "/etc/os-release" NN_OS_RELEASE)
if(NOT NN_OS_RELEASE MATCHES "(^|\n)(ID=ubuntu|ID_LIKE=[^\n]*ubuntu)")
    return()
endif()
message(STATUS "Ubuntu-family system detected: bootstrapping system dependencies...")

# ----------------------------------------------------------------------------
# 1. Determine privilege escalation for apt-get.
# ----------------------------------------------------------------------------
# apt-get needs root. If we are already root use it directly; otherwise prepend
# `sudo` (which prompts for a password on the terminal if one is needed).
execute_process(
    COMMAND id -u
    OUTPUT_VARIABLE NN_EUID
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(NN_SUDO)
if(NOT NN_EUID STREQUAL "0")
    set(NN_SUDO sudo)
    message(STATUS "Not running as root: apt-get commands will use `sudo`.")
endif()

# ----------------------------------------------------------------------------
# 2. apt packages required to configure/build/test this project.
# ----------------------------------------------------------------------------
# Notes:
# - `pkg-config` is the pkg-config implementation on Debian/Ubuntu (required by
#   PackageChecking.cmake).
# - `python3` is the Python 3 interpreter package.
# - `build-essential` brings the toolchain (gcc/g++, make) and the autotools
#   stack; the individual autoconf/automake/libtool/m4 entries are still listed
#   for clarity (the vendored NFFT3 build uses them) and are no-ops when already
#   pulled in by build-essential.
# - `git` is kept for parity with the other bootstrap modules (not strictly
#   needed by apt itself).
# - OpenCL headers/loader come from `opencl-headers` + `ocl-icd-libopencl1`.
#   `ocl-icd-opencl-dev` is listed because it also ships the `OpenCL.pc`
#   pkg-config module that PackageChecking.cmake probes. A vendor ICD runtime
#   (e.g. `intel-opencl-icd`, `mesa-opencl-icd`, `pocl-opencl-icd`) is
#   hardware-specific and intentionally NOT auto-installed.
set(NN_APT_DEPENDENCIES
    pkg-config          # pkg-config, required by PackageChecking.cmake
    cmake               # build system
    ninja-build         # Ninja generator used by the CMake presets
    autoconf            # autotools build of vendored NFFT3
    automake            # autotools build of vendored NFFT3
    libtool             # autotools build of vendored NFFT3
    m4                  # autotools build of vendored NFFT3
    libopenblas-dev     # BLAS/LAPACK provider (pkg-config `openblas`)
    libomp-dev          # LLVM OpenMP runtime for the clang presets (GCC uses libgomp)
    libsdl2-dev         # SDL2 (required by PackageChecking.cmake)
    libfftw3-dev        # FFTW3 (avoids the vendored FFTW autotools build)
    libhdf5-dev         # HDF5 (required by vendored matio)
    libeigen3-dev       # Eigen3 (EEG/Audio data loaders)
    libglfw3-dev        # GLFW3 (snn_spike_plotter demo, pkg-config glfw3)
    ccache              # optional compiler cache (Tooling.cmake)
    python3             # Python for the vendored venv provisioning
    git                 # version control (parity with other bootstraps)
    build-essential     # toolchain + autotools
    opencl-headers      # Khronos OpenCL headers (CL/cl.h)
    ocl-icd-opencl-dev  # OpenCL ICD loader + OpenCL.pc pkg-config module
    libsqlite3-dev      # SQLite3 (required by the data loaders, SQLite3::SQLite3)
    zlib1g-dev          # ZLIB (find_package(ZLIB REQUIRED) in the data loaders)
)

# Query installed packages once and install only what is missing. `dpkg -s`
# returns 0 when the package is present and configured; checking first keeps
# `cmake --preset=...` reconfigures fast (no apt cache churn when nothing is
# missing).
set(NN_APT_TO_INSTALL)
foreach(NN_DEP IN LISTS NN_APT_DEPENDENCIES)
    execute_process(
        COMMAND dpkg -s ${NN_DEP}
        RESULT_VARIABLE NN_DPKG_QUERY_RES
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT NN_DPKG_QUERY_RES EQUAL 0)
        list(APPEND NN_APT_TO_INSTALL ${NN_DEP})
    endif()
endforeach()

if(NN_APT_TO_INSTALL)
    message(STATUS "Refreshing apt package index...")
    execute_process(
        COMMAND ${NN_SUDO} apt-get update
        RESULT_VARIABLE NN_APT_UPDATE_RES
        ERROR_VARIABLE  NN_APT_UPDATE_ERR
    )
    if(NOT NN_APT_UPDATE_RES EQUAL 0)
        message(FATAL_ERROR
            "`apt-get update` failed:\n${NN_APT_UPDATE_ERR}")
    endif()

    message(STATUS "Installing missing apt packages: ${NN_APT_TO_INSTALL}")
    execute_process(
        COMMAND ${NN_SUDO} apt-get install -y --no-install-recommends
            ${NN_APT_TO_INSTALL}
        RESULT_VARIABLE NN_APT_INSTALL_RES
        ERROR_VARIABLE  NN_APT_INSTALL_ERR
    )
    if(NOT NN_APT_INSTALL_RES EQUAL 0)
        message(FATAL_ERROR
            "`apt-get install` failed:\n${NN_APT_INSTALL_ERR}")
    endif()
else()
    message(STATUS "All required apt packages are already installed.")
endif()

# ----------------------------------------------------------------------------
# 3. OpenMP on Linux.
# ----------------------------------------------------------------------------
# Nothing to configure: GCC ships libgomp and the `libomp-dev` package installed
# above provides the LLVM runtime for the clang presets, so FindOpenMP (run later
# in cmake/PackageChecking.cmake) resolves `OpenMP::OpenMP_CXX` natively. The
# AppleClang libomp pre-seeding in MacOSXDependencies.cmake does not apply.
# NFFT3's vendored build likewise takes the stock `--enable-openmp` path here
# (the -Xpreprocessor wrapper in VendorNFFT3.cmake is APPLE-only).
