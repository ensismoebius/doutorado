#+#+#+#+-----------------------------------------------------------------------
# MacOSXDependencies.cmake
#
# macOS-only bootstrap for the system dependencies.
#
# Purpose
# - Ensure Homebrew is installed (installing it via the official installer
#   script when missing) and install every Homebrew package required to
#   configure/build/test this project.
# - Configure OpenMP support for AppleClang: unlike GCC/LLVM on Linux, the
#   AppleClang toolchain has no built-in OpenMP runtime, so we point CMake's
#   FindOpenMP at Homebrew's `libomp`.
#
# Platform policy
# - This module is a no-op on every non-Apple platform (`return()`), so Arch
#   Linux / Ubuntu / Docker CI builds are completely unaffected.
# - It is included from the top-level CMakeLists.txt inside an `if(APPLE)`
#   guard, which keeps Linux configure paths byte-for-byte identical.
#+#+#+#+-----------------------------------------------------------------------

if(NOT APPLE)
    return()
endif()

# ----------------------------------------------------------------------------
# 1. Ensure Homebrew is installed.
# ----------------------------------------------------------------------------
find_program(NN_BREW_EXECUTABLE brew)
if(NOT NN_BREW_EXECUTABLE)
    message(STATUS "Homebrew not found. Installing Homebrew (the installer needs sudo)...")
    execute_process(
        COMMAND /bin/bash -c
            "NONINTERACTIVE=1 $(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        RESULT_VARIABLE NN_BREW_INSTALL_RES
        ERROR_VARIABLE  NN_BREW_INSTALL_ERR
    )
    if(NOT NN_BREW_INSTALL_RES EQUAL 0)
        message(FATAL_ERROR
            "Homebrew installation failed:\n${NN_BREW_INSTALL_ERR}\n"
            "Install Homebrew manually (https://brew.sh) and re-run cmake.")
    endif()
    find_program(NN_BREW_EXECUTABLE brew)
    if(NOT NN_BREW_EXECUTABLE)
        message(FATAL_ERROR
            "Homebrew was installed but `brew` is not on PATH. Add "
            "/opt/homebrew/bin (Apple Silicon) or /usr/local/bin (Intel) to "
            "PATH and re-run cmake.")
    endif()
endif()
message(STATUS "Homebrew found at: ${NN_BREW_EXECUTABLE}")

# ----------------------------------------------------------------------------
# 2. Homebrew packages required to configure/build/test this project.
# ----------------------------------------------------------------------------
# Notes:
# - `sdl2` is an alias for `sdl2-compat` in Homebrew; the alias keeps working
#   for `find_package(SDL2)` / pkg-config, which is all this project needs.
# - `python3` is an alias for the current `python@3.x` formula.
# - Compilers (Xcode Command Line Tools, i.e. the system AppleClang) and
#   `git` cannot/should not be managed by brew and are assumed present.
set(NN_BREW_DEPENDENCIES
    pkgconf             # pkg-config, required by PackageChecking.cmake
    cmake               # build system
    ninja               # Ninja generator used by the CMake presets
    autoconf            # autotools build of vendored NFFT3
    automake            # autotools build of vendored NFFT3
    libtool             # autotools build of vendored NFFT3
    m4                  # autotools build of vendored NFFT3
    openblas            # BLAS/LAPACK provider (pkg-config `openblas`)
    libomp              # OpenMP runtime for AppleClang
    sdl2                # SDL2 (required by PackageChecking.cmake)
    fftw                # FFTW3 (avoids the vendored FFTW autotools build)
    hdf5                # HDF5 (required by vendored matio)
    eigen               # Eigen3 (EEG/Audio data loaders)
    glfw                # GLFW3 (snn_spike_plotter demo, pkg-config glfw3)
    ccache              # optional compiler cache (Tooling.cmake)
    python3             # Python for the vendored venv provisioning
)

# Query installed formulae once and install only what is missing. `brew install`
# on an already-installed formula is a harmless no-op, but checking first keeps
# `cmake --preset=...` reconfigures fast.
execute_process(
    COMMAND ${NN_BREW_EXECUTABLE} list --formula --versions
    OUTPUT_VARIABLE NN_BREW_INSTALLED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(NN_BREW_TO_INSTALL)
foreach(NN_DEP IN LISTS NN_BREW_DEPENDENCIES)
    # Match "<formula> <version>" on its own line (formula name followed by a
    # space at line start). Aliases like `sdl2`/`python3` that resolve to a
    # differently-named formula simply fall through to `brew install`, which
    # is a no-op when the aliased formula is already present.
    if(NOT NN_BREW_INSTALLED MATCHES "(^|\n)${NN_DEP} ")
        list(APPEND NN_BREW_TO_INSTALL ${NN_DEP})
    endif()
endforeach()

if(NN_BREW_TO_INSTALL)
    message(STATUS "Installing missing Homebrew packages: ${NN_BREW_TO_INSTALL}")
    execute_process(
        COMMAND ${NN_BREW_EXECUTABLE} install ${NN_BREW_TO_INSTALL}
        RESULT_VARIABLE NN_BREW_DEPS_RES
        ERROR_VARIABLE  NN_BREW_DEPS_ERR
    )
    if(NOT NN_BREW_DEPS_RES EQUAL 0)
        message(FATAL_ERROR
            "`brew install` failed:\n${NN_BREW_DEPS_ERR}")
    endif()
else()
    message(STATUS "All required Homebrew packages are already installed.")
endif()

# ----------------------------------------------------------------------------
# 3. OpenMP support for AppleClang via Homebrew `libomp`.
# ----------------------------------------------------------------------------
# FindOpenMP for AppleClang cannot locate the runtime by itself; the widely
# used recipe is to pre-seed the FindOpenMP result variables so the imported
# `OpenMP::OpenMP_C` / `OpenMP::OpenMP_CXX` targets get the right flags and
# link library. These variables must be set before `find_package(OpenMP)`
# (which happens later, in cmake/PackageChecking.cmake).
execute_process(
    COMMAND ${NN_BREW_EXECUTABLE} --prefix libomp
    OUTPUT_VARIABLE NN_BREW_LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${NN_BREW_LIBOMP_PREFIX}")
    message(FATAL_ERROR
        "Could not resolve the Homebrew libomp prefix (got: ${NN_BREW_LIBOMP_PREFIX})")
endif()
set(NN_BREW_LIBOMP_INCLUDE "${NN_BREW_LIBOMP_PREFIX}/include")
set(NN_BREW_LIBOMP_LIBRARY "${NN_BREW_LIBOMP_PREFIX}/lib/libomp.dylib")

set(OpenMP_C_FLAGS    "-Xpreprocessor -fopenmp -I${NN_BREW_LIBOMP_INCLUDE}")
set(OpenMP_CXX_FLAGS  "-Xpreprocessor -fopenmp -I${NN_BREW_LIBOMP_INCLUDE}")
set(OpenMP_C_LIB_NAMES   omp)
set(OpenMP_CXX_LIB_NAMES omp)
set(OpenMP_omp_LIBRARY "${NN_BREW_LIBOMP_LIBRARY}")
set(OpenMP_C_LIBRARIES  "${NN_BREW_LIBOMP_LIBRARY}")
set(OpenMP_CXX_LIBRARIES "${NN_BREW_LIBOMP_LIBRARY}")

message(STATUS "OpenMP (libomp) configured for AppleClang: ${NN_BREW_LIBOMP_LIBRARY}")

# AppleClang ships no <omp.h>; it lives in Homebrew's keg-only libomp. On Linux
# <omp.h> is a system header, so exposing libomp's include dir globally to every
# target replicates that (project headers such as include/layers/* include
# <omp.h> from PUBLIC headers, so it cannot be a per-target PRIVATE include).
if(EXISTS "${NN_BREW_LIBOMP_INCLUDE}")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${NN_BREW_LIBOMP_INCLUDE}")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${NN_BREW_LIBOMP_INCLUDE}")
    message(STATUS "Adding keg-only libomp include dir: ${NN_BREW_LIBOMP_INCLUDE}")
endif()

# ----------------------------------------------------------------------------
# 4. Make keg-only Homebrew pkg-config modules discoverable.
# ----------------------------------------------------------------------------
# `openblas` is keg-only in Homebrew: its `openblas.pc` is not on pkg-config's
# default search path. Prepend its pkgconfig dir so the required
# `pkg_check_modules(OPENBLAS REQUIRED openblas)` in PackageChecking.cmake
# succeeds. This environment change only affects this configure process.
execute_process(
    COMMAND ${NN_BREW_EXECUTABLE} --prefix openblas
    OUTPUT_VARIABLE NN_BREW_OPENBLAS_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(EXISTS "${NN_BREW_OPENBLAS_PREFIX}/lib/pkgconfig")
    set(ENV{PKG_CONFIG_PATH}
        "${NN_BREW_OPENBLAS_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    message(STATUS "PKG_CONFIG_PATH += ${NN_BREW_OPENBLAS_PREFIX}/lib/pkgconfig")
endif()
# openblas is keg-only, so its headers (cblas.h, lapacke.h, ...) are not on the
# default include path. xtensor-blas needs them (it includes `<cblas.h>` /
# `<lapacke.h>`), so expose them to every target on Apple.
if(EXISTS "${NN_BREW_OPENBLAS_PREFIX}/include")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${NN_BREW_OPENBLAS_PREFIX}/include")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${NN_BREW_OPENBLAS_PREFIX}/include")
    message(STATUS "Adding keg-only openblas include dir: ${NN_BREW_OPENBLAS_PREFIX}/include")
endif()

# ----------------------------------------------------------------------------
# 5. Expose the Homebrew prefix include dir to every target.
# ----------------------------------------------------------------------------
# On Linux all these packages install headers into the system include path
# (/usr/include), so targets that include a header without linking the package
# (e.g. the vendored imgui backend `imgui_impl_glfw.cpp` includes
# `<GLFW/glfw3.h>`) still compile. AppleClang does NOT search
# /opt/homebrew/include by default, so replicate the Linux behaviour here.
execute_process(
    COMMAND ${NN_BREW_EXECUTABLE} --prefix
    OUTPUT_VARIABLE NN_BREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(EXISTS "${NN_BREW_PREFIX}/include")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${NN_BREW_PREFIX}/include")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${NN_BREW_PREFIX}/include")
    message(STATUS "Adding Homebrew prefix include dir: ${NN_BREW_PREFIX}/include")
endif()
