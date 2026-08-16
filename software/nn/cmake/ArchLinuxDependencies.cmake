#+#+#+#+-----------------------------------------------------------------------
# ArchLinuxDependencies.cmake
#
# Arch Linux bootstrap for the system dependencies.
#
# Purpose
# - Install every package required to configure/build/test this project, using
#   `pacman` for official-repo packages and `yay` (the AUR helper) for anything
#   that only exists in the AUR.
# - Unlike Homebrew on macOS, Arch packages install into the system prefix
#   (/usr), so headers, libraries and pkg-config modules are already on the
#   default search paths: no include-dir or PKG_CONFIG_PATH surgery is needed
#   here. OpenMP on Linux is also handled by FindOpenMP out of the box (GCC
#   ships libgomp; the `openmp` package provides the LLVM runtime for the clang
#   presets), so there is no AppleClang-style libomp pre-seeding.
#
# Platform policy
# - This module is a no-op on every non-Arch platform (`return()`), so macOS /
#   Ubuntu / Docker CI builds are completely unaffected.
# - It is included from the top-level CMakeLists.txt inside an `if(UNIX AND NOT
#   APPLE)` guard; the /etc/os-release `ID=arch` check below keeps non-Arch
#   Linux configure paths byte-for-byte identical.
#+#+#+#+-----------------------------------------------------------------------

# Only run on Arch Linux. Reject everything else before doing anything.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
endif()
file(READ "/etc/os-release" NN_OS_RELEASE)
if(NOT NN_OS_RELEASE MATCHES "(^|\n)ID=arch")
    return()
endif()
message(STATUS "Arch Linux detected: bootstrapping system dependencies...")

# ----------------------------------------------------------------------------
# 1. Determine privilege escalation for pacman.
# ----------------------------------------------------------------------------
# pacman needs root. If we are already root use it directly; otherwise prepend
# `sudo` (which prompts for a password on the terminal if one is needed).
execute_process(
    COMMAND id -u
    OUTPUT_VARIABLE NN_EUID
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(NN_SUDO)
if(NOT NN_EUID STREQUAL "0")
    set(NN_SUDO sudo)
    message(STATUS "Not running as root: pacman commands will use `sudo`.")
endif()

# ----------------------------------------------------------------------------
# 2. pacman packages required to configure/build/test this project.
# ----------------------------------------------------------------------------
# Notes:
# - `pkgconf` is Arch's pkg-config implementation (provides the `pkg-config`
#   command, required by PackageChecking.cmake).
# - `python` is the Arch package providing `python3`.
# - `base-devel` brings the toolchain (gcc, make, ...) and the autotools stack;
#   the individual autoconf/automake/libtool/m4 entries are still listed for
#   clarity (the vendored NFFT3 build uses them) and are no-ops when already
#   pulled in by base-devel.
# - `git` is needed to bootstrap `yay` from the AUR below.
# - OpenCL headers/loader come from `opencl-headers` + `ocl-icd`. A vendor ICD
#   runtime (e.g. `intel-compute-runtime` for the Intel iGPU, or Mesa's
#   rusticl) is hardware-specific and intentionally NOT auto-installed.
set(NN_PACMAN_DEPENDENCIES
    pkgconf             # pkg-config, required by PackageChecking.cmake
    cmake               # build system
    ninja               # Ninja generator used by the CMake presets
    autoconf            # autotools build of vendored NFFT3
    automake            # autotools build of vendored NFFT3
    libtool             # autotools build of vendored NFFT3
    m4                  # autotools build of vendored NFFT3
    openblas            # BLAS/LAPACK provider (pkg-config `openblas`)
    openmp              # LLVM OpenMP runtime for the clang presets (GCC uses libgomp)
    sdl2-compat         # SDL2 compatibility layer (provides SDL2 for PackageChecking.cmake)
    fftw                # FFTW3 (avoids the vendored FFTW autotools build)
    hdf5                # HDF5 (required by vendored matio)
    eigen               # Eigen3 (EEG/Audio data loaders)
    glfw                # GLFW3 (snn_spike_plotter demo, pkg-config glfw3)
    ccache              # optional compiler cache (Tooling.cmake)
    python              # Python for the vendored venv provisioning
    git                 # clones the AUR yay source
    base-devel          # toolchain + autotools (also required to build yay)
    opencl-headers      # Khronos OpenCL headers (CL/cl.h)
    ocl-icd             # OpenCL ICD loader
)

# Query installed packages once and install only what is missing. `pacman -S`
# is idempotent, but checking first keeps `cmake --preset=...` reconfigures fast.
set(NN_PACMAN_TO_INSTALL)
foreach(NN_DEP IN LISTS NN_PACMAN_DEPENDENCIES)
    execute_process(
        COMMAND pacman -Q ${NN_DEP}
        RESULT_VARIABLE NN_PACMAN_QUERY_RES
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT NN_PACMAN_QUERY_RES EQUAL 0)
        list(APPEND NN_PACMAN_TO_INSTALL ${NN_DEP})
    endif()
endforeach()

if(NN_PACMAN_TO_INSTALL)
    message(STATUS "Installing missing pacman packages: ${NN_PACMAN_TO_INSTALL}")
    execute_process(
        COMMAND ${NN_SUDO} pacman -S --noconfirm --needed ${NN_PACMAN_TO_INSTALL}
        RESULT_VARIABLE NN_PACMAN_DEPS_RES
        ERROR_VARIABLE  NN_PACMAN_DEPS_ERR
    )
    if(NOT NN_PACMAN_DEPS_RES EQUAL 0)
        message(FATAL_ERROR
            "`pacman -S` failed:\n${NN_PACMAN_DEPS_ERR}")
    endif()
else()
    message(STATUS "All required pacman packages are already installed.")
endif()

# ----------------------------------------------------------------------------
# 3. AUR dependencies via yay.
# ----------------------------------------------------------------------------
# Everything in the list above lives in the official Arch repositories, so by
# default no AUR package is needed and yay is not installed. If the project
# later gains an AUR-only dependency, add it to NN_AUR_DEPENDENCIES and this
# block installs yay (from source via makepkg) and then pulls the packages.
set(NN_AUR_DEPENDENCIES
    # Example: "intel-compute-runtime-bin"  # AUR-only OpenCL runtime, if desired
)

if(NN_AUR_DEPENDENCIES)
    find_program(NN_YAY_EXECUTABLE yay)
    if(NOT NN_YAY_EXECUTABLE)
        if(NN_EUID STREQUAL "0")
            message(FATAL_ERROR
                "yay cannot be built as root (makepkg refuses to run as root). "
                "Re-run cmake as a normal user so yay can be built and installed.")
        endif()
        message(STATUS "yay not found. Building yay from the AUR...")
        set(NN_YAY_BUILD_DIR "${CMAKE_BINARY_DIR}/yay-build")
        execute_process(
            COMMAND git clone --depth 1 https://aur.archlinux.org/yay.git
                "${NN_YAY_BUILD_DIR}"
            RESULT_VARIABLE NN_YAY_CLONE_RES
            ERROR_VARIABLE  NN_YAY_CLONE_ERR
        )
        if(NOT NN_YAY_CLONE_RES EQUAL 0)
            message(FATAL_ERROR
                "`git clone` of yay failed:\n${NN_YAY_CLONE_ERR}")
        endif()
        execute_process(
            COMMAND makepkg -si --noconfirm
            WORKING_DIRECTORY "${NN_YAY_BUILD_DIR}"
            RESULT_VARIABLE NN_YAY_BUILD_RES
            ERROR_VARIABLE  NN_YAY_BUILD_ERR
        )
        if(NOT NN_YAY_BUILD_RES EQUAL 0)
            message(FATAL_ERROR
                "`makepkg -si` of yay failed:\n${NN_YAY_BUILD_ERR}")
        endif()
        find_program(NN_YAY_EXECUTABLE yay)
        if(NOT NN_YAY_EXECUTABLE)
            message(FATAL_ERROR
                "yay was built but is not on PATH. Add ~/.local/bin or the "
                "makepkg install prefix to PATH and re-run cmake.")
        endif()
    endif()
    message(STATUS "yay found at: ${NN_YAY_EXECUTABLE}")

    message(STATUS "Installing AUR packages: ${NN_AUR_DEPENDENCIES}")
    execute_process(
        COMMAND ${NN_YAY_EXECUTABLE} -S --noconfirm --needed ${NN_AUR_DEPENDENCIES}
        RESULT_VARIABLE NN_YAY_DEPS_RES
        ERROR_VARIABLE  NN_YAY_DEPS_ERR
    )
    if(NOT NN_YAY_DEPS_RES EQUAL 0)
        message(FATAL_ERROR
            "`yay -S` failed:\n${NN_YAY_DEPS_ERR}")
    endif()
else()
    message(STATUS "No AUR dependencies requested; yay bootstrap skipped.")
endif()

# ----------------------------------------------------------------------------
# 4. OpenMP on Linux.
# ----------------------------------------------------------------------------
# Nothing to configure: GCC ships libgomp and the `openmp` package installed
# above provides the LLVM runtime for the clang presets, so FindOpenMP (run later in
# cmake/PackageChecking.cmake) resolves `OpenMP::OpenMP_CXX` natively. The
# AppleClang libomp pre-seeding in MacOSXDependencies.cmake does not apply.
# NFFT3's vendored build likewise takes the stock `--enable-openmp` path here
# (the -Xpreprocessor wrapper in VendorNFFT3.cmake is APPLE-only).
