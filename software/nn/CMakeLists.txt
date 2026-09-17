cmake_minimum_required(VERSION 3.10)

# -----------------------------------------------------------------------------
# Top-level build entrypoint
#
# Intent:
# - Keep the root CMakeLists.txt small and “index-like”.
# - Delegate real logic to `cmake/*.cmake` modules for clarity and reuse.
#
# Include order matters:
# - Policies/tooling/analysis targets/flags are configured before vendored deps.
# - Vendored dependencies are configured before adding project subdirectories.
# -----------------------------------------------------------------------------

# Project declaration
project(nn VERSION 0.2.0 LANGUAGES CXX C)

enable_testing()

# Backend selection for backend-agnostic modules.
set(NN_BACKEND "XTensor" CACHE STRING "Primary tensor backend (XTensor, Device)")
set_property(CACHE NN_BACKEND PROPERTY STRINGS XTensor Device)

if(NOT NN_BACKEND STREQUAL "XTensor" AND
	 NOT NN_BACKEND STREQUAL "Device")
	message(FATAL_ERROR "NN_BACKEND must be one of: XTensor, Device")
endif()

if(NN_BACKEND STREQUAL "XTensor")
	add_compile_definitions(NN_BACKEND_XTENSOR)
elseif(NN_BACKEND STREQUAL "Device")
	add_compile_definitions(NN_BACKEND_DEVICE)
	# DeviceTensorBackend (include/tensor/DeviceTensorBackend.hpp) is a
	# documentation skeleton for contributors adding a new backend: it has no
	# real device allocator, no real hardware to be "unsupported" on, and its
	# math methods always delegate to an XTensorBackend host mirror (its
	# simulated device buffer only exercises copy-semantics bookkeeping in
	# tests). Warn loudly so selecting NN_BACKEND=Device is never mistaken for
	# testing a real accelerator — the "no silent fallback" policy this
	# project applies to real accelerator backends doesn't map onto a backend
	# that was never backed by hardware in the first place.
	message(WARNING
		"NN_BACKEND=Device selected: DeviceTensorBackend is a documentation "
		"skeleton (see include/tensor/DeviceTensorBackend.hpp) — it always runs "
		"on host (XTensorBackend) math, not a real accelerator. Use this only "
		"as a template for implementing a new backend, not to benchmark or "
		"validate GPU behaviour.")
endif()

message(STATUS "NN backend: ${NN_BACKEND}")

# Define paths for convenience
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

## Top-level CMake: delegate to modular includes in cmake/

# Sets the CMake policies for this project
include(cmake/Policies.cmake)

# Tooling setup (ccache, analysis tool detection, compile_commands.json export)
include(cmake/Tooling.cmake)

# Development and analysis custom targets
include(cmake/DevAndAnalysisTargets.cmake)

# Enforce backend implementation reference boundaries.
include(cmake/BackendImplementationGuard.cmake)

# Sets up the flags for CMake and compiler
include(cmake/Flags.cmake)

# Optional coverage helper (adds --coverage flags when NN_ENABLE_COVERAGE=ON)
include(cmake/EnableCoverage.cmake)

# Optional helpers for precompiled headers on selected targets
include(cmake/PrecompiledHeaders.cmake)

# macOS-only bootstrap: installs Homebrew + required formulae and configures
# OpenMP for AppleClang. No-op on all other platforms (Arch Linux / CI).
if(APPLE)
    include(cmake/MacOSXDependencies.cmake)
endif()

# Arch Linux bootstrap: installs required packages via pacman (+ yay for any
# AUR-only dependencies). No-op on all other platforms (macOS / Ubuntu / CI).
if(UNIX AND NOT APPLE)
    include(cmake/ArchLinuxDependencies.cmake)
endif()

# Ubuntu-family bootstrap: installs required packages via apt-get (incl. Ubuntu
# flavours, Mint, Pop!_OS). No-op on all other platforms (macOS / Arch / CI).
if(UNIX AND NOT APPLE)
    include(cmake/UbuntuDependencies.cmake)
endif()

# Do the required libraries check (package checks remain small and local)
include(cmake/PackageChecking.cmake)

# Vendor includes and helpers (cnpy, eigen parallel settings, fftw, nfft3)
include(cmake/VendorIncludes.cmake)

# Auto-discover layer headers and generate include/nn/layers/Layers.hpp
include(cmake/GenerateLayers.cmake)

# Optional Python bindings (nn_microscope) for software/experiment_microscope/.
# OFF by default — enabled by the `python-bindings` preset. VendorPybind11 is a
# no-op unless NN_BUILD_PY_BINDINGS is set.
option(NN_BUILD_PY_BINDINGS "Build the nn_microscope Python extension" OFF)
include(cmake/VendorPybind11.cmake)

# Source subdirectories
add_subdirectory(${SRC_DIR}/core)
add_subdirectory(${SRC_DIR}/demos)
add_subdirectory(${SRC_DIR}/experiments)

if(NN_BUILD_PY_BINDINGS)
    add_subdirectory(${SRC_DIR}/bindings)
endif()
