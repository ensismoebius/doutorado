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
set(NN_BACKEND "XTensor" CACHE STRING "Primary tensor backend (XTensor, OpenCL, Device)")
set_property(CACHE NN_BACKEND PROPERTY STRINGS XTensor OpenCL Device)

if(NOT NN_BACKEND STREQUAL "XTensor" AND
	 NOT NN_BACKEND STREQUAL "OpenCL" AND
	 NOT NN_BACKEND STREQUAL "Device")
	message(FATAL_ERROR "NN_BACKEND must be one of: XTensor, OpenCL, Device")
endif()

if(NN_BACKEND STREQUAL "XTensor")
	add_compile_definitions(NN_BACKEND_XTENSOR)
elseif(NN_BACKEND STREQUAL "OpenCL")
	add_compile_definitions(NN_BACKEND_OPENCL)
elseif(NN_BACKEND STREQUAL "Device")
	add_compile_definitions(NN_BACKEND_DEVICE)
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

# Sets up the flags for CMake and compiler
include(cmake/Flags.cmake)

# Optional coverage helper (adds --coverage flags when NN_ENABLE_COVERAGE=ON)
include(cmake/EnableCoverage.cmake)

# Optional helpers for precompiled headers on selected targets
include(cmake/PrecompiledHeaders.cmake)

# Do the required libraries check (package checks remain small and local)
include(cmake/PackageChecking.cmake)

# Vendor includes and helpers (cnpy, imgui, implot, eigen parallel settings, fftw, nfft3)
include(cmake/VendorIncludes.cmake)

# Auto-discover layer headers and generate include/nn/layers/Layers.hpp
include(cmake/GenerateLayers.cmake)

# Source subdirectories
add_subdirectory(${SRC_DIR}/core)
add_subdirectory(${SRC_DIR}/demos)
add_subdirectory(${SRC_DIR}/experiments)
