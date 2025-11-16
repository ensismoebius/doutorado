cmake_minimum_required(VERSION 3.10)

# Project declaration
project(
		nn
		VERSION 0.1.0
		LANGUAGES CXX)

# Fetch and make available googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

## Top-level CMake: delegate to modular includes in cmake/

# Sets the CMake policies for this project
include(cmake/Policies.cmake)

# Project setup (language standards, etc.)
include(cmake/ProjectSetup.cmake)

# Do the required libraries check (package checks remain small and local)
include(cmake/PackageChecking.cmake)

# Vendor includes and helpers (cnpy, imgui, implot, eigen parallel settings)
include(cmake/VendorIncludes.cmake)

# Vendored matio handling (sets MATIO_ROOT_DIR, MATIO_INCLUDE_DIR, MATIO_LIBRARY, exposes MATIO::MATIO)
include(cmake/VendorMatio.cmake)

# Workarounds / shims for matio-cpp configure/export behavior
include(cmake/VendorMatioCppShim.cmake)

# Main wiring
include(cmake/Main.cmake)
