cmake_minimum_required(VERSION 3.10)

# Project declaration
project(nn VERSION 0.1.0 LANGUAGES CXX C)
    
## Top-level CMake: delegate to modular includes in cmake/

# Sets the CMake policies for this project
include(cmake/Policies.cmake)

# Sets up GoogleTest (downloads, configures, exposes GTest::GTest and GTest::Main)
include(cmake/VendorGTest.cmake)

# Project setup (language standards, etc.)
include(cmake/ProjectSetup.cmake)

# Do the required libraries check (package checks remain small and local)
include(cmake/PackageChecking.cmake)

# Vendor includes and helpers (cnpy, imgui, implot, eigen parallel settings, fftw, nfft3)
include(cmake/VendorIncludes.cmake)

add_subdirectory(${SRC_DIR}/core)
add_subdirectory(${SRC_DIR}/demos)
add_subdirectory(${SRC_DIR}/experiments)
