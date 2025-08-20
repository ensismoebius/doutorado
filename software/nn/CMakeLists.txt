cmake_minimum_required(VERSION 3.10)

project(
  nn
  VERSION 0.1.0
  LANGUAGES CXX)

# Define paths needed for the step bellow
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

# Sets up the flags for CMake and compiler
include(Flags.cmake)

# Do the required libraries check
include(PackageChecking.cmake)

# Download (if needed) another libraries
include(Cnpy.cmake)

include(Main.cmake)
