cmake_minimum_required(VERSION 3.10)

# Sets up the flags for CMake and compiler
include(Flags.cmake)

project(
  nn
  VERSION 0.1.0
  LANGUAGES CXX)

enable_testing()
include(CTest)

# Prevent vendored subprojects from enabling their own tests during configure
# so we don't require test frameworks for third-party code while configuring.
set(BUILD_TESTING OFF CACHE BOOL "Disable building tests in subprojects" FORCE)

# Define paths needed for the step bellow
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

# Do the required libraries check
include(PackageChecking.cmake)

# Download (if needed) another libraries
include(Cnpy.cmake)

# Add vendored matio library so matio-cpp can find MATIO::MATIO
include(lib/matio.cmake)

# Hint FindMATIO to look into the vendored matio sources if system matio is not installed
set(MATIO_ROOT_DIR "${CMAKE_SOURCE_DIR}/lib/matio" CACHE PATH "Folder contains MatIO")

# Point FindMATIO directly to the vendored headers and expected library output in the build tree.
# The library will be created under ${CMAKE_BINARY_DIR}/lib/libmatio.so (shared) or libmatio.a (static)
set(MATIO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/matio/src" CACHE PATH "MatIO include dir")
set(MATIO_LIBRARY "${CMAKE_BINARY_DIR}/lib/libmatio.so" CACHE FILEPATH "MatIO library file")

# If the vendored matio directory exists, create an IMPORTED target so downstream
# projects (matio-cpp) can link to MATIO::MATIO even if FindMATIO doesn't run.
if(EXISTS "${CMAKE_SOURCE_DIR}/lib/matio")
  # If the vendored matio adds a target named "matio", create an alias so
  # downstream consumers can link to MATIO::MATIO and CMake will ensure proper
  # build order. Otherwise, fall back to creating an IMPORTED target.
  if(TARGET matio AND NOT TARGET MATIO::MATIO)
    add_library(MATIO::MATIO ALIAS matio)
  elseif(NOT TARGET MATIO::MATIO)
    add_library(MATIO::MATIO UNKNOWN IMPORTED)
    set_target_properties(MATIO::MATIO PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/lib/matio/src"
      IMPORTED_LOCATION "${MATIO_LIBRARY}")
  endif()
endif()


include(lib/matio-cpp.cmake)

include(Imgui.cmake)

include(Implot.cmake)

include(EigenParallel.cmake)

include(Main.cmake)
