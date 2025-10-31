cmake_minimum_required(VERSION 3.10)

# Sets up the flags for CMake and compiler
include(Flags.cmake)

project(
  nn
  VERSION 0.1.0
  LANGUAGES CXX)

# Ensure our custom modules are visible to find_package() so we can provide
# a lightweight FindMATIO that prefers the vendored copy or an existing
# in-tree target. This must be set before matio-cpp does find_package(MATIO).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

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
## Disable vendored matio's own tests (they cause many failures in our
## environment and are not required for building the project). This only
## affects the in-tree vendor build and keeps third-party sources untouched.
set(MATIO_BUILD_TESTING OFF CACHE BOOL "Disable building tests in vendored matio" FORCE)

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

# If the vendored matio sources exist but the in-tree "matio" target is not yet
# created (or the subproject isn't configured to export it), create a lightweight
# IMPORTED target named "matio" that points to the expected build output so
# downstream export() and install() calls that reference "matio" succeed.
if(EXISTS "${CMAKE_SOURCE_DIR}/lib/matio" AND NOT TARGET matio)
  if(NOT TARGET matio)
    add_library(matio UNKNOWN IMPORTED)
    if(EXISTS "${CMAKE_BINARY_DIR}/lib/libmatio.so")
      set_target_properties(matio PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/lib/libmatio.so")
    elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libmatio.a")
      set_target_properties(matio PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/lib/libmatio.a")
    endif()
    set_target_properties(matio PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/lib/matio/src")
  endif()
endif()


# Temporarily disable install rules while configuring vendored matio-cpp so
# its install(EXPORT ...) calls don't try to export external targets (like
# the vendored "matio" target) which aren't part of the same export set.
## Workaround: make the matio-cpp subproject behave as if it's top-level so
## it doesn't generate install/export logic that requires the "matio" target
## to be part of the same export set. We set this cache variable before
## including the vendored matio-cpp project.
set(matioCpp_IS_TOP_LEVEL ON CACHE BOOL "Treat matio-cpp as top-level" FORCE)
# Provide a no-op implementation of install_basic_package_files so the
# vendored matio-cpp project can call it during configure without creating
# export/install files that reference external targets. This is safer than
# editing third-party CMake files.
if(NOT COMMAND install_basic_package_files)
  function(install_basic_package_files)
    # intentionally empty to avoid generating install/export files for vendored libs
  endfunction()
endif()
include(lib/matio-cpp.cmake)

include(Imgui.cmake)

include(Implot.cmake)

include(EigenParallel.cmake)

include(Main.cmake)
