##
## VendorMatio.cmake
##
## Purpose
## - Fetch and build `matio` (MATLAB .mat file IO library) as a vendored dependency.
##
## What it provides
## - Target: `matio` (as defined by the upstream project).
## - Alias: `MATIO::MATIO` when `matio` exists.
##
## Local policy
## - Disable upstream matio tests to keep the main configure/build fast and quiet.
##   This is done by overwriting `<SOURCE_DIR>/cmake/test.cmake` during FetchContent.
## - Expose include dirs as SYSTEM and suppress warnings/clang-tidy for vendor code.
##

# VendorMatio.cmake
# Configure vendored lib/matio presence and make MATIO::MATIO available

# Disable vendored matio's own tests by default (safe for most builds)
set(MATIO_BUILD_TESTS OFF CACHE BOOL "Disable building tests in vendored matio" FORCE)

# Create empty test CMakeLists
file(WRITE "${CMAKE_BINARY_DIR}/matio_disable_tests.cmake" "## tests disabled\n")

include(FetchContent)

FetchContent_Declare(
    matio
    GIT_REPOSITORY https://github.com/tbeu/matio.git
    GIT_TAG        v1.5.23 # Pinned to a specific tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE

    # Overwrite the test.cmake file to disable tests
    PATCH_COMMAND
        ${CMAKE_COMMAND} -E copy
            "${CMAKE_BINARY_DIR}/matio_disable_tests.cmake"
            "<SOURCE_DIR>/cmake/test.cmake"
)

FetchContent_MakeAvailable(matio)

target_include_directories(matio SYSTEM
  PUBLIC
    "${matio_SOURCE_DIR}/src"
    "${matio_BINARY_DIR}/src"
)
target_compile_options(matio PRIVATE -w)
if(TARGET matio)
    set_target_properties(matio PROPERTIES CXX_CLANG_TIDY "")
endif()


# Provide a lightweight imported "matio" target if the subproject doesn't
# create it early enough. This helps export/install steps in other vendored
# CMakeLists succeed.
if(TARGET matio)
  add_library(MATIO::MATIO ALIAS matio)
endif()
