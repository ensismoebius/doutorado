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

# Disable vendored matio's command-line tools when vendoring to avoid
# building example/tool binaries that can fail under LTO + fast-linker.
set(MATIO_BUILD_TOOLS OFF CACHE BOOL "Disable building matio tools" FORCE)

# Create empty test CMakeLists
file(WRITE "${CMAKE_BINARY_DIR}/matio_disable_tests.cmake" "## tests disabled\n")
file(WRITE "${CMAKE_BINARY_DIR}/matio_disable_tools.cmake" "## tools disabled\n")

include(FetchContent)

FetchContent_Declare(
    matio
    GIT_REPOSITORY https://github.com/tbeu/matio.git
    GIT_TAG        v1.5.23 # Pinned to a specific tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE

    # Overwrite the test.cmake file to disable tests
    PATCH_COMMAND
      /bin/sh -c
        "${CMAKE_COMMAND} -E copy '${CMAKE_BINARY_DIR}/matio_disable_tests.cmake' '<SOURCE_DIR>/cmake/test.cmake' && ${CMAKE_COMMAND} -E copy '${CMAKE_BINARY_DIR}/matio_disable_tools.cmake' '<SOURCE_DIR>/cmake/tools.cmake'"
)

FetchContent_MakeAvailable(matio)

# Some distro HDF5 CMake packages can export an absolute soname path that no
# longer exists after package upgrades (for example: libhdf5.so.X.Y.Z exact
# patch level). When that happens, linking matio fails even though libhdf5 is
# installed. Detect this case and fall back to linker-name resolution.
if(TARGET MATIO::HDF5)
  get_target_property(_nn_matio_hdf5_links MATIO::HDF5 INTERFACE_LINK_LIBRARIES)
  set(_nn_matio_hdf5_has_missing_abs_path OFF)
  foreach(_nn_link_item IN LISTS _nn_matio_hdf5_links)
    if(IS_ABSOLUTE "${_nn_link_item}" AND
       _nn_link_item MATCHES "libhdf5\\.so" AND
       NOT EXISTS "${_nn_link_item}")
      set(_nn_matio_hdf5_has_missing_abs_path ON)
      break()
    endif()
  endforeach()

  if(_nn_matio_hdf5_has_missing_abs_path)
    message(WARNING
      "Detected stale absolute HDF5 soname path from MATIO::HDF5; "
      "falling back to linker-name resolution (-lhdf5).")
    set_property(TARGET MATIO::HDF5 PROPERTY INTERFACE_LINK_LIBRARIES hdf5)
  endif()
endif()

target_include_directories(matio SYSTEM
  PUBLIC
    "${matio_SOURCE_DIR}/src"
    "${matio_BINARY_DIR}/src"
)
target_compile_options(matio PRIVATE -w)
if(TARGET matio)
  # Disable LTO for vendored matio to avoid LTO + fast-linker (lld/mold)
  # version-script symbol assignment issues when building with LTO.
  target_compile_options(matio PRIVATE -fno-lto)
  set_property(TARGET matio PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
  set_target_properties(matio PROPERTIES CXX_CLANG_TIDY "")
endif()


# Provide a lightweight imported "matio" target if the subproject doesn't
# create it early enough. This helps export/install steps in other vendored
# CMakeLists succeed.
if(TARGET matio)
  add_library(MATIO::MATIO ALIAS matio)
endif()
