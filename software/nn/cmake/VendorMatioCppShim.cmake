# VendorMatioCppShim.cmake
# Make matio-cpp configure safely inside the top-level project without
# generating install/export files that would reference external targets.

# Treat matio-cpp as top-level to reduce packaging/export side-effects
set(matioCpp_IS_TOP_LEVEL ON CACHE BOOL "Treat matio-cpp as top-level" FORCE)

# Provide a no-op implementation of install_basic_package_files so the
# vendored matio-cpp project can call it during configure without creating
# export/install files that reference external targets. This avoids editing
# third-party files.
if(NOT COMMAND install_basic_package_files)
  function(install_basic_package_files)
    # intentionally empty to avoid generating install/export files for vendored libs
  endfunction()
endif()

# Include vendored matio-cpp
include(FetchContent)

FetchContent_Declare(
    matio-cpp
    GIT_REPOSITORY https://github.com/ami-iit/matio-cpp.git
    GIT_TAG        HEAD # Consider using a specific commit hash or tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Provide matio information to matio-cpp's build system
set(MATIO_ROOT_DIR "${matio_BINARY_DIR}")
set(MATIO_INCLUDE_DIR "${matio_SOURCE_DIR}/src") # matio.h is in matio-src/src
set(MATIO_LIBRARY "${matio_BINARY_DIR}/libmatio.so") # Or .a, depending on build type

set(MATIO_FOUND TRUE) # Explicitly set to TRUE

FetchContent_Declare(
    matio-cpp
    GIT_REPOSITORY https://github.com/ami-iit/matio-cpp.git
    GIT_TAG        HEAD # Consider using a specific commit hash or tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(matio-cpp)
