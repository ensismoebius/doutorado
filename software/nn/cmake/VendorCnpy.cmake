##
## VendorCnpy.cmake
##
## Purpose
## - Fetch and configure `cnpy` (NumPy .npy/.npz IO) as a vendored dependency.
##
## What it provides
## - Targets: `cnpy` and (in upstream) `cnpy-static`.
## - Include path: exposes `${cnpy_SOURCE_DIR}` as a SYSTEM include directory.
##
## Local policy
## - Suppress warnings for third-party code (`-w`) and disable clang-tidy on cnpy targets.
## - Pin to a specific commit for reproducibility.
##

# Cnpy configuration
# Cnpy lets you read and write to .npy and .npz formats in C++.

include(FetchContent)

FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG        4e8810b # Pinned to a specific commit for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(cnpy)

target_include_directories(cnpy INTERFACE SYSTEM "$<BUILD_INTERFACE:${cnpy_SOURCE_DIR}>")
target_compile_options(cnpy PRIVATE -w)
target_compile_options(cnpy PRIVATE -fno-lto)
target_compile_options(cnpy-static PRIVATE -w)
target_compile_options(cnpy-static PRIVATE -fno-lto)

# Disable clang-tidy for this vendor
if(TARGET cnpy)
    # Disable LTO for vendored cnpy to avoid LTO + fast-linker issues
    target_compile_options(cnpy PRIVATE -fno-lto)
    set_property(TARGET cnpy PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    set_target_properties(cnpy PROPERTIES CXX_CLANG_TIDY "")
endif()
if(TARGET cnpy-static)
    target_compile_options(cnpy-static PRIVATE -fno-lto)
    set_property(TARGET cnpy-static PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    set_target_properties(cnpy-static PROPERTIES CXX_CLANG_TIDY "")
endif()