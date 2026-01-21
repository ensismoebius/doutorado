##
## VendorArgparse.cmake
##
## Purpose
## - Fetch and configure `argparse` (command line parser) as a vendored dependency.
##
## What it provides
## - Targets: `argparse`.
## - Include path: exposes `argparse`'s include directory.
##
## Local policy
## - Suppress warnings for third-party code (`-w`) and disable clang-tidy on the target.
## - Pin to a specific commit for reproducibility.
##

include(FetchContent)

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG        v3.2 # Pinned to a specific commit for reproducibility
)

FetchContent_MakeAvailable(argparse)

# Disable clang-tidy for this vendor
set_target_properties(argparse PROPERTIES CXX_CLANG_TIDY "")
