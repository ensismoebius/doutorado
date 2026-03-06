##
## VendorCLI11.cmake
##
## Purpose
## - Fetch and configure CLI11 (command-line parser) as a vendored dependency.
##
## What it provides
## - Target: `CLI11::CLI11` (from upstream).
##
## Local policy
## - Pin version for reproducibility.
## - Disable clang-tidy diagnostics for third-party target when possible.
##

include(FetchContent)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.5.0
)

FetchContent_MakeAvailable(CLI11)

if(TARGET CLI11)
    set_target_properties(CLI11 PROPERTIES CXX_CLANG_TIDY "")
endif()
