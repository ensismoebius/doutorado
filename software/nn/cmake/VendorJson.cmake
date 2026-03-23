##
## VendorJson.cmake
##
## Purpose
## - Fetch and configure `nlohmann/json` as a vendored dependency.
##
## What it provides
## - Target: `nlohmann_json::nlohmann_json`
## - Include path: exposes `${nlohmann_json_SOURCE_DIR}/include` as a SYSTEM include directory.
##

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.2 # pinned to a known release
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(nlohmann_json)

# The nlohmann::json FetchContent provides a header-only target
# `nlohmann_json::nlohmann_json`. Do not attempt to modify properties
# on an ALIAS/INTERFACE target here; other modules that consume the
# target should check `if(TARGET nlohmann_json::nlohmann_json)` and
# link against it. Suppressing warnings or clang-tidy on alias
# targets is not portable, so we avoid modifying the target.
