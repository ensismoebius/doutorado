# VendorMatio.cmake
# Configure vendored lib/matio presence and make MATIO::MATIO available

# Disable vendored matio's own tests by default (safe for most builds)
set(MATIO_BUILD_TESTING OFF CACHE BOOL "Disable building tests in vendored matio" FORCE)

include(FetchContent)

FetchContent_Declare(
    matio
    GIT_REPOSITORY https://github.com/tbeu/matio.git
    GIT_TAG        HEAD # Consider using a specific commit hash or tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(matio)

# Provide a lightweight imported "matio" target if the subproject doesn't
# create it early enough. This helps export/install steps in other vendored
# CMakeLists succeed.
if(TARGET matio)
  add_library(MATIO::MATIO ALIAS matio)
endif()
