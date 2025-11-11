# VendorMatio.cmake
# Configure vendored lib/matio presence and make MATIO::MATIO available

# Disable vendored matio's own tests by default (safe for most builds)
set(MATIO_BUILD_TESTING OFF CACHE BOOL "Disable building tests in vendored matio" FORCE)

include(lib/matio.cmake)

# Provide a lightweight imported "matio" target if the subproject doesn't
# create it early enough. This helps export/install steps in other vendored
# CMakeLists succeed.
if(TARGET matio)
  add_library(MATIO::MATIO ALIAS matio)
endif()
