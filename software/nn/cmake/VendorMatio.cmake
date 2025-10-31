# VendorMatio.cmake
# Configure vendored lib/matio presence and make MATIO::MATIO available

# Disable vendored matio's own tests by default (safe for most builds)
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
  if(TARGET matio AND NOT TARGET MATIO::MATIO)
    add_library(MATIO::MATIO ALIAS matio)
  elseif(NOT TARGET MATIO::MATIO)
    add_library(MATIO::MATIO UNKNOWN IMPORTED)
    set_target_properties(MATIO::MATIO PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/lib/matio/src"
      IMPORTED_LOCATION "${MATIO_LIBRARY}")
  endif()
endif()

# Provide a lightweight imported "matio" target if the subproject doesn't
# create it early enough. This helps export/install steps in other vendored
# CMakeLists succeed.
if(EXISTS "${CMAKE_SOURCE_DIR}/lib/matio" AND NOT TARGET matio)
  add_library(matio UNKNOWN IMPORTED)
  if(EXISTS "${CMAKE_BINARY_DIR}/lib/libmatio.so")
    set_target_properties(matio PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/lib/libmatio.so")
  elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libmatio.a")
    set_target_properties(matio PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/lib/libmatio.a")
  endif()
  set_target_properties(matio PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/lib/matio/src")
endif()
