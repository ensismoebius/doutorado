# Lightweight FindMATIO.cmake
# This module prefers an in-tree "matio" target (created by add_subdirectory(lib/matio))
# or the vendored build output under ${CMAKE_BINARY_DIR}/lib/libmatio.so.
# It sets MATIO_INCLUDE_DIR, MATIO_LIBRARY and creates an imported target MATIO::MATIO
# when possible so downstream find_package(MATIO) callers succeed.

if(TARGET matio AND NOT TARGET MATIO::MATIO)
  add_library(MATIO::MATIO ALIAS matio)
  get_target_property(_inc TARGET matio INTERFACE_INCLUDE_DIRECTORIES)
  if(_inc)
    set(MATIO_INCLUDE_DIR "${_inc}")
  endif()
  # No MATIO_LIBRARY - linking against the target is preferred.
  set(MATIO_FOUND TRUE)
  return()
endif()

# If a vendored include dir exists, use it.
if(NOT MATIO_INCLUDE_DIR)
  if(EXISTS "${CMAKE_SOURCE_DIR}/lib/matio/src")
    set(MATIO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/matio/src")
  endif()
endif()

# If a library was built in the current build tree, prefer it.
if(NOT MATIO_LIBRARY)
  if(EXISTS "${CMAKE_BINARY_DIR}/lib/libmatio.so")
    set(MATIO_LIBRARY "${CMAKE_BINARY_DIR}/lib/libmatio.so")
  elseif(EXISTS "${CMAKE_BINARY_DIR}/lib/libmatio.a")
    set(MATIO_LIBRARY "${CMAKE_BINARY_DIR}/lib/libmatio.a")
  endif()
endif()

if(MATIO_INCLUDE_DIR AND MATIO_LIBRARY)
  set(MATIO_FOUND TRUE)
  # Create IMPORTED target so consumers can link to MATIO::MATIO
  if(NOT TARGET MATIO::MATIO)
    add_library(MATIO::MATIO UNKNOWN IMPORTED)
    set_target_properties(MATIO::MATIO PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MATIO_INCLUDE_DIR}"
      IMPORTED_LOCATION "${MATIO_LIBRARY}")
  endif()
endif()

mark_as_advanced(MATIO_INCLUDE_DIR MATIO_LIBRARY)

if(NOT MATIO_FOUND)
  find_path(MATIO_INCLUDE_DIR matio.h PATHS /usr/include /usr/local/include PATH_SUFFIXES matio)
  find_library(MATIO_LIBRARY NAMES matio PATHS /usr/lib /usr/local/lib)
  if(MATIO_INCLUDE_DIR AND MATIO_LIBRARY)
    set(MATIO_FOUND TRUE)
    add_library(MATIO::MATIO UNKNOWN IMPORTED)
    set_target_properties(MATIO::MATIO PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MATIO_INCLUDE_DIR}"
      IMPORTED_LOCATION "${MATIO_LIBRARY}")
  endif()
endif()

if(MATIO_FOUND)
  set(MATIO_INCLUDE_DIR "${MATIO_INCLUDE_DIR}" CACHE PATH "MatIO include dir" FORCE)
  set(MATIO_LIBRARY "${MATIO_LIBRARY}" CACHE FILEPATH "MatIO library file" FORCE)
endif()
