# -----------------------------------------------------------------------------
# BackendImplementationGuard.cmake
#
# Enforces architecture rule:
# concrete backend implementation names must not be referenced outside
# include/nn/Backend.hpp and backend implementation directories.
#
# On violation, generates a C++ source with a compile-time #error using the
# exact required message.
# -----------------------------------------------------------------------------

set(_NN_BACKEND_GUARD_MESSAGE
    "Backend implementation must only be refereced inside include/nn/Backend.hpp !")

set(_NN_BACKEND_IMPL_TOKENS
    "XTensorBackend"
    "OpenCLTensorBackend"
    "DeviceTensorBackend")

file(GLOB_RECURSE _NN_BACKEND_GUARD_CANDIDATES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/*.c"
    "${CMAKE_SOURCE_DIR}/include/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.h"
    "${CMAKE_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_SOURCE_DIR}/src/*.c"
    "${CMAKE_SOURCE_DIR}/src/*.cpp")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_NN_BACKEND_GUARD_CANDIDATES})

set(_NN_BACKEND_GUARD_VIOLATIONS "")
foreach(_NN_FILE IN LISTS _NN_BACKEND_GUARD_CANDIDATES)
    if(_NN_FILE STREQUAL "${CMAKE_SOURCE_DIR}/include/nn/Backend.hpp")
        continue()
    endif()

    # Allowed backend implementation/self-reference zones.
    if(_NN_FILE MATCHES "/include/nn/tensor/"
       OR _NN_FILE MATCHES "/src/core/tensor/")
        continue()
    endif()

    file(READ "${_NN_FILE}" _NN_CONTENT)
    foreach(_NN_TOKEN IN LISTS _NN_BACKEND_IMPL_TOKENS)
        string(REGEX MATCH "(^|[^A-Za-z0-9_])${_NN_TOKEN}([^A-Za-z0-9_]|$)" _NN_HIT "${_NN_CONTENT}")
        if(_NN_HIT)
            file(RELATIVE_PATH _NN_REL_PATH "${CMAKE_SOURCE_DIR}" "${_NN_FILE}")
            list(APPEND _NN_BACKEND_GUARD_VIOLATIONS "${_NN_REL_PATH}: token ${_NN_TOKEN}")
        endif()
    endforeach()
endforeach()

set(NN_BACKEND_REFERENCE_GUARD_SOURCE
    "${CMAKE_BINARY_DIR}/generated/backend_reference_guard.cpp")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")

if(_NN_BACKEND_GUARD_VIOLATIONS)
    string(JOIN "\n// - " _NN_VIOLATION_LINES ${_NN_BACKEND_GUARD_VIOLATIONS})
    file(WRITE "${NN_BACKEND_REFERENCE_GUARD_SOURCE}"
"#error \"${_NN_BACKEND_GUARD_MESSAGE}\"\n// Violations:\n// - ${_NN_VIOLATION_LINES}\n")
else()
    file(WRITE "${NN_BACKEND_REFERENCE_GUARD_SOURCE}"
"namespace nn::build_guard { int backend_reference_guard_ok = 0; }\n")
endif()
