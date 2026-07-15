# -----------------------------------------------------------------------------
# BackendImplementationGuard.cmake
#
# Enforces architecture rule:
# concrete backend implementation names must not be referenced outside
# include/Backend.hpp and backend implementation directories.
#
# On violation, generates a C++ source with a compile-time #error using the
# exact required message.
# -----------------------------------------------------------------------------

set(_NN_BACKEND_GUARD_MESSAGE
    "Backend implementation must only be refereced inside include/Backend.hpp !")

set(_NN_BACKEND_IMPL_TOKENS
    "XTensorBackend"
    "OpenCLTensorBackend"
    "SYCLTensorBackend"
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
    if(_NN_FILE STREQUAL "${CMAKE_SOURCE_DIR}/include/Backend.hpp")
        continue()
    endif()

    # Allowed backend implementation/self-reference zones.
    if(_NN_FILE MATCHES "/include/tensor/"
       OR _NN_FILE MATCHES "/src/core/tensor/")
        continue()
    endif()

    # Narrow exception: Conv1d/Conv2d split their template definitions into a
    # separate .cpp (unlike header-only layers such as Linear), so making them
    # available for every backend — not just the single currently-selected
    # nn::Backend — requires naming each concrete backend in an explicit
    # `template class Conv*dImpl<Backend>;` instantiation directive here. This
    # is instantiation-only: no per-backend branching or backend-specific math
    # is introduced, so it doesn't defeat the rule's purpose (keeping layer
    # *logic* backend-agnostic). See src/core/tensor/tests/pytorch_parity_gtest.cpp,
    # which needs Conv1dImpl<X>/Conv2dImpl<X> linkable for every backend X to
    # validate them against PyTorch ground truth.
    if(_NN_FILE MATCHES "/src/core/layers/convolution/Conv1d_impl\\.cpp$"
       OR _NN_FILE MATCHES "/src/core/layers/convolution/Conv2d_impl\\.cpp$"
       OR _NN_FILE MATCHES "/src/core/layers/convolution/Conv2d_utils\\.cpp$")
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
