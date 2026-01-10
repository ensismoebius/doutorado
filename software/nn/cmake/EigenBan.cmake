##
## EigenBan.cmake
##
## Purpose
## - Enforce a “no accidental Eigen includes” rule for targets that are meant to stay
##   independent of Eigen.
##
## What it provides
## - Interface target: `nn_eigen_guard`.
## - Helper functions:
##   - `nn_disallow_eigen(<target>)`: links the guard, which forces pre-including `EigenBan.hpp`.
##   - `nn_allow_eigen(<target>)`: defines `NN_ALLOW_EIGEN=1` to opt out for that target.
##
## How it works
## - Uses the compiler's `-include <header>` mechanism to inject `cmake/EigenBan.hpp` before
##   any compilation unit. The header triggers a hard error if Eigen headers are seen unless
##   `NN_ALLOW_EIGEN` is defined.
##
## Rationale
## - Keeps the codebase honest about where Eigen is used (especially useful for portability
##   experiments and for isolating dependencies).
##

# Eigen ban support: provide an interface target that pre-includes EigenBan.hpp.
# Link nn_eigen_guard to any target that must not include Eigen.
# For targets that legitimately use Eigen (e.g., EigenTensorBackend), add the
# compile definition NN_ALLOW_EIGEN.

add_library(nn_eigen_guard INTERFACE)

set(_NN_EIGEN_BAN_HEADER "${CMAKE_SOURCE_DIR}/cmake/EigenBan.hpp")
if(NOT EXISTS "${_NN_EIGEN_BAN_HEADER}")
    message(FATAL_ERROR "EigenBan.hpp not found at ${_NN_EIGEN_BAN_HEADER}")
endif()

target_compile_options(nn_eigen_guard INTERFACE "-include" "${_NN_EIGEN_BAN_HEADER}")

target_compile_definitions(nn_eigen_guard INTERFACE NN_EIGEN_GUARD_ACTIVE=1)

function(nn_disallow_eigen target)
    target_link_libraries(${target} PRIVATE nn_eigen_guard)
endfunction()

function(nn_allow_eigen target)
    target_compile_definitions(${target} PRIVATE NN_ALLOW_EIGEN=1)
endfunction()
