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
