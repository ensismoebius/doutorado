##
## Tensor API unit tests (GoogleTest)
##
## Builds:
## - `tensor_gtest`: construction/indexing/shape behavior for `nn::Tensor`.
##
## Notes:
## - Uses `configure_xtensor_parallel_target` because the default backend is xtensor-based.
## - There is a commented-out mock-backend test sketch kept for future xtensor-free testing.
##
add_executable(tensor_gtest tensor_gtest.cpp)

target_link_libraries(tensor_gtest 
    PRIVATE 
        google_test 
         xtensor xtensor-blas 
        tensor
)

target_include_directories(tensor_gtest 
    PRIVATE 
         ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
nn_gtest_discover_tests(tensor_gtest)
configure_xtensor_parallel_target(tensor_gtest)

add_executable(tensor_all_backends_gtest tensor_all_backends_gtest.cpp)

target_link_libraries(tensor_all_backends_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
)

target_include_directories(tensor_all_backends_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

nn_gtest_discover_tests(tensor_all_backends_gtest)
configure_xtensor_parallel_target(tensor_all_backends_gtest)

# Device backend tests (skeleton implementation delegated to xtensor host mirror)
add_executable(device_backend_gtest device_backend_gtest.cpp)

target_link_libraries(device_backend_gtest 
    PRIVATE 
        google_test 
         xtensor xtensor-blas 
        tensor
)

target_include_directories(device_backend_gtest 
    PRIVATE 
         ${CMAKE_SOURCE_DIR}/src
)

nn_gtest_discover_tests(device_backend_gtest)
configure_xtensor_parallel_target(device_backend_gtest)

add_executable(opencl_tensor_backend_gtest opencl_tensor_backend_gtest.cpp)

target_link_libraries(opencl_tensor_backend_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
)

target_include_directories(opencl_tensor_backend_gtest
    PRIVATE
         ${CMAKE_SOURCE_DIR}/src
)

nn_gtest_discover_tests(opencl_tensor_backend_gtest)
configure_xtensor_parallel_target(opencl_tensor_backend_gtest)

add_executable(tensor_backend_switchability_gtest tensor_backend_switchability_gtest.cpp)

target_link_libraries(tensor_backend_switchability_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
)

target_include_directories(tensor_backend_switchability_gtest
    PRIVATE
         ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(tensor_backend_switchability_gtest
    PRIVATE
        NN_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

nn_gtest_discover_tests(tensor_backend_switchability_gtest)
configure_xtensor_parallel_target(tensor_backend_switchability_gtest)

add_executable(gpu_buffer_pool_gtest gpu_buffer_pool_gtest.cpp)

target_link_libraries(gpu_buffer_pool_gtest
    PRIVATE
        google_test
        tensor
)

target_include_directories(gpu_buffer_pool_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

nn_gtest_discover_tests(gpu_buffer_pool_gtest)

add_executable(tensor_perf_bench tensor_perf_bench.cpp)

target_link_libraries(tensor_perf_bench
    PRIVATE
        tensor
        xtensor xtensor-blas
)

target_include_directories(tensor_perf_bench
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

configure_xtensor_parallel_target(tensor_perf_bench)

# Mock-backend focused tests (xtensor-free math, but Tensor header still requires xtensor include)
# add_executable(tensor_mock_gtest 
#     mock_backend_gtest.cpp
#     MockTensorBackend.cpp
# )
# 
# target_link_libraries(tensor_mock_gtest
#     PRIVATE
#         google_test
#         xtensor xtensor-blas
#         tensor
# )
# 
# target_include_directories(tensor_mock_gtest
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/src
# )
# 
# gtest_discover_tests(tensor_mock_gtest)
# XTensor vs OpenCL numerical parity (ops, layers, SNN-AE chain). Skips
# gracefully when no OpenCL device is present.
add_executable(backend_parity_gtest backend_parity_gtest.cpp)

target_link_libraries(backend_parity_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
)

target_include_directories(backend_parity_gtest
    PRIVATE
         ${CMAKE_SOURCE_DIR}/src
)

nn_gtest_discover_tests(backend_parity_gtest)
configure_xtensor_parallel_target(backend_parity_gtest)

# SYCL parity suite: only meaningful (and only linkable) when the tensor lib
# was built with the SYCL kernels TU, i.e. under NN_BACKEND=SYCL.
if(NN_BACKEND STREQUAL "SYCL")
    add_executable(sycl_backend_parity_gtest sycl_backend_parity_gtest.cpp)

    target_link_libraries(sycl_backend_parity_gtest
        PRIVATE
            google_test
            xtensor xtensor-blas
            tensor
    )

    target_include_directories(sycl_backend_parity_gtest
        PRIVATE
             ${CMAKE_SOURCE_DIR}/src
    )

    nn_gtest_discover_tests(sycl_backend_parity_gtest FORCE_GPU_LOCK)
    configure_xtensor_parallel_target(sycl_backend_parity_gtest)
endif()

##
## pytorch_parity_gtest: ground-truth parity vs PyTorch/snnTorch, for EVERY
## concrete tensor backend side-by-side (XTensor, OpenCL, Device, and SYCL
## when NN_BACKEND=SYCL). Loads committed fixtures (fixtures/pytorch_refs.npz,
## generated by scripts/testing/gen_pytorch_refs.py) and compares
## forward/backward results per backend via TYPED_TEST.
##
add_executable(pytorch_parity_gtest
    pytorch_parity_gtest.cpp
)

target_link_libraries(pytorch_parity_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        layers
        tensor
        cnpy-static
        z
)

target_include_directories(pytorch_parity_gtest
    PRIVATE
        ${cnpy_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(pytorch_parity_gtest PROPERTIES CXX_CLANG_TIDY "")

nn_gtest_discover_tests(pytorch_parity_gtest FORCE_GPU_LOCK)
configure_xtensor_parallel_target(pytorch_parity_gtest)
