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
gtest_discover_tests(tensor_gtest)
configure_xtensor_parallel_target(tensor_gtest)

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

gtest_discover_tests(device_backend_gtest)
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

gtest_discover_tests(opencl_tensor_backend_gtest)
configure_xtensor_parallel_target(opencl_tensor_backend_gtest)

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

gtest_discover_tests(gpu_buffer_pool_gtest)

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