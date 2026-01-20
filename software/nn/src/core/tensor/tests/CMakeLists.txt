##
## Tensor API unit tests (GoogleTest)
##
## Builds:
## - `tensor_gtest`: construction/indexing/shape behavior for `nn::Tensor`.
##
## Notes:
## - Uses `configure_eigen_parallel_target` because the default backend is Eigen-based.
## - There is a commented-out mock-backend test sketch kept for future Eigen-free testing.
##
add_executable(tensor_gtest tensor_gtest.cpp)

target_link_libraries(tensor_gtest 
    PRIVATE 
        google_test 
        Eigen3::Eigen 
        tensor
)

target_include_directories(tensor_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
        ${EIGEN3_INCLUDE_DIR}
)

include(GoogleTest)
gtest_discover_tests(tensor_gtest)
configure_eigen_parallel_target(tensor_gtest)

# Device backend tests (skeleton implementation delegated to Eigen host mirror)
add_executable(device_backend_gtest device_backend_gtest.cpp)

target_link_libraries(device_backend_gtest 
    PRIVATE 
        google_test 
        Eigen3::Eigen 
        tensor
)

target_include_directories(device_backend_gtest 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
        ${EIGEN3_INCLUDE_DIR}
)

gtest_discover_tests(device_backend_gtest)
configure_eigen_parallel_target(device_backend_gtest)

# Mock-backend focused tests (Eigen-free math, but Tensor header still requires Eigen include)
# add_executable(tensor_mock_gtest 
#     mock_backend_gtest.cpp
#     MockTensorBackend.cpp
# )
# 
# target_link_libraries(tensor_mock_gtest
#     PRIVATE
#         google_test
#         Eigen3::Eigen
#         tensor
# )
# 
# target_include_directories(tensor_mock_gtest
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/src
# )
# 
# gtest_discover_tests(tensor_mock_gtest)