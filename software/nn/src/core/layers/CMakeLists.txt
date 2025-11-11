# CMake configuration for GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(layers_gtest
    tests/layers_gtest.cpp
)

target_link_libraries(layers_gtest PRIVATE GTest::gtest_main Eigen3::Eigen)

target_include_directories(
  layers_gtest PRIVATE ${SRC_DIR} ${EIGEN3_INCLUDE_DIR}
                       ${CMAKE_SOURCE_DIR}/lib/cnpy)

include(GoogleTest)
gtest_discover_tests(layers_gtest)

configure_eigen_parallel_target(layers_gtest)
