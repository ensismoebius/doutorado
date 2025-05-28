# CMake configuration for GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(layers_gtest
    ${SRC_DIR}/layers/layers_gtest.cpp
    ${SRC_DIR}/layers/Leaky.hpp
    ${SRC_DIR}/layers/Linear.hpp
    ${SRC_DIR}/layers/ReLU.hpp
    ${SRC_DIR}/layers/Sequential.hpp
    ${SRC_DIR}/layers/MSELoss.hpp
    ${SRC_DIR}/tensor/Tensor.hpp
    ${SRC_DIR}/util/batching.hpp
)

target_link_libraries(layers_gtest
    PRIVATE
        GTest::gtest_main
        Eigen3::Eigen
)

target_include_directories(layers_gtest
    PRIVATE
        ${SRC_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/lib/cnpy
)

include(GoogleTest)
gtest_discover_tests(layers_gtest)
