##
## SNN spike plotter unit tests (GoogleTest) — Lif layer only (no OpenGL/GLFW/ImGui)
##
if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

add_executable(snn_spike_plotter_gtest
    snn_spike_plotter_gtest.cpp
)

target_link_libraries(snn_spike_plotter_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
        util
)

target_include_directories(snn_spike_plotter_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${LIB_DIR}/util
        ${OpenMP_INCLUDE_DIRS}
)

configure_xtensor_parallel_target(snn_spike_plotter_gtest)

include(GoogleTest)
nn_gtest_discover_tests(snn_spike_plotter_gtest)
