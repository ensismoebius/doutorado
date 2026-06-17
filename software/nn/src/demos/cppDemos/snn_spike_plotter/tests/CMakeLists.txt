##
## SNN spike plotter unit tests (GoogleTest) — Lif layer only (no OpenGL/GLFW/ImGui)
##
find_package(GTest REQUIRED)

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
gtest_discover_tests(snn_spike_plotter_gtest)
