##
## SNN speaker demo unit tests (GoogleTest) — codificacao functions
##
if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

add_executable(snn_speaker_demo_gtest
    snn_speaker_demo_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../codificacao.cpp
)

target_link_libraries(snn_speaker_demo_gtest
    PRIVATE
        google_test
        xtensor xtensor-blas
        tensor
        util
)

target_include_directories(snn_speaker_demo_gtest
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${LIB_DIR}/util
        ${OpenMP_INCLUDE_DIRS}
)

configure_xtensor_parallel_target(snn_speaker_demo_gtest)

include(GoogleTest)
nn_gtest_discover_tests(snn_speaker_demo_gtest)
