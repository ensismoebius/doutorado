cmake_minimum_required(VERSION 3.16)

add_library(nn_progress STATIC
    ProgressManager.cpp
)

target_include_directories(nn_progress
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(nn_progress
    PUBLIC
        Threads::Threads
)

# Tests
enable_testing()
add_executable(progress_gtest tests/progress_gtest.cpp)
target_link_libraries(progress_gtest
    PRIVATE
        nn_progress
        GTest::gtest_main
        GTest::gtest
)
add_test(NAME progress_gtest COMMAND progress_gtest)


target_include_directories(nn_progress
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(nn_progress
    PUBLIC
    Threads::Threads
)
