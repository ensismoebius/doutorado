add_executable(mat_file_gtest mat_file_gtest.cpp mat_file.cpp mat_file.h)

target_link_libraries(mat_file_gtest PRIVATE GTest::gtest_main)

target_include_directories(mat_file_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(mat_file_gtest)
