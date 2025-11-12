include(GoogleTest)

add_library(statistics
    statistics.cpp
    confusionMatrix.cpp
)

target_include_directories(statistics 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(statistics_gtest 
    tests/statistics_gtest.cpp
)

target_link_libraries(statistics_gtest
    PRIVATE
    statistics
    gtest
    gtest_main
)

add_executable(confusionMatrix_gtest 
    tests/confusionMatrix_gtest.cpp
)

target_link_libraries(confusionMatrix_gtest
    PRIVATE
    statistics
    gtest
    gtest_main
)

target_link_libraries(confusionMatrix_gtest 
    PRIVATE 
        statistics
        GTest::gtest_main
)

gtest_discover_tests(statistics_gtest)
gtest_discover_tests(confusionMatrix_gtest)

# Add tests to CTest
gtest_discover_tests(statistics_gtest)
gtest_discover_tests(confusionMatrix_gtest)