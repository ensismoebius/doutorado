include(GoogleTest)

add_executable(statistics_gtest 
    statistics_gtest.cpp
)

target_link_libraries(statistics_gtest
    PRIVATE
        statistics
        GTest::gtest_main
)

add_executable(confusionMatrix_gtest 
    confusionMatrix_gtest.cpp
)

target_link_libraries(confusionMatrix_gtest
    PRIVATE
        statistics
        GTest::gtest_main
)
gtest_discover_tests(statistics_gtest)
gtest_discover_tests(confusionMatrix_gtest)