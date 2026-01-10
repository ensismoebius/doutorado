##
## Statistics unit tests (GoogleTest)
##
## Builds:
## - `statistics_gtest`: descriptive stats + multi-class metric helpers.
## - `confusionMatrix_gtest`: confusion-matrix derived binary metrics.
##
include(GoogleTest)

add_executable(statistics_gtest 
    statistics_gtest.cpp
)

target_link_libraries(statistics_gtest
    PRIVATE
        statistics
        google_test
)

add_executable(confusionMatrix_gtest 
    confusionMatrix_gtest.cpp
)

target_link_libraries(confusionMatrix_gtest
    PRIVATE
        statistics
        google_test
)
gtest_discover_tests(statistics_gtest)
gtest_discover_tests(confusionMatrix_gtest)