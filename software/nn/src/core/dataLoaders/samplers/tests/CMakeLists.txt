add_executable(dataLoaders_samplers_gtest
    sampler_gtest.cpp
)

target_include_directories(dataLoaders_samplers_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dataLoaders_samplers_gtest
    PRIVATE
        google_test
        dataLoaders_samplers
)

include(GoogleTest)
gtest_discover_tests(dataLoaders_samplers_gtest)
