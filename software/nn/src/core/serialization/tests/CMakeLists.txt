add_executable(serialization_gtest
    StateIO_gtest.cpp
)

target_include_directories(serialization_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(serialization_gtest
    PRIVATE
        google_test
        Eigen3::Eigen
        tensor
)

include(GoogleTest)
gtest_discover_tests(serialization_gtest)
