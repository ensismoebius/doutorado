add_executable(fsdd_loader_gtest
    fsdd_loader_gtest.cpp
)

target_include_directories(fsdd_loader_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(fsdd_loader_gtest
    PRIVATE
        google_test
        data_loaders_fsdd
        tensor
)

include(GoogleTest)
nn_gtest_discover_tests(fsdd_loader_gtest)
