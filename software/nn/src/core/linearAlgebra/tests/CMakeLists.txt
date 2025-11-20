add_executable(linearAlgebra_gtest
    linearAlgebra_gtest.cpp
)

target_link_libraries(linearAlgebra_gtest
    PRIVATE
        GTest::gtest_main
        linearAlgebra
)

target_include_directories(linearAlgebra_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(linearAlgebra_gtest)