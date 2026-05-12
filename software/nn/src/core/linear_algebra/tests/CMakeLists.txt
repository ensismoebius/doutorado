##
## linear_algebra unit tests (GoogleTest)
##
## Builds:
## - `linearAlgebra_gtest`: tests for legacy numeric/signal helpers.
##
add_executable(linearAlgebra_gtest
    linearAlgebra_gtest.cpp
)

target_link_libraries(linearAlgebra_gtest
    PRIVATE
        google_test
        linear_algebra
)

target_include_directories(linearAlgebra_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(linearAlgebra_gtest)