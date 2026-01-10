##
## linearAlgebra unit tests (GoogleTest)
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
        linearAlgebra
)

target_include_directories(linearAlgebra_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(linearAlgebra_gtest)