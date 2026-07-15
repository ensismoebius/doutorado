##
## Paraconsistent logic unit tests (GoogleTest)
##
## Builds:
## - `paraconsistent_gtest`: tests for certainty/contradiction computations.
##
include(GoogleTest)

add_executable(paraconsistent_gtest
    paraconsistent_gtest.cpp
)

target_link_libraries(paraconsistent_gtest
    PRIVATE
        paraconsistent
        util
        google_test
)

nn_gtest_discover_tests(paraconsistent_gtest)