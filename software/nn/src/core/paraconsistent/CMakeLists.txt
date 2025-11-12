include(GoogleTest)

add_library(paraconsistent
    paraconsistent.cpp
)

target_include_directories(paraconsistent
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(paraconsistent
    PUBLIC
        linearAlgebra
        utility
)

add_executable(paraconsistent_gtest
    tests/paraconsistent_gtest.cpp
)

target_link_libraries(paraconsistent_gtest
    PRIVATE
        paraconsistent
        GTest::gtest_main
)

gtest_discover_tests(paraconsistent_gtest)