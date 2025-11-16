add_executable(paraconsistent_gtest
    paraconsistent_gtest.cpp
)

target_link_libraries(paraconsistent_gtest
    PRIVATE
        paraconsistent
        utility
        GTest::gtest_main
)

gtest_discover_tests(paraconsistent_gtest)