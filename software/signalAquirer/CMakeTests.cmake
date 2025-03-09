add_executable(tests 
    tests/main.cpp
)
target_include_directories(tests
    PRIVATE
        ${GTEST_DIR}/googletest/include
        ${GTEST_DIR}/googlemock/include
)

target_link_libraries(tests 
    PRIVATE
        gtest
        gmock
)


# Register the test
add_test(NAME Test01 COMMAND tests)