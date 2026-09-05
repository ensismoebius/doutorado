##
## Characterization tests for the shared experiment configuration.
##
## `Config::load` is 193 lines of validation that had no tests: it could not be
## refactored, because there was nothing to refactor against.
##

add_executable(experiments_config_gtest
    experiments_config_gtest.cpp
)

target_link_libraries(experiments_config_gtest
    PRIVATE
        google_test
        experiments_config
        nlohmann_json_wrapper
)

nn_gtest_discover_tests(experiments_config_gtest)
