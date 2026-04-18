add_executable(experiment_04_lstm_gtest
    LSTMAutoencoder_gtest.cpp
)

target_include_directories(experiment_04_lstm_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(experiment_04_lstm_gtest
    PRIVATE
        google_test
        experiment04_lib
        tensor
        layers
        util
        Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(experiment_04_lstm_gtest)

if(TARGET configure_eigen_parallel_target)
    configure_eigen_parallel_target(experiment_04_lstm_gtest)
endif()
