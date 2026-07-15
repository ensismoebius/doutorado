add_executable(experiment_03_autoencoder_redesign_gtest
    AutoencoderRedesign_gtest.cpp
    ProfileAndResults_gtest.cpp
    DatasetBuilder_gtest.cpp
    LSTMAutoencoder_gtest.cpp
)

target_include_directories(experiment_03_autoencoder_redesign_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(experiment_03_autoencoder_redesign_gtest
    PRIVATE
        google_test
        experiment03_lib
    tensor
    layers
        matioCpp
        data_loaders
        util
        xtensor xtensor-blas
)

include(GoogleTest)
nn_gtest_discover_tests(experiment_03_autoencoder_redesign_gtest)
configure_xtensor_parallel_target(experiment_03_autoencoder_redesign_gtest)
