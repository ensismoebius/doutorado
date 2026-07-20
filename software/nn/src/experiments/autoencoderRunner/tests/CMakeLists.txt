add_executable(autoencoderRunner_redesign_gtest
    AutoencoderRedesign_gtest.cpp
    ProfileAndResults_gtest.cpp
    DatasetBuilder_gtest.cpp
    LSTMAutoencoder_gtest.cpp
)

target_include_directories(autoencoderRunner_redesign_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(autoencoderRunner_redesign_gtest
    PRIVATE
        google_test
        autoencoderRunner_lib
    tensor
    layers
        matioCpp
        data_loaders
        util
        xtensor xtensor-blas
)

include(GoogleTest)
nn_gtest_discover_tests(autoencoderRunner_redesign_gtest)
configure_xtensor_parallel_target(autoencoderRunner_redesign_gtest)
