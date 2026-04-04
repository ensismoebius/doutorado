##
## windowing_gtest — unit tests for WindowSpec, WindowingEngine and windowed datasets.
##
## Covers:
##   - WindowSpec::hop_size(), num_windows(), validate()
##   - WindowingEngine::compute_windows() (boundary cases + EEG/Audio schema sizes)
##
## No MAT file fixtures are required for these tests.
##
add_executable(windowing_gtest
    windowing_gtest.cpp
    ../utils/MockImaginedSpeechDatasetGenerator.cpp
)

target_include_directories(windowing_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(windowing_gtest
    PRIVATE
        google_test
        dataLoaders_10_1117_windowing
        tensor
        Eigen3::Eigen
)

configure_eigen_parallel_target(windowing_gtest)

include(GoogleTest)
gtest_discover_tests(windowing_gtest)
