##
## 10.1117 dataset loader unit tests (GoogleTest)
##
## Builds:
## - `dataLoaders_10_1117_gtest`: tests for EEG/Audio MAT loaders following the
##   10.1117-style dataset schema.
##
## Notes:
## - Links against `dataLoaders_10_1117` and uses `MatTestUtilsLib` fixtures.
##
add_executable(dataLoaders_10_1117_gtest
	audio_loader_gtest.cpp
    eeg_loader_gtest.cpp
	eeg_loader_edge_gtest.cpp
)

target_include_directories(dataLoaders_10_1117_gtest 
	PRIVATE 
		${CMAKE_SOURCE_DIR}/src 
		${CMAKE_CURRENT_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/src/core/dataLoaders/tests
)

target_link_libraries(dataLoaders_10_1117_gtest 
	PRIVATE 
		google_test 
		Eigen3::Eigen 
		${ZLIB_LIBRARIES} 
		matioCpp
        dataLoaders_10_1117 # Link to the new library
        MatTestUtilsLib
)

configure_eigen_parallel_target(dataLoaders_10_1117_gtest)
include(GoogleTest)
gtest_discover_tests(dataLoaders_10_1117_gtest)