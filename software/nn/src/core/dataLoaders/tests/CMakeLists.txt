##
## DataLoader + MAT I/O unit tests (GoogleTest)
##
## Builds:
## - `dataLoaders_gtest`: consolidated test binary for DataLoader iteration,
##   dataset collation, and MAT-file utility conversions.
##
## Notes:
## - Links `dataLoaders` and `tensor` plus matio-cpp (`matioCpp`) and a small
##   internal helper lib (`MatTestUtilsLib`) for generating MAT fixtures.
##
## Consolidated GoogleTest target for dataLoaders
add_executable(dataLoaders_gtest
	dataLoader_more_gtest.cpp
	batch_prefetcher_gtest.cpp
	matfile_utils_gtest.cpp
	dataLoader_gtest.cpp
	sampler_option_resolution_gtest.cpp
	matfile_row_vector_gtest.cpp
	mat_file_dataset_gtest.cpp
	matfile_vector_gtest.cpp
	mat_file_gtest.cpp
	SqliteBatchSource_windowing_gtest.cpp
	state_io_gtest.cpp
    count_mat_rows_gtest.cpp
    dataset_collate_gtest.cpp
	# BatchPrefetcher tests removed (too slow for default runs)
)

target_include_directories(dataLoaders_gtest 
	PRIVATE 
		${CMAKE_SOURCE_DIR}/src 
		${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dataLoaders_gtest 
	PRIVATE 
		google_test 
		Eigen3::Eigen 
		${ZLIB_LIBRARIES} 
		matioCpp
        dataLoaders # Link to the new library
        MatTestUtilsLib
        tensor
)

configure_eigen_parallel_target(dataLoaders_gtest)
include(GoogleTest)
gtest_discover_tests(dataLoaders_gtest)

add_subdirectory(MatTestUtils)