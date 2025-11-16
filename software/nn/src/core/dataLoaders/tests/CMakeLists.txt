## Consolidated GoogleTest target for dataLoaders
add_executable(dataLoaders_gtest
	dataLoader_more_gtest.cpp
	matfile_utils_gtest.cpp
	dataLoader_gtest.cpp
	matfile_row_vector_gtest.cpp
	mat_file_dataset_gtest.cpp
	matfile_vector_gtest.cpp
	mat_file_gtest.cpp
)

target_include_directories(dataLoaders_gtest 
	PRIVATE 
		${CMAKE_SOURCE_DIR}/src 
		${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dataLoaders_gtest 
	PRIVATE 
		GTest::gtest_main 
		Eigen3::Eigen 
		${ZLIB_LIBRARIES} 
		matioCpp
        dataLoaders # Link to the new library
        MatTestUtilsLib
)

configure_eigen_parallel_target(dataLoaders_gtest)
include(GoogleTest)
gtest_discover_tests(dataLoaders_gtest)

add_subdirectory(MatTestUtils)