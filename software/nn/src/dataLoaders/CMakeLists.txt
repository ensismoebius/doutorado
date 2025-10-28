find_package(ZLIB REQUIRED)

## Consolidated GoogleTest target for dataLoaders
add_executable(dataLoaders_gtest
	${CMAKE_SOURCE_DIR}/src/dataLoaders/mat_file_gtest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/matfile_utils_gtest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/dataLoader_gtest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/dataLoader_more_gtest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/MatFile.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/MatFileUtils.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/DataLoader.cpp
)
target_include_directories(dataLoaders_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dataLoaders_gtest PRIVATE GTest::gtest_main Eigen3::Eigen ${ZLIB_LIBRARIES})
configure_eigen_parallel_target(dataLoaders_gtest)
include(GoogleTest)
gtest_discover_tests(dataLoaders_gtest)

# small demo executable to run the data loader + spiking autoencoder
add_executable(dataLoader_demo
	${CMAKE_SOURCE_DIR}/src/dataLoaderTest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/DataLoader.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/MatFile.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/MatFileUtils.cpp
)
target_include_directories(dataLoader_demo PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(dataLoader_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
target_link_libraries(dataLoader_demo PRIVATE ${ZLIB_LIBRARIES} Eigen3::Eigen)
configure_eigen_parallel_target(dataLoader_demo)

# Additional DataLoader unit tests
## (previously had a separate target for more DataLoader tests — now consolidated)
