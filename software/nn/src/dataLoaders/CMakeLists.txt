find_package(ZLIB REQUIRED)

add_executable(mat_file_gtest mat_file_gtest.cpp MatFile.cpp MatFile.h)

target_link_libraries(mat_file_gtest PRIVATE GTest::gtest_main ${ZLIB_LIBRARIES})

target_include_directories(mat_file_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(mat_file_gtest)

# small demo executable to run the data loader + spiking autoencoder
add_executable(dataLoader_demo
	${CMAKE_SOURCE_DIR}/src/dataLoaderTest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/DataLoader.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/MatFile.cpp
)
target_include_directories(dataLoader_demo PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(dataLoader_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
target_link_libraries(dataLoader_demo PRIVATE ${ZLIB_LIBRARIES} Eigen3::Eigen)
configure_eigen_parallel_target(dataLoader_demo)

# Additional DataLoader unit tests
add_executable(dataLoader_more_gtest
	${CMAKE_SOURCE_DIR}/src/dataLoaders/dataLoader_gtest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/dataLoader_more_gtest.cpp
	${CMAKE_SOURCE_DIR}/src/dataLoaders/DataLoader.cpp
)
target_include_directories(dataLoader_more_gtest PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dataLoader_more_gtest PRIVATE GTest::gtest_main Eigen3::Eigen ${ZLIB_LIBRARIES})
configure_eigen_parallel_target(dataLoader_more_gtest)
gtest_discover_tests(dataLoader_more_gtest)
