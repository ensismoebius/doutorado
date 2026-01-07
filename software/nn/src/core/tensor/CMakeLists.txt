add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE
	${CMAKE_SOURCE_DIR}/include
	${EIGEN3_INCLUDE_DIR}
)

add_subdirectory(tests)