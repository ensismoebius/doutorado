add_library(tensor STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/OpenCLContext.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/CLBuffer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/DeviceMemory.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/KernelManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/OpenCLProfiling.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/GPUBufferPool.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/OpenCLTensorBackend.cpp
)

target_include_directories(tensor PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(tensor PUBLIC
	Eigen3::Eigen
)

# Link OpenCL if available
if(OpenCL_FOUND)
	target_link_libraries(tensor PUBLIC ${OpenCL_LIBRARIES})
	target_include_directories(tensor PRIVATE ${OpenCL_INCLUDE_DIRS})
endif()

add_subdirectory(tests)