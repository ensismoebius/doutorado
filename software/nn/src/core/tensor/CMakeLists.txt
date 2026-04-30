add_library(tensor STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLContext.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/CLBuffer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/DeviceMemory.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/KernelManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLProfiling.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/GPUBufferPool.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackend.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/DeviceRuntime.cpp
)

target_include_directories(tensor PUBLIC
	${CMAKE_SOURCE_DIR}/include)

target_include_directories(tensor SYSTEM PUBLIC
	$<TARGET_PROPERTY:xtensor,INTERFACE_INCLUDE_DIRECTORIES>
	$<TARGET_PROPERTY:xtensor-blas,INTERFACE_INCLUDE_DIRECTORIES>
	$<TARGET_PROPERTY:xtl,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(tensor PUBLIC
	xtensor xtensor-blas
)

# Link OpenCL if available
if(OpenCL_FOUND)
	target_link_libraries(tensor PUBLIC ${OpenCL_LIBRARIES})
	target_include_directories(tensor PRIVATE ${OpenCL_INCLUDE_DIRS})
	target_compile_definitions(tensor PUBLIC CL_TARGET_OPENCL_VERSION=200)
endif()

add_subdirectory(tests)