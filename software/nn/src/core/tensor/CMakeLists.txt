add_library(tensor STATIC
	${NN_BACKEND_REFERENCE_GUARD_SOURCE}
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

target_compile_definitions(tensor PUBLIC XTENSOR_USE_XSIMD)
target_link_libraries(tensor PUBLIC xsimd)

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

# SYCL backend: the kernels TU needs a SYCL implementation (AdaptiveCpp or
# oneAPI DPC++). Only compiled when the SYCL backend is selected so the other
# presets never require the toolchain.
if(NN_BACKEND STREQUAL "SYCL")
	# AdaptiveCpp installs an acpp-config.cmake providing add_sycl_to_target();
	# oneAPI provides IntelSYCL. Try AdaptiveCpp first (primary target on this
	# project), then IntelSYCL, then fail with an actionable message.
	find_package(AdaptiveCpp CONFIG QUIET)
	if(NOT AdaptiveCpp_FOUND)
		find_package(IntelSYCL CONFIG QUIET)
	endif()
	if(NOT AdaptiveCpp_FOUND AND NOT IntelSYCL_FOUND)
		message(FATAL_ERROR
			"NN_BACKEND=SYCL requires a SYCL implementation.\n"
			"Install AdaptiveCpp (Arch: yay -S adaptivecpp) or oneAPI DPC++, "
			"or pick another backend preset.")
	endif()

	target_sources(tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sycl/SYCLTensorBackend.cpp)
	add_sycl_to_target(TARGET tensor
		SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/sycl/SYCLTensorBackend.cpp)

	# add_sycl_to_target links the SYCL runtime PRIVATE; re-export it so that
	# executables linking the static tensor lib resolve acpp runtime symbols.
	if(TARGET AdaptiveCpp::acpp-rt)
		target_link_libraries(tensor PUBLIC AdaptiveCpp::acpp-rt)
	elseif(TARGET hipSYCL::hipSYCL-rt)
		target_link_libraries(tensor PUBLIC hipSYCL::hipSYCL-rt)
	endif()
endif()

add_subdirectory(tests)