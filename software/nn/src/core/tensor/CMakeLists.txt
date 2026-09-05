add_library(tensor STATIC
	${NN_BACKEND_REFERENCE_GUARD_SOURCE}
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLContext.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/CLBuffer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/DeviceMemory.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/KernelManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLProfiling.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/GPUBufferPool.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackend.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendInplace.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendElementwise.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendReductions.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendUnaryLif.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendMatmul.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendTransposeBlock.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendCompare.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/opencl/OpenCLTensorBackendRuntime.cpp
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

# When OpenCL is the active nn::Backend, refuse to configure without a real
# device present rather than silently building a program whose tensor ops
# will all throw at first use — see cmake/OpenCLGpuCapabilityCheck.cmake.
if(NN_BACKEND STREQUAL "OpenCL")
	include(${CMAKE_SOURCE_DIR}/cmake/OpenCLGpuCapabilityCheck.cmake)
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

	# Refuses to configure on known-unsupported GPU hardware, or when no GPU
	# is present at all — see cmake/SyclGpuCapabilityCheck.cmake for why.
	include(${CMAKE_SOURCE_DIR}/cmake/SyclGpuCapabilityCheck.cmake)

	# ThinLTO's final-link backend codegen for this TU goes through
	# AdaptiveCpp's own bundled LLVM (independent of our -march=native), whose
	# host-feature table doesn't recognize '-amx-transpose'/'-avx10.1-256'/
	# '-avx10.2-256' — confirmed by reproducing the warnings even with
	# -march=native removed entirely from project flags, so they aren't
	# caused by anything this project passes. Excluding just this one TU from
	# LTO sidesteps that codegen path entirely (confirmed: 0 occurrences vs.
	# 24 with LTO enabled, parity tests still pass) at the cost of losing
	# cross-TU inlining for this thin device-dispatch wrapper only — the hot
	# math it delegates to (XTensorBackend, SYCL device kernels) is
	# unaffected.
	set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/sycl/SYCLTensorBackend.cpp
		PROPERTIES COMPILE_OPTIONS "-fno-lto")

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