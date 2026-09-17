add_library(tensor STATIC
	${NN_BACKEND_REFERENCE_GUARD_SOURCE}
	${CMAKE_CURRENT_SOURCE_DIR}/DeviceRuntime.cpp
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

add_subdirectory(tests)
