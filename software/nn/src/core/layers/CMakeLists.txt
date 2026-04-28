enable_testing()

# Layers library - mostly header-only, with Conv2d temporarily disabled
add_library(layers STATIC
    convolution/Conv1d_impl.cpp
    convolution/Conv2d_impl.cpp
    convolution/Conv2d_utils.cpp
)

target_include_directories(layers
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} # For category-organized layer implementation files
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

target_link_libraries(layers
    PUBLIC Eigen3::Eigen
    PUBLIC tensor
)

# Avoid LTO-related unresolved template member symbols from split Conv2d TUs.
set_property(TARGET layers PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
target_compile_options(layers PRIVATE -fno-lto)
target_link_options(layers PRIVATE -fno-lto)

configure_eigen_parallel_target(layers)

nn_enable_target_pch(layers)

add_subdirectory(tests)

