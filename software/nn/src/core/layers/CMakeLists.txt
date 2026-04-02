enable_testing()

# Layers library - mostly header-only, with Conv2d temporarily disabled
add_library(layers STATIC
    Conv2d_impl.cpp
    Conv2d_utils.cpp
)

target_include_directories(layers 
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/layers
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

target_link_libraries(layers
    PUBLIC Eigen3::Eigen
    PUBLIC tensor
)

configure_eigen_parallel_target(layers)

nn_enable_target_pch(layers)

add_subdirectory(tests)

