enable_testing()

add_library(layers STATIC
    Conv2d_impl.cpp
    Conv2d_utils.cpp
)

target_include_directories(layers 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/layers
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

target_link_libraries(layers
    PRIVATE Eigen3::Eigen
    PUBLIC tensor
)

add_subdirectory(tests)
