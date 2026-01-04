enable_testing()

# Layers library - mostly header-only, with Conv2d temporarily disabled
add_library(layers STATIC
    dummy.cpp  # Placeholder for header-only library
)

target_include_directories(layers 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/layers
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

target_link_libraries(layers
    PUBLIC Eigen3::Eigen
    PUBLIC tensor
)

add_subdirectory(tests)

