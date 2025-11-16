enable_testing()

add_library(layers INTERFACE)

target_include_directories(layers 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/layers
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)
