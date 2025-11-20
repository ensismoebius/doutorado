add_library(tensor INTERFACE)
target_include_directories(tensor 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/tensor
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

add_subdirectory(${SRC_DIR}/tests)
