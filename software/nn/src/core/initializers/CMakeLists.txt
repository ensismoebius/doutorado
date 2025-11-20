add_library(initializers INTERFACE)
target_include_directories(initializers 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/initializers
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

add_subdirectory(tests)
