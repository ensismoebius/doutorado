add_library(optimizers INTERFACE)

target_include_directories(optimizers 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/optimizers
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)