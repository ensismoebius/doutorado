add_library(
    utility STATIC
    comparison.cpp
)

target_include_directories(
    utility
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
