add_library(utility 
    STATIC
        comparison.cpp
)

target_include_directories(utility
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(${SRC_DIR}/tests)
