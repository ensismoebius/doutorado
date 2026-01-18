add_library(initializers INTERFACE)
target_include_directories(initializers 
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(tests)
