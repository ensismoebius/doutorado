add_library(optimizers INTERFACE)

target_include_directories(optimizers 
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(tests)
