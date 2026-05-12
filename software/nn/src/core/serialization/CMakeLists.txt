add_library(serialization INTERFACE)

target_include_directories(serialization
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(tests)
