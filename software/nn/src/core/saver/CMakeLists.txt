add_library(saver INTERFACE)

target_include_directories(saver
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(tests)
