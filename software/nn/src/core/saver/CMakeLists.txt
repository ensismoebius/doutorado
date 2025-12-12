
add_library(saver INTERFACE)

target_include_directories(saver
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR} 
        ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(tests)
