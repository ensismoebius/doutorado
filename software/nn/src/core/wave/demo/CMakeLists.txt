add_executable(create_wav_demo 
    create_wav_demo.cpp
)
target_link_libraries(create_wav_demo 
    PRIVATE
        filtersOperationsLibStatic
        linear_algebra
)

target_include_directories(create_wav_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

