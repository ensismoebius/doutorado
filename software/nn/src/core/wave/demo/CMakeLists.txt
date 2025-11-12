add_library(waveLibstatic
    STATIC
    ../Wav.cpp
    ../filtersOperations.cpp
    ../simpleSignalOperations.cpp
    ../../linearAlgebra/linearAlgebra.cpp
)

target_include_directories(waveLibstatic
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(create_wav_demo 
    create_wav_demo.cpp
)
target_link_libraries(create_wav_demo 
    PRIVATE
        waveLibstatic
)

