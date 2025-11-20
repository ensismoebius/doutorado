add_library(filtersOperationsLibStatic
    STATIC
        filtersOperations.cpp
        simpleSignalOperations.cpp
        Wav.cpp
)

target_include_directories(filtersOperationsLibStatic
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(${SRC_DIR}/tests)
