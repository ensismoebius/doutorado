## exec_loadingData.cmake
# Target for loadingData utility

add_executable(loadingData
    ${SRC_DIR}/util/loadingData.cpp
)

# Link libraries
target_link_libraries(loadingData
    PUBLIC
        matioCpp
)

# Include directories
target_include_directories(loadingData
    PUBLIC
        ${SRC_DIR}/util
)
