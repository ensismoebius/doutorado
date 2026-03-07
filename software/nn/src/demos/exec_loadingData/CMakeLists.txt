## exec_loadingData.cmake
# Target for loadingData utility

add_executable(loadingData
    ${SRC_DIR}/demos/exec_loadingData/lib/src/cli.cpp
    ${SRC_DIR}/demos/exec_loadingData/loadingData.cpp
)

# Link libraries
target_link_libraries(loadingData
    PRIVATE
        matioCpp
)

target_link_libraries(loadingData PRIVATE
    CLI11::CLI11
    Eigen3::Eigen
    cnpy-static
    dataLoaders_10_1117
    dataLoaders
    matioCpp
    z
)