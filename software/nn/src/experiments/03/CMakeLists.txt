## exec_loadingData.cmake
# Target for loadingData utility

add_library(loadingData_lib STATIC
    ${SRC_DIR}/experiments/03/lib/src/cli.cpp
    ${SRC_DIR}/experiments/03/lib/src/SamplerOptionsFactory.cpp
    ${SRC_DIR}/experiments/03/lib/src/batch_util.cpp
)

target_include_directories(loadingData_lib
    PUBLIC
        ${SRC_DIR}/experiments/03/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(loadingData_lib
    PUBLIC
        CLI11::CLI11
        dataLoaders_10_1117
        dataLoaders
        matioCpp
        z
    Eigen3::Eigen
)

add_executable(loadingData
    ${SRC_DIR}/experiments/03/experiment04.cpp
)

# Link libraries
target_link_libraries(loadingData
    PRIVATE
    loadingData_lib
    Eigen3::Eigen
    cnpy-static
    dataLoaders_10_1117
    dataLoaders
    matioCpp
    z
)