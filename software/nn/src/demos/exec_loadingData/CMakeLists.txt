## exec_loadingData.cmake
# Target for loadingData utility

add_library(loadingData_lib STATIC
    ${SRC_DIR}/demos/exec_loadingData/lib/src/cli.cpp
    ${SRC_DIR}/demos/exec_loadingData/lib/src/SamplerOptionsFactory.cpp
    ${SRC_DIR}/demos/exec_loadingData/lib/src/batch_util.cpp
    ${SRC_DIR}/demos/exec_loadingData/lib/src/subject_discovery.cpp
    ${SRC_DIR}/demos/exec_loadingData/lib/src/BatchPrefetcher.cpp
    ${SRC_DIR}/demos/exec_loadingData/lib/src/SamplePacking.cpp
    ${SRC_DIR}/demos/exec_loadingData/lib/src/Protocol101117Dataset.cpp
    ${SRC_DIR}/demos/exec_loadingData/lib/src/SynchronizedBatchAssembler.cpp
)

target_include_directories(loadingData_lib
    PUBLIC
        ${SRC_DIR}/demos/exec_loadingData/lib/include
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
    ${SRC_DIR}/demos/exec_loadingData/loadingData.cpp
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