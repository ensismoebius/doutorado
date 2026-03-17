## exec_experiment04.cmake
# Target for experiment04 utility

add_library(experiment03_lib STATIC
    ${SRC_DIR}/experiments/03/lib/src/cli.cpp
    ${SRC_DIR}/experiments/03/lib/src/progress.cpp
    ${SRC_DIR}/experiments/03/lib/src/dataset_info.cpp
    ${SRC_DIR}/experiments/03/lib/src/experiment03.cpp
)

target_include_directories(experiment03_lib
    PUBLIC
        ${SRC_DIR}/experiments/03/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(experiment03_lib
    PUBLIC
        CLI11::CLI11
        dataLoaders_10_1117
        dataLoaders
        matioCpp
        z
    Eigen3::Eigen
)

add_executable(experiment03
    ${SRC_DIR}/experiments/03/experiment03.cpp
)

# Link libraries
target_link_libraries(experiment03
    PRIVATE
    experiment03_lib
    Eigen3::Eigen
    cnpy-static
    dataLoaders_10_1117
    dataLoaders
    matioCpp
    z
)