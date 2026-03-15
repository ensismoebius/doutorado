## exec_experiment04.cmake
# Target for experiment04 utility

add_library(experiment04_lib STATIC
    ${SRC_DIR}/experiments/03/lib/src/cli.cpp
    ${SRC_DIR}/experiments/03/lib/src/batch_util.cpp
)

target_include_directories(experiment04_lib
    PUBLIC
        ${SRC_DIR}/experiments/03/lib/include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(experiment04_lib
    PUBLIC
        CLI11::CLI11
        dataLoaders_10_1117
        dataLoaders
        matioCpp
        z
    Eigen3::Eigen
)

add_executable(experiment04
    ${SRC_DIR}/experiments/03/experiment04.cpp
)

# Link libraries
target_link_libraries(experiment04
    PRIVATE
    experiment04_lib
    Eigen3::Eigen
    cnpy-static
    dataLoaders_10_1117
    dataLoaders
    matioCpp
    z
)