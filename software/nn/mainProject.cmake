## AutoEncoderTargets.cmake
# Targets for mainProject

add_executable(mainProject
    ${CNPY_SOURCES}
    ${SRC_DIR}/initializers/xavier.hpp
    ${SRC_DIR}/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/optimizers/Adam.hpp
    ${SRC_DIR}/optimizers/SGD.hpp
    ${SRC_DIR}/layers/Sequential.hpp
    ${SRC_DIR}/layers/Linear.hpp
    ${SRC_DIR}/layers/Leaky.hpp
    ${SRC_DIR}/layers/ReLU.hpp
    ${SRC_DIR}/layers/LeakyReLU.hpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
    ${SRC_DIR}/dataLoaders/MatFile.cpp
    ${SRC_DIR}/dataLoaders/MatFileUtils.cpp
    ${SRC_DIR}/main.cpp
)

# Link libraries
target_link_libraries(mainProject
    PUBLIC
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(mainProject
    PUBLIC
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(mainProject)