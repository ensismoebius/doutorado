## AutoEncoderTargets.cmake
# Targets for mainProject

add_executable(mainProject
    ${SRC_DIR}/core/initializers/xavier.hpp
    ${SRC_DIR}/core/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/core/optimizers/Adam.hpp
    ${SRC_DIR}/core/optimizers/SGD.hpp
    ${SRC_DIR}/core/layers/Sequential.hpp
    ${SRC_DIR}/core/layers/Linear.hpp
    ${SRC_DIR}/core/layers/Leaky.hpp
    ${SRC_DIR}/core/layers/ReLU.hpp
    ${SRC_DIR}/core/layers/LeakyReLU.hpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/core/NnSaver.hpp
    ${SRC_DIR}/core/dataLoaders/MatFileUtils.cpp
    ${SRC_DIR}/main_app/main.cpp
)

# Link libraries
target_link_libraries(mainProject
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
        matioCpp
)

# Include directories
target_include_directories(mainProject
    PRIVATE
        ${SRC_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${cnpy_SOURCE_DIR} # Added back for NetworkSerializer.hpp
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(mainProject)