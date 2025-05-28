
add_subdirectory(${CNPY_DIR})

# Add subdirectories for all testable modules
add_subdirectory(${SRC_DIR}/util)
add_subdirectory(${SRC_DIR}/layers)
add_subdirectory(${SRC_DIR}/tensor)
add_subdirectory(${SRC_DIR}/optimizers)
add_subdirectory(${SRC_DIR}/initializers)

# Add executable target
add_executable(nn 
    ${CNPY_SOURCES}
    ${SRC_DIR}/initializers/xavier.hpp
    ${SRC_DIR}/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/optimizers/Adam.hpp
    ${SRC_DIR}/optimizers/SGD.hpp
    ${SRC_DIR}/layers/Sequential.hpp
    ${SRC_DIR}/layers/Linear.hpp
    ${SRC_DIR}/layers/Leaky.hpp
    ${SRC_DIR}/layers/ReLU.hpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
    ${SRC_DIR}/main.cpp
)

# Link libraries
target_link_libraries(nn
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(nn
    PRIVATE
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)