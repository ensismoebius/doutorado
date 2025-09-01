
add_subdirectory(${CNPY_DIR})

# Add subdirectories for all testable modules
add_subdirectory(${SRC_DIR}/util)
add_subdirectory(${SRC_DIR}/layers)
add_subdirectory(${SRC_DIR}/tensor)
add_subdirectory(${SRC_DIR}/optimizers)
add_subdirectory(${SRC_DIR}/initializers)

# -------------------------------------
# Add executable target for main_test01
# -------------------------------------
add_executable(main_test01 
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
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
    ${SRC_DIR}/main_test01.cpp
)

# Link libraries
target_link_libraries(main_test01
    PUBLIC
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(main_test01
    PUBLIC
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)

# -------------------------------------
# Add executable target for auto-encoder test
# -------------------------------------
add_executable(autoEncoderTest
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
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
    ${SRC_DIR}/autoEncoderTest.cpp
)

# Link libraries
target_link_libraries(autoEncoderTest
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(autoEncoderTest
    PRIVATE
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)

# -------------------------------------
# Add executable target for main_test03
# -------------------------------------
add_executable(main_test03
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
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
    ${SRC_DIR}/main_test03.cpp
)

# Link libraries
target_link_libraries(main_test03
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
)

# Include directories
target_include_directories(main_test03
    PRIVATE
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)