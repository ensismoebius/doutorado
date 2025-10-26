## LoadingDataTarget.cmake
# Target for matio data loader test

add_executable(loadingData
    ${SRC_DIR}/dataLoaders/DataLoader.cpp
    ${SRC_DIR}/dataLoaders/MatFile.cpp
    ${SRC_DIR}/loadingData.cpp
)

find_package(ZLIB REQUIRED)

# Link libraries
target_link_libraries(loadingData
    PRIVATE
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
        ${ZLIB_LIBRARIES}
)

# Include directories
target_include_directories(loadingData
    PRIVATE
        ${SRC_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(loadingData)
