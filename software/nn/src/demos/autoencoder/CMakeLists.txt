add_executable(autoencoder_scaffold_example
    ../autoencoder/autoencoder_scaffold_example.cpp
)

# Link libraries
target_link_libraries(autoencoder_scaffold_example
    PRIVATE
        experiment03_lib
        Eigen3::Eigen
)

# Include directories
target_include_directories(autoencoder_scaffold_example
    PRIVATE
        ${SRC_DIR}
        ${SRC_DIR}/experiments/03/lib/autoencoder/include
        ${SRC_DIR}/experiments/03/lib/include
        ${PROJECT_SOURCE_DIR}/include
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(autoencoder_scaffold_example)
