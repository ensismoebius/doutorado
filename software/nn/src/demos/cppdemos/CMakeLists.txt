add_executable(speaker_demo
    speaker_demo.cpp
)

target_link_libraries(speaker_demo PRIVATE
    Eigen3::Eigen
    tensor
    wavelet
    waveCoreLib
    linearAlgebra
)

configure_eigen_parallel_target(speaker_demo)
