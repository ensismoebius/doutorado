add_executable(speaker_demo
    speaker_demo.cpp
    comandos.cpp
    codificacao.cpp
)

target_link_libraries(speaker_demo PRIVATE
    Eigen3::Eigen
    tensor
    argparse
    wavelet
    waveCoreLib
    util
    linearAlgebra
)

configure_eigen_parallel_target(speaker_demo)


add_library(rede_snn SHARED rede_snn.cpp)
set_target_properties(rede_snn PROPERTIES PREFIX "" OUTPUT_NAME "rede_snn")
target_link_libraries(rede_snn PRIVATE tensor Eigen3::Eigen)
configure_eigen_parallel_target(rede_snn)