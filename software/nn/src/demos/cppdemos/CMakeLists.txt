add_executable(speaker_demo
    speaker_demo.cpp
    comandos.cpp
    codificacao.cpp
)

target_link_libraries(speaker_demo PRIVATE
    xtensor xtensor-blas
    tensor
    argparse
    wavelet
    waveCoreLib
    util
    linearAlgebra
)

configure_xtensor_parallel_target(speaker_demo)


add_library(rede_snn SHARED rede_snn.cpp)
set_target_properties(rede_snn PROPERTIES PREFIX "" OUTPUT_NAME "rede_snn")
target_link_libraries(rede_snn PRIVATE tensor xtensor xtensor-blas)
configure_xtensor_parallel_target(rede_snn)