add_executable(voice_biometrics_cpp
    main.cpp
    ../cppdemos/codificacao.cpp
)

target_include_directories(voice_biometrics_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../cppdemos
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(voice_biometrics_cpp PRIVATE
    Eigen3::Eigen
    tensor
    layers
    argparse
    wavelet
    waveCoreLib
    util
    linearAlgebra
)

configure_eigen_parallel_target(voice_biometrics_cpp)
