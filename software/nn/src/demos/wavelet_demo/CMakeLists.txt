add_executable(wavelet_demo
        wavelet_demo.cpp
  )

  target_link_libraries(wavelet_demo PRIVATE
    xtensor xtensor-blas
    matioCpp
    dataLoaders
    util
    tensor
    wavelet
    linearAlgebra # Added missing dependency
    saver
    MatplotlibCpp::MatplotlibCpp
  )

  configure_xtensor_parallel_target(wavelet_demo)
