add_executable(wavelet_demo
        wavelet_demo.cpp
  )

  target_link_libraries(wavelet_demo PRIVATE
    xtensor xtensor-blas
    matioCpp
    data_loaders
    util
    tensor
    wavelet
    linear_algebra # Added missing dependency
    serialization
    MatplotlibCpp::MatplotlibCpp
  )

  configure_xtensor_parallel_target(wavelet_demo)
