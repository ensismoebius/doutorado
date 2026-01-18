add_executable(wavelet_demo
        wavelet_demo.cpp
  )

  target_link_libraries(wavelet_demo PRIVATE
    Eigen3::Eigen
    matioCpp
    dataLoaders
    util
    tensor
    wavelet
    linearAlgebra # Added missing dependency
    saver
    MatplotlibCpp::MatplotlibCpp
  )

  configure_eigen_parallel_target(wavelet_demo)
