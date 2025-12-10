add_executable(fftw3_demo
        fftw3_demo.cpp
  )

  target_link_libraries(fftw3_demo PRIVATE
    Eigen3::Eigen
    matioCpp
    dataLoaders
    util
    tensor
    FFTW::FFTW
  )

  configure_eigen_parallel_target(fftw3_demo)
