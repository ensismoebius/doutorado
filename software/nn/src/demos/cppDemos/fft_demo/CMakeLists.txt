add_executable(fftw3_demo
        fftw3_demo.cpp
  )

  target_link_libraries(fftw3_demo PRIVATE
    xtensor xtensor-blas
    matioCpp
    data_loaders
    util
    tensor
    FFTW::FFTW
    MatplotlibCpp::MatplotlibCpp
  )

  configure_xtensor_parallel_target(fftw3_demo)

add_subdirectory(tests)
