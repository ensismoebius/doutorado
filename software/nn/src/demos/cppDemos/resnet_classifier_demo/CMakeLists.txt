add_executable(resnet_demo
        resnet_demo.cpp
  )

  target_link_libraries(resnet_demo PRIVATE
    xtensor xtensor-blas
    matioCpp
    data_loaders
    util
    tensor
  )

  configure_xtensor_parallel_target(resnet_demo)

add_subdirectory(tests)
