add_executable(resnet_demo
        resnet_demo.cpp
  )

  target_link_libraries(resnet_demo PRIVATE
    Eigen3::Eigen
    matioCpp
    dataLoaders
    util
    tensor
  )

  configure_eigen_parallel_target(resnet_demo)
