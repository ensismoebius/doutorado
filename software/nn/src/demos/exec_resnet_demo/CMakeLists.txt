add_executable(resnet_demo
        resnet_demo.cpp
  )

  target_link_libraries(resnet_demo PRIVATE
    Eigen3::Eigen
    ${OpenMP_CXX_LIBRARIES}
    matioCpp
    dataLoaders
    util
  )

  target_include_directories(resnet_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
  configure_eigen_parallel_target(resnet_demo)
