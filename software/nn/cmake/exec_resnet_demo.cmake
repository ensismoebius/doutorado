if (EXISTS ${CMAKE_SOURCE_DIR}/src/resnet_demo.cpp)
  add_executable(resnet_demo
    ${CMAKE_SOURCE_DIR}/src/resnet_demo.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/dataLoaders/MatFileUtils.cpp
  )

  target_link_libraries(resnet_demo PRIVATE
    Eigen3::Eigen
    ${OpenMP_CXX_LIBRARIES}
    matioCpp
  )

  target_include_directories(resnet_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR} ${LIB_DIR}/matio-cpp/include ${LIB_DIR}/matio/include)
  configure_eigen_parallel_target(resnet_demo)
endif()
