add_executable(loadingData ${CMAKE_SOURCE_DIR}/src/core/utility/loadingData.cpp)
target_link_libraries(loadingData PRIVATE
    dataLoaders
    Eigen3::Eigen
    matioCpp
    cnpy-static
    z
)