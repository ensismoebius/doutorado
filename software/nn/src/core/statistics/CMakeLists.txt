include(GoogleTest)

add_library(statistics
    statistics.cpp
    confusionMatrix.cpp
    multiClassMetrics.cpp
)

target_include_directories(statistics 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(statistics
    PUBLIC
        Eigen3::Eigen
        tensor
)