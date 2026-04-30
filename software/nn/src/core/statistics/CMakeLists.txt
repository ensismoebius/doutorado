include(GoogleTest)

add_library(statistics
    statistics.cpp
    inference_tests.cpp
    confusion_matrix.cpp
    multi_class_metrics.cpp
    kfold.cpp
)

target_include_directories(statistics 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(statistics
    PUBLIC
        xtensor xtensor-blas
        tensor
)

add_subdirectory(tests)