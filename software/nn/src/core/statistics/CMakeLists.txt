include(GoogleTest)

add_library(statistics
    statistics.cpp
    confusionMatrix.cpp
)

target_include_directories(statistics 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)