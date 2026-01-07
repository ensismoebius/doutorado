add_library(tensor STATIC
    Tensor.cpp
    ${CMAKE_SOURCE_DIR}/include/nn/tensor/Tensor.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/tensor/EigenTensorBackend.hpp
    EigenTensorBackend.cpp
)

set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(tensor
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/tensor
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

target_link_libraries(tensor
    PRIVATE Eigen3::Eigen
)

configure_eigen_parallel_target(tensor)

add_subdirectory(tests)