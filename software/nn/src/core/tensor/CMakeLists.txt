add_library(tensor STATIC
    Tensor.cpp
    ITensorBackend.hpp
    EigenTensorBackend.hpp
    EigenTensorBackend.cpp
    TensorBackendFactory.hpp
    TensorBackendFactory.cpp
)

set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} # For headers directly in src/core/tensor
        ${CMAKE_SOURCE_DIR}/src     # For other project headers
)

target_link_libraries(tensor
    PRIVATE Eigen3::Eigen
)

add_subdirectory(tests)