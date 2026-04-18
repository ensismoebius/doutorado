## PlotTarget.cmake
# Target for plotSpikingNetwork

add_executable(plotSpikingNetwork
    ${CNPY_SOURCES}
    ${CMAKE_SOURCE_DIR}/include/nn/initializers/xavier.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/optimizers/SGDMinimal.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/optimizers/Adam.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/optimizers/SGD.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/base/Sequential.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/dense/Linear.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/spiking/Leaky.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/activations/ReLU.hpp
    ${CMAKE_SOURCE_DIR}/include/nn/layers/activations/LeakyReLU.hpp
    ${SRC_DIR}/core/utility/imgui_glfw.cpp
    ${SRC_DIR}/demos/exec_plotSpikingNetwork/plotSpikingNetwork.cpp
)

# Link libraries
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

target_link_libraries(plotSpikingNetwork
    PRIVATE
        cnpy
        Eigen3::Eigen
        ${OpenMP_CXX_LIBRARIES}
        OpenGL::GL
        ${GLFW_LIBRARIES}
        imgui
        implot
        util
        tensor
)

# Include directories
target_include_directories(plotSpikingNetwork
    PRIVATE
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIRS}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(plotSpikingNetwork)
