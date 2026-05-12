## PlotTarget.cmake
# Target for plotSpikingNetwork

add_executable(plotSpikingNetwork
    ${CNPY_SOURCES}
    ${CMAKE_SOURCE_DIR}/include/initializers/xavier.hpp
    ${CMAKE_SOURCE_DIR}/include/optimizers/SGDMinimal.hpp
    ${CMAKE_SOURCE_DIR}/include/optimizers/Adam.hpp
    ${CMAKE_SOURCE_DIR}/include/optimizers/SGD.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/base/Sequential.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/dense/Linear.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/spiking/Lif.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/activations/ReLU.hpp
    ${CMAKE_SOURCE_DIR}/include/layers/activations/LeakyReLU.hpp
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
        xtensor xtensor-blas
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
        ${OpenMP_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIRS}
)

# Configure xtensor parallelism for the target
configure_xtensor_parallel_target(plotSpikingNetwork)
