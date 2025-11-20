## PlotTarget.cmake
# Target for plotSpikingNetwork

add_executable(plotSpikingNetwork
    ${CNPY_SOURCES}
    ${SRC_DIR}/core/initializers/xavier.hpp
    ${SRC_DIR}/core/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/core/optimizers/Adam.hpp
    ${SRC_DIR}/core/optimizers/SGD.hpp
    ${SRC_DIR}/core/layers/Sequential.hpp
    ${SRC_DIR}/core/layers/Linear.hpp
    ${SRC_DIR}/core/layers/Leaky.hpp
    ${SRC_DIR}/core/layers/ReLU.hpp
    ${SRC_DIR}/core/layers/LeakyReLU.hpp
    ${SRC_DIR}/core/utility/imguiGlfw.cpp
    ${SRC_DIR}/demos/plotSpikingNetwork/plotSpikingNetwork.cpp
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
