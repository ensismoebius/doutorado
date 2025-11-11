## PlotTarget.cmake
# Target for plotSpikingNetwork

# ImGui and ImPlot sources from Imgui.cmake and Implot.cmake
set(IMGUI_DIR "${LIB_DIR}/imgui")
set(IMPLOT_DIR "${LIB_DIR}/implot")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
set(IMPLOT_SOURCES
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
)

add_executable(plotSpikingNetwork
    ${CNPY_SOURCES}
    ${SRC_DIR}/initializers/xavier.hpp
    ${SRC_DIR}/optimizers/SGDMinimal.hpp
    ${SRC_DIR}/optimizers/Adam.hpp
    ${SRC_DIR}/optimizers/SGD.hpp
    ${SRC_DIR}/layers/Sequential.hpp
    ${SRC_DIR}/layers/Linear.hpp
    ${SRC_DIR}/layers/Leaky.hpp
    ${SRC_DIR}/layers/ReLU.hpp
    ${SRC_DIR}/layers/LeakyReLU.hpp
    ${SRC_DIR}/util/synthetic_spike_data.cpp
    ${SRC_DIR}/util/vectorizationCheck.cpp
    ${SRC_DIR}/util/batching.cpp
    ${SRC_DIR}/util/NnSaver.hpp
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
    ${SRC_DIR}/util/imguiGlfw.cpp
    ${SRC_DIR}/experiments/plotSpikingNetwork.cpp
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
)

# Include directories
target_include_directories(plotSpikingNetwork
    PRIVATE
        ${SRC_DIR}
        ${CNPY_DIR}
        ${LIB_DIR}/util
        ${EIGEN3_INCLUDE_DIR}
        ${OpenMP_INCLUDE_DIRS}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
        ${GLFW_INCLUDE_DIRS}
)

# Configure Eigen parallelism for the target
configure_eigen_parallel_target(plotSpikingNetwork)
