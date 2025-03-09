# Add executable target
add_executable(SignalAcquirer 
    ${IMGUI_BACKEND_SOURCES}
    ${IMPLOT_SOURCES}
    ${IMGUI_SOURCES}
    ${LIB_DIR}/util/imguiGlfw.cpp
    ${LIB_DIR}/util/CapturerManager.cpp
    ${LIB_DIR}/util/AudioCapture.cpp
    ${LIB_DIR}/util/MockCapturer.cpp
    ${SRC_DIR}/main.cpp
)

# Link libraries
target_link_libraries(SignalAcquirer
    PRIVATE
        ${SDL2_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        raylib
        glfw
        ${ASOUND_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${PORTAUDIO_LIBRARIES}
)

# Include directories
target_include_directories(SignalAcquirer
    PRIVATE
        ${SRC_DIR}
        ${SDL2_INCLUDE_DIRS}
        ${ARMADILLO_INCLUDE_DIRS}
        ${OpenMP_INCLUDE_DIRS}
        ${ASOUND_INCLUDE_DIRS}
        ${IMGUI_DIR}
        ${IMPLOT_DIR}
        ${IMGUI_DIR}/backends
        ${OPENGL_INCLUDE_DIRS}
        ${PORTAUDIO_INCLUDE_DIRS}
        ${LIB_DIR}/util
)