cmake_minimum_required(VERSION 3.10)
project(SignalAcquirer VERSION 1.0 LANGUAGES CXX)

# Sets up the flags for CMake and compiler
include(CMakeFlags.cmake)

# Define paths
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

# Do the required libraries check
include(CMakePackageChecking.cmake)

# Enable testing
enable_testing()

# Loads an compiles the required libraries
include(CMakeGTestGMock.cmake)
include(CMakeRaylib.cmake)
include(CMakeImplot.cmake)
include(CMakeImgui.cmake)

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

# Compiles unity tests
include(CMakeTests.cmake)


# Copy the shared library to the executable's directory
# add_custom_command(TARGET SignalAcquirer POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy
#         ${LSLLIB_BUILD_DIR}/liblsl.so
#         $<TARGET_FILE_DIR:SignalAcquirer>
# )

# Print configuration for debugging
message(STATUS "ASOUND_INCLUDE_DIRS: ${ASOUND_INCLUDE_DIRS}")
message(STATUS "ASOUND_LIBRARIES: ${ASOUND_LIBRARIES}")
message(STATUS "SDL2_INCLUDE_DIRS: ${SDL2_INCLUDE_DIRS}")
message(STATUS "SDL2_LIBRARIES: ${SDL2_LIBRARIES}")