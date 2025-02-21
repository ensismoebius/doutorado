cmake_minimum_required(VERSION 3.10)
project(SignalAcquirer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Generate compile_commands.json for code completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define paths
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

# Raylib configuration
set(RAYLIB_DIR "${LIB_DIR}/raylib")
set(RAYLIB_BUILD_DIR "${RAYLIB_DIR}/build")

# Lsllib configuration
set(LSLLIB_DIR "${LIB_DIR}/lsllib")
set(LSLLIB_BUILD_DIR "${LSLLIB_DIR}/build")

# IMGui configuration
set(IMGUI_DIR "${LIB_DIR}/imgui")

# IMPlot configuration
set(IMPLOT_DIR "${LIB_DIR}/implot")

# Add Armadillo and OpenMP dependencies
find_package(Armadillo REQUIRED)
find_package(OpenMP REQUIRED)

# Add OpenGL dependencies
find_package(OpenGL REQUIRED)

# Package finder
find_package(PkgConfig REQUIRED)

# Find ALSA library
pkg_check_modules(ASOUND REQUIRED alsa)

# Find GLFW package
pkg_check_modules(GLFW REQUIRED glfw3)

# Find PortAudio
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Find SDL2
find_package(SDL2 REQUIRED)

# Check if lsllib is already downloaded and compiled
if(NOT EXISTS "${LSLLIB_BUILD_DIR}/liblsl.so" OR NOT EXISTS "${LSLLIB_DIR}/include/lsl_cpp.h")
    message(STATUS "Lab Streaming Layer not found, downloading and compiling...")

    if(NOT EXISTS "${LSLLIB_DIR}/.git")
        # Clone lsllib repository if not already cloned
        execute_process(
            COMMAND git clone --depth 1 https://github.com/sccn/liblsl.git ${LSLLIB_DIR}
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to clone Lab Streaming Layer library repository")
        endif()
    else()
        # Update lsllib repository if already cloned
        execute_process(
            COMMAND git -C ${LSLLIB_DIR} pull
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to update Lab Streaming Layer repository")
        endif()
    endif()

    # Create build directory and compile lsllib with -fPIC
    file(MAKE_DIRECTORY ${LSLLIB_BUILD_DIR})
    execute_process(
        COMMAND cmake -S ${LSLLIB_DIR} -B ${LSLLIB_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "CMake configuration for Lab Streaming Layer library failed")
    endif()

    execute_process(
        COMMAND cmake --build ${LSLLIB_BUILD_DIR} --parallel
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Lab Streaming Layer build failed")
    endif()
else()
    message(STATUS "Lab Streaming Layer already exists, skipping download and compilation...")
endif()

# Define lsllib library
add_library(lsllib SHARED IMPORTED)
set_target_properties(lsllib PROPERTIES
    IMPORTED_LOCATION "${LSLLIB_BUILD_DIR}/liblsl.so"
    INTERFACE_INCLUDE_DIRECTORIES "${LSLLIB_DIR}/include"
)

# Check if raylib is already downloaded and compiled
if(NOT EXISTS "${RAYLIB_BUILD_DIR}/raylib/libraylib.a" OR NOT EXISTS "${RAYLIB_DIR}/src/raylib.h")
    message(STATUS "Raylib not found, downloading and compiling...")

    if(NOT EXISTS "${RAYLIB_DIR}/.git")
        # Clone raylib repository if not already cloned
        execute_process(
            COMMAND git clone --depth 1 https://github.com/raysan5/raylib ${RAYLIB_DIR}
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to clone raylib repository")
        endif()
    else()
        # Update raylib repository if already cloned
        execute_process(
            COMMAND git -C ${RAYLIB_DIR} pull
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to update raylib repository")
        endif()
    endif()

    # Create build directory and compile raylib with -fPIC
    file(MAKE_DIRECTORY ${RAYLIB_BUILD_DIR})
    execute_process(
        COMMAND cmake -S ${RAYLIB_DIR} -B ${RAYLIB_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "CMake configuration for raylib failed")
    endif()

    execute_process(
        COMMAND cmake --build ${RAYLIB_BUILD_DIR} --parallel
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Raylib build failed")
    endif()
else()
    message(STATUS "Raylib already exists, skipping download and compilation...")
endif()

# Define raylib library
add_library(raylib STATIC IMPORTED)
set_target_properties(raylib PROPERTIES
    IMPORTED_LOCATION "${RAYLIB_BUILD_DIR}/raylib/libraylib.a"
    INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_DIR}/src"
)

# Check if imgui is already downloaded and compiled
if(NOT EXISTS "${IMGUI_DIR}/src/imgui.h")
    message(STATUS "ImGui not found, downloading and compiling...")

    if(NOT EXISTS "${IMGUI_DIR}/.git")
        # Clone imgui repository if not already cloned
        execute_process(
            COMMAND git clone --depth 1 https://github.com/ocornut/imgui.git ${IMGUI_DIR}
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to clone ImGui repository")
        endif()
    else()
        # Update imgui repository if already cloned
        execute_process(
            COMMAND git -C ${IMGUI_DIR} pull
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to update ImGui repository")
        endif()
    endif()
else()
    message(STATUS "ImGui already exists, skipping download and compilation...")
endif()

# Check if ImPlot is already downloaded and compiled
if(NOT EXISTS "${IMPLOT_DIR}/implot.h")
    message(STATUS "ImPlot not found, downloading and compiling...")

    if(NOT EXISTS "${IMPLOT_DIR}/.git")
        # Clone ImPlot repository if not already cloned
        execute_process(
            COMMAND git clone --depth 1 --recursive https://github.com/epezent/implot.git ${IMPLOT_DIR}
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to clone ImPlot repository")
        endif()
    else()
        # Update ImPlot repository if already cloned
        execute_process(
            COMMAND git -C ${IMPLOT_DIR} pull
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to update ImPlot repository")
        endif()
    endif()
else()
    message(STATUS "ImPlot already exists, skipping download and compilation...")
endif()

# Add ImGui source files
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)

# Add ImPlot source files
set(IMPLOT_SOURCES
    ${IMPLOT_DIR}/implot_items.cpp
    ${IMPLOT_DIR}/implot.cpp
)

# Add ImGui backend source files
set(IMGUI_BACKEND_SOURCES
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Add executable target
add_executable(SignalAcquirer 
    ${IMGUI_BACKEND_SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
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
        lsllib
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
        ${LSLLIB_DIR}/include
)

# Copy the shared library to the executable's directory
add_custom_command(TARGET SignalAcquirer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${LSLLIB_BUILD_DIR}/liblsl.so
        $<TARGET_FILE_DIR:SignalAcquirer>
)

# Print configuration for debugging
message(STATUS "ASOUND_INCLUDE_DIRS: ${ASOUND_INCLUDE_DIRS}")
message(STATUS "ASOUND_LIBRARIES: ${ASOUND_LIBRARIES}")
message(STATUS "SDL2_INCLUDE_DIRS: ${SDL2_INCLUDE_DIRS}")
message(STATUS "SDL2_LIBRARIES: ${SDL2_LIBRARIES}")
message(STATUS "LSLLIB_INCLUDE_DIRS: ${LSLLIB_DIR}/include")
message(STATUS "LSLLIB_LIBRARY: ${LSLLIB_BUILD_DIR}/liblsl.so")