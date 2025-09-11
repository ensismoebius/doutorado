# IMGui configuration
set(IMGUI_DIR "${LIB_DIR}/imgui")

# Check if imgui is already downloaded and compiled
if(NOT EXISTS "${IMGUI_DIR}/imgui.h")
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

# Add ImGui source files
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)

# Add ImGui backend source files
set(IMGUI_BACKEND_SOURCES
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)