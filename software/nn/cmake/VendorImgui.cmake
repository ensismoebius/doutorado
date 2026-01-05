message(STATUS "Configuring ImGui...")
include(FetchContent)

# IMGui configuration
FetchContent_Declare(
    imgui
    URL            https://github.com/ocornut/imgui/archive/refs/tags/v1.88.zip
    URL_HASH       SHA256=b70d61578a9360d7d67c71a87d655819d118a742c53828d9640f94d8d369479e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(imgui)

# Add ImGui source files
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
)

# Add ImGui backend source files
set(IMGUI_BACKEND_SOURCES
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# Create ImGui library target
add_library(imgui 
    STATIC 
        ${IMGUI_SOURCES} 
        ${IMGUI_BACKEND_SOURCES}
)
target_compile_options(imgui PRIVATE -w)
target_include_directories(imgui 
    SYSTEM PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)