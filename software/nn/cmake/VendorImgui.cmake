message(STATUS "Configuring ImGui...")
include(FetchContent)

# IMGui configuration
FetchContent_Declare(
    imgui
    URL            https://github.com/ocornut/imgui/archive/refs/tags/v1.88.zip
    URL_HASH       SHA256=81087a74599e5890a07b636887cee73a7dc1a9eb9e1f19a4a0d82a76090bf4c2
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

# Suppress warnings and disable clang-tidy for ImGui
target_compile_options(imgui PRIVATE -w)
set_target_properties(imgui PROPERTIES CXX_CLANG_TIDY "")

target_include_directories(imgui 
    SYSTEM PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)