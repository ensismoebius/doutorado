include(FetchContent)

# IMPlot configuration
FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG        v0.14 # Pinned to a specific tag for reproducibility
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(implot)

# Add ImPlot source files
set(IMPLOT_SOURCES
    ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot_SOURCE_DIR}/implot.cpp
)

# Create ImPlot library target
add_library(implot STATIC ${IMPLOT_SOURCES})

target_include_directories(implot 
    SYSTEM PUBLIC
    ${implot_SOURCE_DIR}
    ${imgui_SOURCE_DIR} # ImPlot depends on ImGui headers
)
target_link_libraries(implot PRIVATE imgui)

# Suppress warnings for ImPlot and disable clang-tidy
target_compile_options(implot PRIVATE -w -O0 -fno-var-tracking)
set_target_properties(implot PROPERTIES CXX_CLANG_TIDY "")
