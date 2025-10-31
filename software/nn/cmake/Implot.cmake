# IMPlot configuration
set(IMPLOT_DIR "${LIB_DIR}/implot")

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

# Add ImPlot source files
set(IMPLOT_SOURCES
    ${IMPLOT_DIR}/implot_items.cpp
    ${IMPLOT_DIR}/implot.cpp
)