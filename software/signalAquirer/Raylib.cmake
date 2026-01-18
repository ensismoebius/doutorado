# Raylib configuration
set(RAYLIB_DIR "${LIB_DIR}/raylib")
set(RAYLIB_BUILD_DIR "${RAYLIB_DIR}/build")

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