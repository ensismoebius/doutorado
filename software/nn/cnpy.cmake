# Cnpy configuration
# Cnpy lets you read and write to .npy and .npz formats in C++.
set(CNPY_DIR "${LIB_DIR}/cnpy")

# Check if cnpy is already downloaded and compiled
if(NOT EXISTS "${CNPY_DIR}/cnpy.h")
    message(STATUS "Cnpy not found, downloading and compiling...")

    if(NOT EXISTS "${CNPY_DIR}/.git")
        # Clone cnpy repository if not already cloned
        execute_process(
            COMMAND git clone --depth 1 https://github.com/rogersce/cnpy.git ${CNPY_DIR}
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to clone Cnpy repository")
        endif()
    else()
        # Update cnpy repository if already cloned
        execute_process(
            COMMAND git -C ${CNPY_DIR} pull
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to update Cnpy repository")
        endif()
    endif()
else()
    message(STATUS "Cnpy already exists, skipping download and compilation...")
endif()