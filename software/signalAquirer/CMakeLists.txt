cmake_minimum_required(VERSION 3.10)
project(SignalAcquirer VERSION 1.0 LANGUAGES CXX)

# Enable testing
enable_testing()

# Sets up the flags for CMake and compiler
include(Flags.cmake)

# Do the required libraries check
include(PackageChecking.cmake)

# Define paths needed for the step bellow
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")

# Loads and compiles the required libraries
include(Imgui.cmake)
include(Implot.cmake)
include(Raylib.cmake)
include(GTestGMock.cmake)

# Compiles the main program
include(Main.cmake)

# Compiles unity tests
include(Tests.cmake)

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