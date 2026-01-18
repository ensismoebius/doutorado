set(CMAKE_VERBOSE_MAKEFILE ON)

# Generate compile_commands.json for code completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set more readable and friendly error messages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -fdiagnostics-show-option -g -Wall -Wpedantic -Wshadow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always -fdiagnostics-show-option -g -Wall -Wpedantic -Wshadow")

# Sets opengl provider to a more
# modern option: GLVND (OpenGL 
# Vendor-Neutral Dispatch).
# If you are having compatibilities
# issues set to "LEGACY"
set(OpenGL_GL_PREFERENCE "GLVND")