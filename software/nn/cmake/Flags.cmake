# Set the policy for timestamp handling in FetchContent
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

find_program(CLANG_CXX_COMPILER_PATH NAMES clang++)
find_program(CLANG_C_COMPILER_PATH NAMES clang)

if(NOT CLANG_CXX_COMPILER_PATH)
    message(FATAL_ERROR "clang++ compiler not found. Please ensure clang is installed and in your system's PATH, or specify its location via CMAKE_CXX_COMPILER.")
endif()
if(NOT CLANG_C_COMPILER_PATH)
    message(FATAL_ERROR "clang C compiler not found. Please ensure clang is installed and in your system's PATH, or specify its location via CMAKE_C_COMPILER.")
endif()

set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER_PATH}")
set(CMAKE_C_COMPILER "${CLANG_C_COMPILER_PATH}")
set(CMAKE_LINKER lld)

add_compile_options(-g -gdwarf-5)

# Verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Generate compile_commands.json for code completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set more readable and friendly error messages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -fdiagnostics-show-option -g -Wall -Wpedantic -Wshadow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always -fdiagnostics-show-option -g -Wall -Wpedantic -Wshadow")

# Set debug flags for the Debug build type
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb -O0 -fno-inline -fno-omit-frame-pointer -march=native")

# Set optimization flags for the Release build type
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")

# Sets opengl provider to a more
# modern option: GLVND (OpenGL 
# Vendor-Neutral Dispatch).
# If you are having compatibilities
# issues set to "LEGACY"
set(OpenGL_GL_PREFERENCE "GLVND")