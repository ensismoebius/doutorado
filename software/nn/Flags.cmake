# Set the policy for timestamp handling in FetchContent
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Generate compile_commands.json for code completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set more readable and friendly error messages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=never -fdiagnostics-show-option -g -Wall -Wpedantic -Wshadow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=never -fdiagnostics-show-option -g -Wall -Wpedantic -Wshadow")

# Set debug flags for the Debug build type
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb -O0 -fno-inline -fno-omit-frame-pointer -march=native")

# Set optimization flags for the Release build type
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")