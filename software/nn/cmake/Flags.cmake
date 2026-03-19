#+#+#+#+-----------------------------------------------------------------------
# Flags.cmake
#
# Central compiler and configuration flags for the whole project.
#
# Philosophy:
# - Keep flags consistent across all targets.
# - Prefer warnings that catch common mistakes.
# - Keep subprojects (vendored deps) from enabling their own tests.
#+#+#+#+-----------------------------------------------------------------------

# To use the LLVM/Clang toolchain, set the CC and CXX environment variables
# before configuring CMake, for example:
#
#   $ env CC=clang CXX=clang++ cmake -S . -B build
#
# To use the lld linker, you can pass it via a CMake variable:
#
#   $ cmake -S . -B build -DCMAKE_LINKER=lld
#
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

# Prefer a faster linker when available to speed up incremental relinking.
option(NN_ENABLE_FAST_LINKER "Enable mold/lld fast linker when available" ON)

# Optional optimization for incremental builds: use OBJECT libraries for select
# internal targets to reduce archive/relink overhead.
option(NN_USE_OBJECT_LIBRARIES "Enable opt-in OBJECT libraries for selected internal targets" OFF)

# Optional optimization for incremental builds: per-target precompiled headers.
option(NN_ENABLE_PCH "Enable opt-in precompiled headers for selected targets" ON)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prevent vendored subprojects from enabling their own tests during configure
# so we don't require test frameworks for third-party code while configuring.
set(BUILD_TESTING OFF CACHE BOOL "Disable building tests in subprojects" FORCE)

# Verbose output during builds
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NN_ENABLE_FAST_LINKER)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        find_program(NN_MOLD_LINKER mold)
        find_program(NN_LLD_LINKER NAMES ld.lld lld)

        if(NN_MOLD_LINKER)
            if(NOT CMAKE_EXE_LINKER_FLAGS MATCHES "(^| )-fuse-ld=")
                string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=mold")
            endif()
            if(NOT CMAKE_SHARED_LINKER_FLAGS MATCHES "(^| )-fuse-ld=")
                string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fuse-ld=mold")
            endif()
            if(NOT CMAKE_MODULE_LINKER_FLAGS MATCHES "(^| )-fuse-ld=")
                string(APPEND CMAKE_MODULE_LINKER_FLAGS " -fuse-ld=mold")
            endif()
            message(STATUS "Fast linker enabled: mold (${NN_MOLD_LINKER})")
        elseif(NN_LLD_LINKER)
            if(NOT CMAKE_EXE_LINKER_FLAGS MATCHES "(^| )-fuse-ld=")
                string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")
            endif()
            if(NOT CMAKE_SHARED_LINKER_FLAGS MATCHES "(^| )-fuse-ld=")
                string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fuse-ld=lld")
            endif()
            if(NOT CMAKE_MODULE_LINKER_FLAGS MATCHES "(^| )-fuse-ld=")
                string(APPEND CMAKE_MODULE_LINKER_FLAGS " -fuse-ld=lld")
            endif()
            message(STATUS "Fast linker enabled: lld (${NN_LLD_LINKER})")
        else()
            message(STATUS "Fast linker requested but mold/lld not found; using default linker")
        endif()
    else()
        message(STATUS "Fast linker option is only supported for GNU/Clang toolchains")
    endif()
else()
    message(STATUS "Fast linker disabled (NN_ENABLE_FAST_LINKER=OFF)")
endif()

# --- Common flags for all build types ---
add_compile_options(
    -Wall
    -Wpedantic
    -Wshadow
    -fdiagnostics-color=always
    -fdiagnostics-show-option
    -Wpessimizing-move
    -Wredundant-move
    -Wno-user-defined-literals
    -Wno-unknown-warning-option
)

# --- Debug-specific flags ---
# Note: -g3 implies -g (level 3 is a superset); -fno-inline is a no-op at -O0.
add_compile_options(
    $<$<CONFIG:Debug>:-gdwarf-5>
    $<$<CONFIG:Debug>:-g3>
    $<$<CONFIG:Debug>:-ggdb>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-march=native>
)

# --- Release-specific flags ---
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)

# Sets opengl provider to a more
# modern option: GLVND (OpenGL 
# Vendor-Neutral Dispatch).
# If you are having compatibilities
# issues set to "LEGACY"
set(OpenGL_GL_PREFERENCE "GLVND")