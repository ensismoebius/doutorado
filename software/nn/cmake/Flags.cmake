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

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prevent vendored subprojects from enabling their own tests during configure
# so we don't require test frameworks for third-party code while configuring.
set(BUILD_TESTING OFF CACHE BOOL "Disable building tests in subprojects" FORCE)

# Verbose output during builds
set(CMAKE_VERBOSE_MAKEFILE ON)

# Generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Common flags for all build types ---
add_compile_options(
    # -Wall
    # -Wpedantic
    # -Wshadow
    -fdiagnostics-color=always
    -fdiagnostics-show-option
)

# --- Debug-specific flags ---
add_compile_options(
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-gdwarf-5>
    $<$<CONFIG:Debug>:-g3>
    $<$<CONFIG:Debug>:-ggdb>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-fno-inline>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
    $<$<CONFIG:Debug>:-march=native>
)

# Enable sanitizers for Debug builds when not using MSVC.
# AddressSanitizer and UndefinedBehaviorSanitizer need both compile and
# link flags so add them via generator expressions for Debug config.
if (NOT MSVC)
    add_compile_options(
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fsanitize=undefined>
    )

    # Ensure linker also receives sanitizer options. Use add_link_options when
    # available (older CMake installs may not provide it), otherwise fall back
    # to setting the debug linker flags. The CMP0069 policy is set in
    # `cmake/Policies.cmake` so we don't need to test for it here.
    if (COMMAND add_link_options)
        add_link_options(
            $<$<CONFIG:Debug>:-fsanitize=address>
            $<$<CONFIG:Debug>:-fsanitize=undefined>
        )
    else()
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")
        set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")
    endif()
endif()

# --- Release-specific flags ---
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-ffast-math>
)

# Sets opengl provider to a more
# modern option: GLVND (OpenGL 
# Vendor-Neutral Dispatch).
# If you are having compatibilities
# issues set to "LEGACY"
set(OpenGL_GL_PREFERENCE "GLVND")