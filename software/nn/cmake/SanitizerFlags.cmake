# cmake/SanitizerFlags.cmake

add_compile_options(
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
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
