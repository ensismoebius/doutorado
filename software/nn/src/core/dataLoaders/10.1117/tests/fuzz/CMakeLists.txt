##
## Optional libFuzzer targets for 10.1117 MAT loaders
##
## Enable with:
##   -DNN_ENABLE_FUZZING=ON
##
## Notes:
## - Clang is required for `-fsanitize=fuzzer`.
## - Targets are intentionally separate from GoogleTest executables.
##

option(NN_ENABLE_FUZZING "Build libFuzzer targets for 10.1117 loaders" ON)

if(NOT NN_ENABLE_FUZZING)
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "NN_ENABLE_FUZZING=ON ignored: libFuzzer targets currently require Clang.")
    return()
endif()

set(FUZZ_SANITIZER_COMPILE_FLAGS
    -fsanitize=fuzzer,address,undefined
    -fno-omit-frame-pointer
)

set(FUZZ_SANITIZER_LINK_FLAGS
    -fsanitize=fuzzer,address,undefined
)

function(add_10_1117_loader_fuzz_target target_name source_file)
    add_executable(${target_name}
        ${source_file}
    )

    target_include_directories(${target_name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(${target_name}
        PRIVATE
            dataLoaders_10_1117
    )

    target_compile_options(${target_name}
        PRIVATE
            ${FUZZ_SANITIZER_COMPILE_FLAGS}
    )

    target_link_options(${target_name}
        PRIVATE
            ${FUZZ_SANITIZER_LINK_FLAGS}
    )
endfunction()

add_10_1117_loader_fuzz_target(fuzz_eeg_loader fuzz_eeg_loader.cpp)
add_10_1117_loader_fuzz_target(fuzz_audio_loader fuzz_audio_loader.cpp)
