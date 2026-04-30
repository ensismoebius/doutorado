#+#+#+#+-----------------------------------------------------------------------
# PrecompiledHeaders.cmake
#
# Optional PCH helpers for faster incremental rebuilds on heavy C++ targets.
# Enabled via NN_ENABLE_PCH option and only when target_precompile_headers is
# available in the running CMake version.
#+#+#+#+-----------------------------------------------------------------------

function(nn_enable_target_pch target_name)
    if(NOT NN_ENABLE_PCH)
        return()
    endif()

    if(NOT COMMAND target_precompile_headers)
        message(STATUS "PCH requested but target_precompile_headers is unavailable in this CMake version")
        return()
    endif()

    if(NOT TARGET ${target_name})
        message(STATUS "PCH skipped: target ${target_name} does not exist")
        return()
    endif()

    target_precompile_headers(${target_name}
        PRIVATE
            <vector>
            <string>
            <memory>
            <array>
            <algorithm>
            <functional>
            <cstddef>
            <cstdint>
            <cmath>
            <stdexcept>
            <optional>
            <numeric>
            <utility>
            <unordered_map>
            <tuple>
    )

    message(STATUS "PCH enabled for target: ${target_name}")
endfunction()
