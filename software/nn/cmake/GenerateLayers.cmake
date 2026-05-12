# cmake/GenerateLayers.cmake
# Scans include/layers/**/*.hpp, derives aliases (Foo -> FooImpl<Backend>),
# and generates include/layers/Layers.hpp via configure_file.
#
# Convention: Foo.hpp must define FooImpl<Backend> for auto-aliasing.
# Files that don't follow this convention are listed in _exclude_patterns.
#
# Trigger: CONFIGURE_DEPENDS causes cmake to re-run when the file list changes.

file(GLOB_RECURSE _all_layer_headers
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/include/layers/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/layers/**/*.hpp")

# Exclude non-Backend-templated files and utility headers
set(_exclude_patterns
    ".*/eigen/.*"                    # old alias file (becomes shim)
    ".*/base/Module\\.hpp$"          # abstract base, no alias
    ".*/convolution/Conv2d_utils\\.hpp$"  # utility structs, not a layer
    ".*/regularization/.*"           # L1/L2/IRegularization: not templated on Backend
    ".*/spiking/ISurrogateGradient\\.hpp$"
    ".*/spiking/BoxcarSurrogate\\.hpp$"
    ".*/spiking/ExponentialSurrogate\\.hpp$"
    ".*/spiking/SurrogateGradient\\.hpp$"  # umbrella include
    ".*/layers/Layers\\.hpp$"        # the generated file itself — avoid self-include
    ".*/activations/FastActivations\\.hpp$"  # inline functions only, no FooImpl<Backend> class
)
foreach(_pat IN LISTS _exclude_patterns)
    list(FILTER _all_layer_headers EXCLUDE REGEX "${_pat}")
endforeach()

list(SORT _all_layer_headers)

set(LAYER_INCLUDES  "")
set(LAYER_ALIASES   "")

foreach(_header IN LISTS _all_layer_headers)
    file(RELATIVE_PATH _rel "${CMAKE_SOURCE_DIR}/include" "${_header}")
    get_filename_component(_name "${_header}" NAME_WE)   # e.g. "ReLU"
    string(APPEND LAYER_INCLUDES  "#include \"${_rel}\"\n")
    string(APPEND LAYER_ALIASES   "    using ${_name} = ${_name}Impl<Backend>;\n")
endforeach()

# Non-template extras included verbatim (regularization + surrogate grab-bags)
set(MANUAL_INCLUDES
"#include \"layers/regularization/Regularization.hpp\"
#include \"layers/spiking/SurrogateGradient.hpp\"")

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/Layers.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/layers/Layers.hpp"
    @ONLY)
