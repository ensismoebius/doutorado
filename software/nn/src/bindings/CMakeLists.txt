# nn_microscope — Python extension consumed by software/experiment_microscope/.
#
# Gated by NN_BUILD_PY_BINDINGS (the `python-bindings` preset). When OFF this
# file is not even added (see the top-level CMakeLists.txt guard), so ordinary
# builds are unaffected.

pybind11_add_module(nn_microscope MODULE
    module.cpp
    bind_wavelet.cpp
    bind_meeting01.cpp
    bind_thesis.cpp
)

target_include_directories(nn_microscope
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/experiments/thesis/lib/include
        ${CMAKE_SOURCE_DIR}/src/experiments/meeting01/lib/include
)

# Concrete, non-templated experiment libraries + the transitive deps their
# executables already declare. wavelet/linear_algebra cover the current
# submodule; meeting01_lib / thesis_lib are wired ahead of the encoding /
# paraconsistent submodules.
target_link_libraries(nn_microscope
    PRIVATE
        wavelet
        linear_algebra
        util
        # meeting01 window pipeline + SNN autoencoder forward (bind_meeting01.cpp).
        # Mirrors the `meeting01` executable's link line + the tensor backend.
        meeting01_lib
        thesis_lib
        autoencoderRunner_lib
        tensor
        xsimd
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        data_loaders
        data_loaders_fsdd
        matioCpp
        z
        xtensor
        xtensor-blas
        OpenMP::OpenMP_CXX
)

configure_xtensor_parallel_target(nn_microscope)

set_target_properties(nn_microscope PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Third-party generated code — do not run clang-tidy over pybind headers.
if(TARGET pybind11::pybind11)
    set_target_properties(pybind11::pybind11 PROPERTIES CXX_CLANG_TIDY "")
endif()
