// tensor_bridge.hpp — nn::Tensor <-> NumPy, for the nn_microscope module.
//
// nn::Tensor's xtensor backend stores a 2-D (rows, cols) logical array in
// COLUMN-MAJOR order (see the "Column-major storage" note in
// src/experiments/meeting01/lib/src/Meeting01Encoding.cpp). A naive
// row-major memcpy from data_ptr() therefore transposes the data. These
// helpers go through element accessors (at(r, c)) so the mapping is correct
// regardless of internal layout, and copy (never alias) so Python cannot
// mutate C++ state.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tensor/Tensor.hpp"

namespace nn_microscope
{

namespace py = pybind11;

// nn::Tensor (2-D, treated as (rows, cols)) -> float64 ndarray of shape
// (rows, cols). Uses at(r, c) so column-major storage is handled.
inline py::array_t<double> to_numpy(const nn::Tensor& t)
{
    const auto shape = t.get_shape();
    const std::size_t rows = t.rows();
    const std::size_t cols = t.cols();
    py::array_t<double> out({rows, cols});
    auto view = out.mutable_unchecked<2>();
    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            view(r, c) =
                static_cast<double>(t.at(static_cast<nn::Index>(r), static_cast<nn::Index>(c)));
        }
    }
    return out;
}

// 1-D or 2-D float ndarray -> nn::Tensor of shape (rows, cols). A 1-D array
// of length n becomes (n, 1) (the window convention used by meeting01).
inline nn::Tensor from_numpy(const py::array& arr)
{
    py::array_t<double, py::array::c_style | py::array::forcecast> a(arr);
    const auto info = a.request();
    std::size_t rows = 0;
    std::size_t cols = 0;
    if (info.ndim == 1)
    {
        rows = static_cast<std::size_t>(info.shape[0]);
        cols = 1;
    }
    else if (info.ndim == 2)
    {
        rows = static_cast<std::size_t>(info.shape[0]);
        cols = static_cast<std::size_t>(info.shape[1]);
    }
    else
    {
        throw std::invalid_argument(
            "from_numpy: expected a 1-D or 2-D array, got ndim=" + std::to_string(info.ndim));
    }
    nn::Tensor t(
        std::vector<nn::Index>{static_cast<nn::Index>(rows), static_cast<nn::Index>(cols)});
    auto view = a.unchecked<>(); // generic, works for 1-D and 2-D
    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            const double value = (info.ndim == 1) ? view(r) : view(r, c);
            t.at(static_cast<nn::Index>(r), static_cast<nn::Index>(c)) = static_cast<float>(value);
        }
    }
    return t;
}

inline std::vector<double> to_double_vector(const py::array& arr)
{
    py::array_t<double, py::array::c_style | py::array::forcecast> a(arr);
    const auto info = a.request();
    const auto* data = static_cast<const double*>(info.ptr);
    return std::vector<double>(data, data + info.size);
}

} // namespace nn_microscope
