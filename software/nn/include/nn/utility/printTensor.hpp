#pragma once

#include <iomanip>
#include <iostream>

#include "nn/tensor/Tensor.hpp"

/**
 * @file printTensor.hpp
 * @brief Debugging helper to print a 2D tensor to stdout.
 *
 * This is intentionally simple and meant for small tensors in debug builds.
 * For large tensors, printing can dominate runtime and generate huge logs.
 */

inline void printTensor(const nn::Tensor& t, const std::string& name = "Tensor")
{
    std::cout << name << " (" << t.rows() << "x" << t.cols() << ")\n";
    for (size_t i = 0; i < t.rows(); ++i) [[likely]]
    {
        std::cout << "[ ";
        for (size_t j = 0; j < t.cols(); ++j) [[likely]]
        {
            std::cout << std::setw(4) << std::setprecision(4) << t.at(i, j);
            if (j + 1 < t.cols())
            {
                std::cout << ", ";
            }
        }
        std::cout << " ]\n";
    }
}
