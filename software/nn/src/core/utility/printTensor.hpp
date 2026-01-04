#pragma once

#include <iomanip>
#include <iostream>

#include "../tensor/Tensor.hpp"

inline void printTensor(const nn::Tensor& t, const std::string& name = "Tensor")
{
    std::cout << name << " (" << t.rows() << "x" << t.cols() << ")\n";
    for (nn::Index i = 0; i < t.rows(); ++i) [[likely]]
    {
        std::cout << "[ ";
        for (nn::Index j = 0; j < t.cols(); ++j) [[likely]]
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
