#pragma once
#include <iomanip>
#include <iostream>

#include "core/tensor/Tensor.hpp"

inline void printTensor(const Tensor& t, const std::string& name = "Tensor")
{
    std::cout << name << " (" << t.data.rows() << "x" << t.data.cols() << ")\n";
    for (int i = 0; i < t.data.rows(); ++i)
    {
        std::cout << "[ ";
        for (int j = 0; j < t.data.cols(); ++j)
        {
            std::cout << std::setw(4) << std::setprecision(4) << t.data(i, j);
            if (j < t.data.cols() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << " ]\n";
    }
}
