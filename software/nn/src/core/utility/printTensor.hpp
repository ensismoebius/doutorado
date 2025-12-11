#pragma once
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>

inline void printTensor(const Eigen::MatrixXf& t, const std::string& name = "Tensor")
{
    std::cout << name << " (" << t.rows() << "x" << t.cols() << ")\n";
    for (auto i = 0; i < t.rows(); ++i) [[likely]]
    {
        std::cout << "[ ";
        for (auto j = 0; j < t.cols(); ++j) [[likely]]
        {
            std::cout << std::setw(4) << std::setprecision(4) << t(i, j);
            if (j < t.cols() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << " ]\n";
    }
}
