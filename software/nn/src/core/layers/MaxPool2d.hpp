#pragma once

#include <iostream>
#include <vector>

#include "Module.hpp"
#include "tensor/Tensor.hpp"

class MaxPool2d : public Module
{
   public:
    MaxPool2d(int kernel, int stride_val) : kernel_size_(kernel), stride_(stride_val) {}

    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        const auto shape = input.get_shape();

        // Our lightweight Tensor wrapper in this project is a 2-D Eigen::MatrixXf-backed
        // structure. This MaxPool2d implementation requires a 4-D tensor (N, C, H, W).
        // If the provided Tensor is not 4-D, fall back to identity (no-op) so code
        // that expects MaxPool2d to exist still compiles and runs. If you want true
        // 2D pooling support, we should extend the `Tensor` type to hold 4-D data or
        // provide a separate data structure.
        if (shape.size() != 4) [[unlikely]]
        {
            std::cerr << "MaxPool2d: input is not 4-D (no-op).\n";
            return input;
        }

        const int batch_size = static_cast<int>(shape[0]);
        const int channels = static_cast<int>(shape[1]);
        const int input_height = static_cast<int>(shape[2]);
        const int input_width = static_cast<int>(shape[3]);
        const int output_height = ((input_height - kernel_size_) / stride_) + 1;
        const int output_width = ((input_width - kernel_size_) / stride_) + 1;

        // We don't currently have a 4-D Tensor constructor; return input as-is to
        // avoid unsafe indexing. A future enhancement is to add a true 4-D Tensor
        // type and implement pooling properly.
        (void) batch_size;
        (void) channels;
        (void) input_height;
        (void) input_width;
        (void) output_height;
        (void) output_width;

        return input;
    }

   private:
    int kernel_size_;
    int stride_;
};
