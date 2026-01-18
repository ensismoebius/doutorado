#pragma once

#include <iostream>
#include <vector>

#include "nn/layers/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file MaxPool2d.hpp
 * @brief Placeholder MaxPool2d layer.
 *
 * This implementation currently behaves as a guarded no-op:
 * - It checks whether the input tensor reports a 4D shape (N,C,H,W).
 * - Regardless, it returns `input` unchanged.
 *
 * Why keep it around:
 * - It allows higher-level model code to reference `MaxPool2d` without having to
 *   conditionalize compilation while the true 4D pooling implementation is in
 *   progress.
 *
 * If you want real max-pooling:
 * - Implement a 4D-aware path consistent with this project's `Tensor` storage
 *   conventions and add a corresponding `backward()`.
 */

class MaxPool2d : public Module
{
   public:
    MaxPool2d(int kernel, int stride_val) : kernel_size_(kernel), stride_(stride_val) {}

    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        const auto shape = input.get_shape();

        // Our lightweight Tensor wrapper in this project uses a backend-agnostic
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
