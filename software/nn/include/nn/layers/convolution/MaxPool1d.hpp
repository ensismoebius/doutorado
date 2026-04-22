#ifndef NN_LAYERS_MAXPOOL1D_HPP
#define NN_LAYERS_MAXPOOL1D_HPP

#include "nn/layers/base/Module.hpp"
#include "nn/logging/Logger.hpp"

/**
 * @file MaxPool1d.hpp
 * @brief 1D Max Pooling Layer for Neural Networks
 *
 * Implements max pooling for 1D input (temporal data).
 *
 * Currently behaves as a guarded no-op:
 * - Checks whether the input tensor reports a 3D shape (N, C, L).
 * - Returns `input` unchanged if not 3D.
 *
 * Why keep it around:
 * - Allows higher-level model code to reference `MaxPool1d` without
 *   conditionalizing compilation while the true 1D pooling implementation
 *   is in progress.
 *
 * If you want real max-pooling:
 * - Implement a 3D-aware path consistent with this project's `Tensor`
 *   conventions and add a corresponding `backward()`.
 */

template <typename Backend>
class MaxPool1dImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    MaxPool1dImpl(int kernel, int stride_val) : kernel_size_(kernel), stride_(stride_val) {}

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const auto shape = input.get_shape();

        if (shape.size() != 3) [[unlikely]]
        {
            NN_LOG_WARN("MaxPool1d: input is not 3-D (no-op)");
            return input;
        }

        const int batch_size = static_cast<int>(shape[0]);
        const int channels = static_cast<int>(shape[1]);
        const int input_length = static_cast<int>(shape[2]);
        const int output_length = ((input_length - kernel_size_) / stride_) + 1;

        (void) batch_size;
        (void) channels;
        (void) input_length;
        (void) output_length;

        return input;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        return grad_output;
    }

   private:
    int kernel_size_;
    int stride_;
};

#endif // NN_LAYERS_MAXPOOL1D_HPP
