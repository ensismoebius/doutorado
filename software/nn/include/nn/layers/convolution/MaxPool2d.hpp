#ifndef NN_LAYERS_MAXPOOL2D_HPP
#define NN_LAYERS_MAXPOOL2D_HPP

#include "nn/layers/base/Module.hpp"
#include "nn/logging/Logger.hpp"

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

template <typename Backend>
class MaxPool2dImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    MaxPool2dImpl(int kernel, int stride_val, int padding = 0, int dilation = 1)
        : kernel_size_(kernel), stride_(stride_val), padding_(padding), dilation_(dilation)
    {
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const auto shape = input.get_shape();

        if (shape.size() != 4) [[unlikely]]
        {
            NN_LOG_WARN("MaxPool2d: input is not 4-D (no-op)");
            return input;
        }

        const int batch_size = static_cast<int>(shape[0]);
        const int channels = static_cast<int>(shape[1]);
        const int input_height = static_cast<int>(shape[2]);
        const int input_width = static_cast<int>(shape[3]);
        const int output_height = ((input_height - kernel_size_) / stride_) + 1;
        const int output_width = ((input_width - kernel_size_) / stride_) + 1;

        (void) batch_size;
        (void) channels;
        (void) input_height;
        (void) input_width;
        (void) output_height;
        (void) output_width;
        (void) requires_grad;

        return input;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        return grad_output;
    }

   private:
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;
};

#endif // NN_LAYERS_MAXPOOL2D_HPP