#pragma once

#include "nn/layers/Conv2d.hpp"
#include "nn/layers/Module.hpp"
#include "nn/layers/ReLU.hpp"

/**
 * @file ResNetBlock.hpp
 * @brief Simple 2-layer convolutional residual block.
 *
 * Intended behavior (high level):
 *   y = ReLU( x + Conv2(ReLU(Conv1(x))) )
 *
 * Notes in this codebase:
 * - There is no learnable projection on the skip path; when shapes differ, the
 *   implementation tries to align by zero-padding/cropping the underlying 2D
 *   storage. This is a pragmatic fallback, not a canonical ResNet design.
 * - For a production-quality conv ResNet, add explicit shape management
 *   (padding/stride choices) or a 1x1 conv projection.
 */

class ResNetBlock : public Module
{
   public:
    ResNetBlock(int in_channels, int out_channels)
        : conv1_(in_channels, out_channels, 3), relu_(), conv2_(out_channels, out_channels, 3)
    {
    }

    nn::Tensor forward(const nn::Tensor& input, bool requires_grad = true) override
    {
        nn::Tensor output = conv1_.forward(input, requires_grad);
        output = relu_.forward(output, requires_grad);
        output = conv2_.forward(output, requires_grad);

        // Add skip connection with shape alignment
        // Handles dimension mismatches by zero-padding/cropping overlapping regions
        const auto out_shape = output.get_shape();
        const auto in_shape = input.get_shape();
        if (out_shape == in_shape) [[likely]]
        {
            output = output + input;
        }
        else
        {
            // Create zero matrix matching output underlying data and copy overlapping region
            nn::Tensor aligned(output.rows(), output.cols());
            aligned.setZero();

            const auto& in_mat = input;
            const auto& out_mat = output;
            auto& aligned_mat = aligned;

            const auto rows_copy = static_cast<int>(std::min(in_mat.rows(), aligned_mat.rows()));
            const auto cols_copy = static_cast<int>(std::min(in_mat.cols(), aligned_mat.cols()));

            aligned_mat.block(0, 0, rows_copy, cols_copy) =
                in_mat.block(0, 0, rows_copy, cols_copy);
            output = out_mat + aligned_mat;
        }

        return relu_.forward(output, requires_grad);
    }

   private:
    Conv2d conv1_;
    ReLU relu_;
    Conv2d conv2_;
};
