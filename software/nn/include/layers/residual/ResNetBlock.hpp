#ifndef NN_LAYERS_RESNETBLOCK_HPP
#define NN_LAYERS_RESNETBLOCK_HPP

#include <algorithm>

#include "layers/activations/ReLU.hpp"
#include "layers/base/Module.hpp"
#include "layers/convolution/Conv2d.hpp"

/**
 * @file ResNetBlock.hpp
 * @brief Two-layer convolutional residual block.
 *
 * Forward:  y = ReLU2( ReLU1(Conv1(x)) |> Conv2 + skip(x) )
 * Backward: full BPTT through both conv layers, both ReLUs, and the skip path.
 *
 * Two separate ReLU instances are required so each caches its own activation
 * mask independently (a single shared instance would overwrite the mask on
 * the second call, breaking the first backward).
 *
 * Skip connection: when shapes match, identity; otherwise the overlapping region
 * is copied and the remainder is zero-padded (2D and 4D supported).
 */

template <typename Backend>
class ResNetBlockImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    ResNetBlockImpl(int in_channels, int out_channels)
        : conv1_(in_channels, out_channels, 3),
          relu1_(),
          conv2_(out_channels, out_channels, 3),
          relu2_()
    {
    }

    Tensor forward(const Tensor& input, bool requires_grad = true) override
    {
        Tensor output = conv1_.forward(input, requires_grad);
        output = relu1_.forward(output, requires_grad);
        output = conv2_.forward(output, requires_grad);

        // Skip connection with shape alignment
        const auto out_shape = output.get_shape();
        const auto in_shape = input.get_shape();
        if (out_shape == in_shape) [[likely]]
        {
            output = output + input; // 
        }
        else
        {
            output = output + align_to_shape(input, out_shape);
        }

        // Store skip input for backward before final relu
        if (requires_grad) skip_input_ = input;

        return relu2_.forward(output, requires_grad);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // 1. Through final ReLU2
        Tensor grad = relu2_.backward(grad_output);

        // 2. Flows to both residual branch and skip path
        Tensor grad_skip = grad;

        // 3. Through Conv2
        Tensor grad_conv2 = conv2_.backward(grad);

        // 4. Through ReLU1
        Tensor grad_relu1 = relu1_.backward(grad_conv2);

        // 5. Through Conv1 → gradient w.r.t. input
        Tensor grad_x = conv1_.backward(grad_relu1);

        // 6. Add skip gradient (same shape-alignment logic as forward)
        const auto out_shape = grad_skip.get_shape();
        const auto in_shape = grad_x.get_shape();
        if (out_shape == in_shape) [[likely]]
        {
            grad_x = grad_x + grad_skip; // 
        }
        else
        {
            grad_x = grad_x + align_to_shape(grad_skip, in_shape);
        }

        return grad_x;
    }

   private:
    static auto align_to_shape(const Tensor& src, const std::vector<nn::Index>& target_shape)
        -> Tensor
    {
        Tensor aligned(target_shape);
        aligned.fill(0.0f);

        const auto src_shape = src.get_shape();
        if (src_shape == target_shape)
        { // 
            return src; // 
        }

        if (src_shape.size() == 2 && target_shape.size() == 2)
        { // 
            const nn::Index rows_copy = std::min(src_shape[0], target_shape[0]); // 
            const nn::Index cols_copy = std::min(src_shape[1], target_shape[1]); // 
            for (nn::Index r = 0; r < rows_copy; ++r)                            // 
            { // 
                for (nn::Index c = 0; c < cols_copy; ++c) // 
                { // 
                    aligned.at(r, c) = src.at(r, c); // 
                }
            } // 
            return aligned; // 
        }

        if (src_shape.size() == 4 && target_shape.size() == 4)
        {
            const nn::Index n_copy = std::min(src_shape[0], target_shape[0]);
            const nn::Index c_copy = std::min(src_shape[1], target_shape[1]);
            const nn::Index h_copy = std::min(src_shape[2], target_shape[2]);
            const nn::Index w_copy = std::min(src_shape[3], target_shape[3]);
            for (nn::Index n = 0; n < n_copy; ++n)
            {
                for (nn::Index c = 0; c < c_copy; ++c)
                {
                    for (nn::Index h = 0; h < h_copy; ++h)
                    {
                        for (nn::Index w = 0; w < w_copy; ++w)
                        {
                            aligned.at(n, c, h, w) = src.at(n, c, h, w);
                        }
                    }
                }
            }
            return aligned;
        }

        const nn::Index linear_copy = std::min(src.size(), aligned.size()); // 
        for (nn::Index i = 0; i < linear_copy; ++i)                         // 
        { // 
            aligned.at(i) = src.at(i); // 
        } // 
        return aligned; // 
    }

    Conv2dImpl<Backend> conv1_;
    ReLUImpl<Backend> relu1_;
    Conv2dImpl<Backend> conv2_;
    ReLUImpl<Backend> relu2_;
    Tensor skip_input_; // cached for potential future use
};

#endif // NN_LAYERS_RESNETBLOCK_HPP
