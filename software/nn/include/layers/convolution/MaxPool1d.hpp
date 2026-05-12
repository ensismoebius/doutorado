#ifndef NN_LAYERS_MAXPOOL1D_HPP
#define NN_LAYERS_MAXPOOL1D_HPP

#include <limits>
#include <stdexcept>
#include <vector>

#include "layers/base/Module.hpp"

/**
 * @file MaxPool1d.hpp
 * @brief 1D Max Pooling with full backward support.
 *
 * Input shape:  (B, C, L)        — batch × channels × length
 * Output shape: (B, C, L_out)    — L_out = (L - kernel) / stride + 1
 *
 * Backward routes each output gradient to the max-position in the
 * corresponding input window (max-unpooling / argmax routing).
 *
 * No padding or dilation supported in this implementation.
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
            throw std::invalid_argument("MaxPool1d: input must be 3-D (B, C, L)");

        const int B = static_cast<int>(shape[0]);
        const int C = static_cast<int>(shape[1]);
        const int L = static_cast<int>(shape[2]);
        const int L_out = (L - kernel_size_) / stride_ + 1;

        if (L_out <= 0)
            throw std::invalid_argument("MaxPool1d: output length <= 0 for given kernel/stride/L");

        Tensor output(
            static_cast<nn::Index>(B), static_cast<nn::Index>(C), static_cast<nn::Index>(L_out));

        if (requires_grad)
        {
            // Argmax stored as flat input indices (cast to float for Tensor storage)
            argmax_flat_.assign(static_cast<size_t>(B * C * L_out), 0);
            input_B_ = B;
            input_C_ = C;
            input_L_ = L;
            input_L_out_ = L_out;
        }

        for (int b = 0; b < B; ++b)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int lo = 0; lo < L_out; ++lo)
                {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_li = lo * stride_;
                    for (int k = 0; k < kernel_size_; ++k)
                    {
                        int li = lo * stride_ + k;
                        if (li < L)
                        {
                            float v = input.at(static_cast<nn::Index>(b),
                                static_cast<nn::Index>(c),
                                static_cast<nn::Index>(li));
                            if (v > max_val)
                            {
                                max_val = v;
                                max_li = li;
                            }
                        }
                    }
                    output.at(static_cast<nn::Index>(b),
                        static_cast<nn::Index>(c),
                        static_cast<nn::Index>(lo)) = max_val;

                    if (requires_grad)
                    {
                        // Flat index: b * (C * L_out) + c * L_out + lo
                        argmax_flat_[static_cast<size_t>(b * C * L_out + c * L_out + lo)] = max_li;
                    }
                }
            }
        }
        return output;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        Tensor dx(static_cast<nn::Index>(input_B_),
            static_cast<nn::Index>(input_C_),
            static_cast<nn::Index>(input_L_));
        dx.setZero();

        for (int b = 0; b < input_B_; ++b)
        {
            for (int c = 0; c < input_C_; ++c)
            {
                for (int lo = 0; lo < input_L_out_; ++lo)
                {
                    int li = argmax_flat_[static_cast<size_t>(
                        b * input_C_ * input_L_out_ + c * input_L_out_ + lo)];
                    dx.at(static_cast<nn::Index>(b),
                        static_cast<nn::Index>(c),
                        static_cast<nn::Index>(li)) += grad_output.at(static_cast<nn::Index>(b),
                        static_cast<nn::Index>(c),
                        static_cast<nn::Index>(lo));
                }
            }
        }
        return dx;
    } //

   private:
    int kernel_size_;
    int stride_;
    // Argmax cache (max input positions for each output element)
    std::vector<int> argmax_flat_;
    int input_B_ = 0, input_C_ = 0, input_L_ = 0, input_L_out_ = 0;
};

#endif // NN_LAYERS_MAXPOOL1D_HPP
