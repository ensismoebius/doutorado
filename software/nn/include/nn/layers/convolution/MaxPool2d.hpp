#ifndef NN_LAYERS_MAXPOOL2D_HPP
#define NN_LAYERS_MAXPOOL2D_HPP

#include <limits>
#include <stdexcept>
#include <vector>

#include "nn/layers/base/Module.hpp"

/**
 * @file MaxPool2d.hpp
 * @brief 2D Max Pooling with full backward support.
 *
 * Input shape:  (B, C, H, W)
 * Output shape: (B, C, H_out, W_out)
 *   H_out = (H - kernel) / stride + 1
 *   W_out = (W - kernel) / stride + 1
 *
 * Backward routes each output gradient to the max-position in the
 * corresponding input window. No padding or dilation in this implementation.
 */
template <typename Backend>
class MaxPool2dImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    MaxPool2dImpl(int kernel, int stride_val, int padding = 0, int dilation = 1)
        : kernel_size_(kernel), stride_(stride_val), padding_(padding), dilation_(dilation)
    {
        (void) padding_;
        (void) dilation_;
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const auto shape = input.get_shape();

        if (shape.size() != 4) [[unlikely]]
            throw std::invalid_argument("MaxPool2d: input must be 4-D (B, C, H, W)");

        const int B = static_cast<int>(shape[0]);
        const int C = static_cast<int>(shape[1]);
        const int H = static_cast<int>(shape[2]);
        const int W = static_cast<int>(shape[3]);
        const int H_out = (H - kernel_size_) / stride_ + 1;
        const int W_out = (W - kernel_size_) / stride_ + 1;

        if (H_out <= 0 || W_out <= 0) throw std::invalid_argument("MaxPool2d: output size <= 0");

        Tensor output(static_cast<nn::Index>(B),
            static_cast<nn::Index>(C),
            static_cast<nn::Index>(H_out),
            static_cast<nn::Index>(W_out));

        if (requires_grad)
        {
            argmax_h_.assign(static_cast<size_t>(B * C * H_out * W_out), 0);
            argmax_w_.assign(static_cast<size_t>(B * C * H_out * W_out), 0);
            input_B_ = B;
            input_C_ = C;
            input_H_ = H;
            input_W_ = W;
            input_H_out_ = H_out;
            input_W_out_ = W_out;
        }

        for (int b = 0; b < B; ++b)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int ho = 0; ho < H_out; ++ho)
                {
                    for (int wo = 0; wo < W_out; ++wo)
                    {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_hi = ho * stride_, max_wi = wo * stride_;
                        for (int kh = 0; kh < kernel_size_; ++kh)
                        {
                            for (int kw = 0; kw < kernel_size_; ++kw)
                            {
                                int hi = ho * stride_ + kh;
                                int wi = wo * stride_ + kw;
                                if (hi < H && wi < W)
                                {
                                    float v = input.at(static_cast<nn::Index>(b),
                                        static_cast<nn::Index>(c),
                                        static_cast<nn::Index>(hi),
                                        static_cast<nn::Index>(wi));
                                    if (v > max_val)
                                    {
                                        max_val = v;
                                        max_hi = hi;
                                        max_wi = wi;
                                    }
                                }
                            }
                        }
                        output.at(static_cast<nn::Index>(b),
                            static_cast<nn::Index>(c),
                            static_cast<nn::Index>(ho),
                            static_cast<nn::Index>(wo)) = max_val;

                        if (requires_grad)
                        {
                            size_t idx = static_cast<size_t>(
                                b * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo);
                            argmax_h_[idx] = max_hi;
                            argmax_w_[idx] = max_wi;
                        }
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
            static_cast<nn::Index>(input_H_),
            static_cast<nn::Index>(input_W_));
        dx.setZero();

        for (int b = 0; b < input_B_; ++b)
        {
            for (int c = 0; c < input_C_; ++c)
            {
                for (int ho = 0; ho < input_H_out_; ++ho)
                {
                    for (int wo = 0; wo < input_W_out_; ++wo)
                    {
                        size_t idx = static_cast<size_t>(
                            b * input_C_ * input_H_out_ * input_W_out_ +
                            c * input_H_out_ * input_W_out_ + ho * input_W_out_ + wo);
                        int hi = argmax_h_[idx];
                        int wi = argmax_w_[idx];
                        dx.at(static_cast<nn::Index>(b),
                            static_cast<nn::Index>(c),
                            static_cast<nn::Index>(hi),
                            static_cast<nn::Index>(wi)) += grad_output.at(static_cast<nn::Index>(b),
                            static_cast<nn::Index>(c),
                            static_cast<nn::Index>(ho),
                            static_cast<nn::Index>(wo));
                    }
                }
            }
        }
        return dx;
    } // LCOV_EXCL_LINE

   private:
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;
    std::vector<int> argmax_h_, argmax_w_;
    int input_B_ = 0, input_C_ = 0, input_H_ = 0, input_W_ = 0;
    int input_H_out_ = 0, input_W_out_ = 0;
};

#endif // NN_LAYERS_MAXPOOL2D_HPP
