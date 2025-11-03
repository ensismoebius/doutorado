#pragma once

#include "Module.hpp"
#include "tensor/Tensor.hpp"

class MaxPool2d : public Module {
public:
    MaxPool2d(int kernel_size, int stride) :
        kernel_size_(kernel_size),
        stride_(stride)
    {
    }

    Tensor forward(const Tensor& input) override {
        const int batch_size = input.get_shape()[0];
        const int channels = input.get_shape()[1];
        const int input_height = input.get_shape()[2];
        const int input_width = input.get_shape()[3];
        const int output_height = (input_height - kernel_size_) / stride_ + 1;
        const int output_width = (input_width - kernel_size_) / stride_ + 1;

        Tensor output(batch_size, channels, output_height, output_width);

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oy = 0; oy < output_height; ++oy) {
                    for (int ox = 0; ox < output_width; ++ox) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int ky = 0; ky < kernel_size_; ++ky) {
                            for (int kx = 0; kx < kernel_size_; ++kx) {
                                max_val = std::max(max_val, input.data(b, c, oy * stride_ + ky, ox * stride_ + kx));
                            }
                        }
                        output.data(b, c, oy, ox) = max_val;
                    }
                }
            }
        }

        return output;
    }

private:
    int kernel_size_;
    int stride_;
};
