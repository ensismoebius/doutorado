#pragma once

#include "Module.hpp"
#include "tensor/Tensor.hpp"

class Conv2d : public Module {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size) :
        in_channels_(in_channels),
        out_channels_(out_channels),
        kernel_size_(kernel_size),
        weights_(kernel_size * kernel_size * in_channels, out_channels),
        bias_(1, out_channels)
    {
        // Initialize weights and bias
    }

    Tensor forward(const Tensor& input) override {
        // Naive implementation of 2D convolution
        // Input shape: (batch_size, in_channels, input_height, input_width)
        // Output shape: (batch_size, out_channels, output_height, output_width)
        const int batch_size = input.get_shape()[0];
        const int input_height = input.get_shape()[2];
        const int input_width = input.get_shape()[3];
        const int output_height = input_height - kernel_size_ + 1;
        const int output_width = input_width - kernel_size_ + 1;

        Tensor output(batch_size, out_channels_, output_height, output_width);

        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels_; ++oc) {
                for (int oy = 0; oy < output_height; ++oy) {
                    for (int ox = 0; ox < output_width; ++ox) {
                        float sum = 0;
                        for (int ic = 0; ic < in_channels_; ++ic) {
                            for (int ky = 0; ky < kernel_size_; ++ky) {
                                for (int kx = 0; kx < kernel_size_; ++kx) {
                                    sum += input.data(b, ic, oy + ky, ox + kx) * weights_.data(ky * kernel_size_ + kx + ic * kernel_size_ * kernel_size_, oc);
                                }
                            }
                        }
                        output.data(b, oc, oy, ox) = sum + bias_.data(0, oc);
                    }
                }
            }
        }

        return output;
    }

    // TODO: Implement backward pass

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    Tensor weights_;
    Tensor bias_;
};
