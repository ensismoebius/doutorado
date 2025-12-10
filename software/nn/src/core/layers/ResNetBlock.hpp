#pragma once

#include "Module.hpp"
#include "Conv2d.hpp"
#include "ReLU.hpp"

class ResNetBlock : public Module {
public:
    ResNetBlock(int in_channels, int out_channels) :
        conv1_(in_channels, out_channels, 3),
        relu_(),
        conv2_(out_channels, out_channels, 3)
    {
    }

    nn::Tensor forward(const nn::Tensor& input) override {
        nn::Tensor output = conv1_.forward(input);
        output = relu_.forward(output);
        output = conv2_.forward(output);

        // Add skip connection
        // TODO: Handle the case where the input and output shapes are different
        output.get_data_ref() = output.get_data_ref() + input.get_data_ref();

        return relu_.forward(output);
    }

private:
    Conv2d conv1_;
    ReLU relu_;
    Conv2d conv2_;
};
