#pragma once

#include "Conv2d.hpp"
#include "Module.hpp"
#include "ReLU.hpp"

class ResNetBlock : public Module
{
   public:
    ResNetBlock(int in_channels, int out_channels)
        : conv1_(in_channels, out_channels, 3), relu_(), conv2_(out_channels, out_channels, 3)
    {
    }

    nn::Tensor forward(const nn::Tensor& input) override
    {
        nn::Tensor output = conv1_.forward(input);
        output = relu_.forward(output);
        output = conv2_.forward(output);

        // Add skip connection with shape alignment
        // Handles dimension mismatches by zero-padding/cropping overlapping regions
        const auto out_shape = output.get_shape();
        const auto in_shape = input.get_shape();
        if (out_shape == in_shape) [[likely]]
        {
            output.get_data_ref() = output.get_data_ref() + input.get_data_ref();
        }
        else
        {
            // Create zero matrix matching output underlying data and copy overlapping region
            Eigen::MatrixXf aligned =
                Eigen::MatrixXf::Zero(output.get_data_ref().rows(), output.get_data_ref().cols());

            const auto& in_mat = input.get_data_ref();
            const auto& out_mat = output.get_data_ref();

            const auto rows_copy = static_cast<int>(std::min(in_mat.rows(), aligned.rows()));
            const auto cols_copy = static_cast<int>(std::min(in_mat.cols(), aligned.cols()));

            aligned.block(0, 0, rows_copy, cols_copy) = in_mat.block(0, 0, rows_copy, cols_copy);
            output.get_data_ref() = out_mat + aligned;
        }

        return relu_.forward(output);
    }

   private:
    Conv2d conv1_;
    ReLU relu_;
    Conv2d conv2_;
};
