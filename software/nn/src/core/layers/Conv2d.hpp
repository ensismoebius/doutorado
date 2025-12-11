#pragma once

#include "../tensor/Tensor.hpp"
#include "Module.hpp"

class Conv2d : public Module
{
   public:
    Conv2d(int in_channels, int out_channels, int kernel_size)
        : in_channels_(in_channels),
          out_channels_(out_channels),
          kernel_size_(kernel_size),
          weights_(nn::Tensor(static_cast<Eigen::Index>(kernel_size) * kernel_size * in_channels,
                              out_channels)),
          bias_(nn::Tensor(1, out_channels))
    {
        // Initialize weights and bias
    }

    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        // Input shape: (batch_size, in_channels, input_height, input_width)
        // Output shape: (batch_size, out_channels, output_height, output_width)
        const auto batch_size = static_cast<int>(input.get_shape()[0]);
        const auto input_height = static_cast<int>(input.get_shape()[2]);
        const auto input_width = static_cast<int>(input.get_shape()[3]);
        const int output_height = input_height - kernel_size_ + 1;
        const int output_width = input_width - kernel_size_ + 1;

        // Cache input for backward
        input_cache_ = input;

        // 1. Transform input using im2col
        nn::Tensor im2col_input_tensor = im2col(input, kernel_size_, input_height, input_width, output_height, output_width);
        
        const int patch_rows = in_channels_ * kernel_size_ * kernel_size_; // C_in * K_H * K_W
        const int patch_cols_per_batch = output_height * output_width; // H_out * W_out
        const int total_patch_cols = batch_size * patch_cols_per_batch; // B * H_out * W_out

        // Map im2col_input_tensor's flattened data to its intended 2D shape for Eigen operations
        // The data in im2col_input_tensor is stored as (patch_rows * total_patch_cols, 1) column vector
        // We need to view it as (patch_rows, total_patch_cols) for Eigen matrix multiplication
        Eigen::Map<const Eigen::MatrixXf> im2col_input_mapped(
            im2col_input_tensor.get_data_ref().data(), // Pointer to the raw data
            patch_rows,                                // Number of rows in the 2D view
            total_patch_cols                           // Number of columns in the 2D view
        );

        // Map weights_ to its intended 2D shape
        // weights_ is (patch_rows, out_channels_), so its m_data is (patch_rows * out_channels_, 1)
        Eigen::Map<const Eigen::MatrixXf> weights_mapped(
            weights_.get_data_ref().data(),
            weights_.rows(),  // Number of rows in the 2D view (patch_rows)
            weights_.cols()   // Number of columns in the 2D view (out_channels_)
        );

        // 2. Perform matrix multiplication: output_2d = weights_transposed * im2col_input
        // weights_mapped is (patch_rows, out_channels_)
        // weights_mapped.transpose() is (out_channels_, patch_rows)
        // im2col_input_mapped is (patch_rows, total_patch_cols)
        // Result: (out_channels_, total_patch_cols)
        Eigen::MatrixXf output_2d_raw = weights_mapped.transpose() * im2col_input_mapped;

        // 3. Add bias
        // bias_ is (1, out_channels_). Its m_data is (out_channels_, 1) as a flattened column vector.
        // We need to add bias to each column of output_2d_raw.
        // output_2d_raw is (out_channels_, total_patch_cols)
        // bias_.get_data_ref() is (out_channels_, 1)
        output_2d_raw.colwise() += bias_.get_data_ref().col(0);


        // 4. Reshape the output_2d_raw back into a 4D nn::Tensor
        nn::Tensor output(batch_size, out_channels_, output_height, output_width);
        
        // The output_2d_raw is (out_channels_, total_patch_cols)
        // We need to copy this data into output's flattened m_data, which is (total_size, 1)
        // Iterating through the output tensor to fill its 4D shape from output_2d_raw
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels_; ++oc) {
                for (int oy = 0; oy < output_height; ++oy) {
                    for (int ox = 0; ox < output_width; ++ox) {
                        int col_idx = b * patch_cols_per_batch + oy * output_width + ox;
                        output.at(b, oc, oy, ox) = output_2d_raw(oc, col_idx);
                    }
                }
            }
        }
        
        return output;
    }

    // Backward pass: compute gradients w.r.t. input, weights and bias
    // Implements full backpropagation through convolution kernels.
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // shapes
        const auto batch_size = static_cast<int>(input_cache_.get_shape()[0]);
        const auto input_height = static_cast<int>(input_cache_.get_shape()[2]);
        const auto input_width = static_cast<int>(input_cache_.get_shape()[3]);
        const int output_height = input_height - kernel_size_ + 1;
        const int output_width = input_width - kernel_size_ + 1;

        // Prepare gradients
        // Zero grads for weights and bias
        weights_.get_grad_ref().setZero();
        bias_.get_grad_ref().setZero();

        nn::Tensor grad_input(batch_size, in_channels_, input_height, input_width);
        grad_input.get_data_ref().setZero();

        for (int b = 0; b < batch_size; ++b) [[likely]]
        {
            for (int oc = 0; oc < out_channels_; ++oc) [[likely]]
            {
                for (int oy = 0; oy < output_height; ++oy) [[likely]]
                {
                    for (int ox = 0; ox < output_width; ++ox) [[likely]]
                    {
                        float go = grad_output.at(b, oc, oy, ox);
                        // bias grad
                        bias_.get_grad_ref()(0, oc) += go;

                        for (int ic = 0; ic < in_channels_; ++ic) [[likely]]
                        {
                            for (int ky = 0; ky < kernel_size_; ++ky) [[likely]]
                            {
                                for (int kx = 0; kx < kernel_size_; ++kx) [[likely]]
                                {
                                    int in_y = oy + ky;
                                    int in_x = ox + kx;
                                    float inp = input_cache_.at(b, ic, in_y, in_x);
                                    // weight grad index
                                    int wrow = (ky * kernel_size_) + kx +
                                               (ic * kernel_size_ * kernel_size_);
                                    weights_.get_grad_ref()(wrow, oc) += inp * go;

                                    // input grad
                                    grad_input.at(b, ic, in_y, in_x) += weights_.at(wrow, oc) * go;
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    // Public getters for weights and bias
    const nn::Tensor& get_weights() const
    {
        return weights_;
    }
    nn::Tensor& get_weights()
    {
        return weights_;
    }
    const nn::Tensor& get_bias() const
    {
        return bias_;
    }
    nn::Tensor& get_bias()
    {
        return bias_;
    }

    // Public setters for weights and bias
    void set_weights(const nn::Tensor& weights)
    {
        weights_ = weights;
    }
    void set_bias(const nn::Tensor& bias)
    {
        bias_ = bias;
    }

   private:
    // Helper function to transform input patches into columns (im2col)
    nn::Tensor im2col(const nn::Tensor& input, int kernel_size, int input_height, int input_width,
                      int output_height, int output_width) const
    {
        const int batch_size = static_cast<int>(input.get_shape()[0]);
        const int in_channels = static_cast<int>(input.get_shape()[1]);

        const int patch_rows = in_channels * kernel_size * kernel_size; // C_in * K_H * K_W
        const int patch_cols_per_batch = output_height * output_width;  // H_out * W_out
        const int total_patch_cols = batch_size * patch_cols_per_batch; // B * H_out * W_out

        nn::Tensor cols_tensor(patch_rows, total_patch_cols); // Output im2col matrix

        for (int b = 0; b < batch_size; ++b)
        {
            for (int oy = 0; oy < output_height; ++oy)
            {
                for (int ox = 0; ox < output_width; ++ox)
                {
                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        for (int ky = 0; ky < kernel_size; ++ky)
                        {
                            for (int kx = 0; kx < kernel_size; ++kx)
                            {
                                int input_y = oy + ky;
                                int input_x = ox + kx;

                                // Calculate index in the output im2col matrix
                                // Col index: iterates through batch, then output height, then output width
                                int col_idx = b * patch_cols_per_batch + oy * output_width + ox;
                                // Row index: iterates through input channels, then kernel height, then kernel width
                                int row_idx = ic * kernel_size * kernel_size + ky * kernel_size + kx;

                                cols_tensor.at(row_idx, col_idx) = input.at(b, ic, input_y, input_x);
                            }
                        }
                    }
                }
            }
        }
        return cols_tensor;
    }

   private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    nn::Tensor weights_;
    nn::Tensor bias_;
    nn::Tensor input_cache_;
};
