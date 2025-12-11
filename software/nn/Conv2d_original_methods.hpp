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

        const int patch_rows = in_channels_ * kernel_size_ * kernel_size_; // C_in * K_H * K_W
        const int patch_cols_per_batch = output_height * output_width; // H_out * W_out
        const int total_patch_cols = batch_size * patch_cols_per_batch; // B * H_out * W_out

        // Zero grads for weights and bias
        weights_.get_grad_ref().setZero();
        bias_.get_grad_ref().setZero();

        // 1. Reshape grad_output to (out_channels_, total_patch_cols)
        // grad_output's m_data is (total_elements, 1)
        Eigen::Map<const Eigen::MatrixXf> grad_output_reshaped_mapped(
            grad_output.get_data_ref().data(),
            out_channels_,
            total_patch_cols
        );
        
        // 2. Compute d_bias
        // bias_.get_grad_ref() is (out_channels_, 1)
        bias_.get_grad_ref().col(0) += grad_output_reshaped_mapped.rowwise().sum().transpose();

        // 3. Create im2col from input_cache_
        nn::Tensor im2col_input_cache_tensor = im2col(input_cache_, kernel_size_, input_height, input_width, output_height, output_width);
        Eigen::Map<const Eigen::MatrixXf> im2col_input_cache_mapped(
            im2col_input_cache_tensor.get_data_ref().data(),
            patch_rows,
            total_patch_cols
        );

        // 4. Map weights_ for d_input calculation
        Eigen::Map<const Eigen::MatrixXf> weights_mapped(
            weights_.get_data_ref().data(),
            patch_rows,
            out_channels_
        );

        // 5. Compute d_weights
        // weights_.get_grad_ref() is (patch_rows, out_channels_)
        weights_.get_grad_ref() += im2col_input_cache_mapped * grad_output_reshaped_mapped.transpose();

        // 6. Compute d_input
        // d_input_col_raw = weights_mapped * grad_output_reshaped
        Eigen::MatrixXf d_input_col_raw = weights_mapped * grad_output_reshaped_mapped;
        
        // Transform d_input_col_raw back to 4D grad_input using col2im
        nn::Tensor grad_input = col2im(nn::Tensor(d_input_col_raw), kernel_size_,
                                       input_height, input_width, output_height, output_width);
        
        return grad_input;
    }
