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

    // Helper function to transform columns back into image patches (col2im)
    nn::Tensor col2im(const nn::Tensor& cols_tensor, int kernel_size,
                      int input_height, int input_width, int output_height, int output_width) const
    {
        const int batch_size = static_cast<int>(input_cache_.get_shape()[0]);
        const int in_channels = static_cast<int>(input_cache_.get_shape()[1]);

        nn::Tensor img_tensor(batch_size, in_channels, input_height, input_width);
        img_tensor.get_data_ref().setZero(); // Initialize with zeros

        const int patch_cols_per_batch = output_height * output_width;  // H_out * W_out
        const int total_patch_cols = batch_size * patch_cols_per_batch; // B * H_out * W_out

        // Map cols_tensor's flattened data to its intended 2D shape for Eigen operations
        const int patch_rows = in_channels * kernel_size * kernel_size; // C_in * K_H * K_W
        Eigen::Map<const Eigen::MatrixXf> cols_mapped(
            cols_tensor.get_data_ref().data(), // Pointer to the raw data
            patch_rows,                         // Number of rows in the 2D view
            total_patch_cols                    // Number of columns in the 2D view
        );

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

                                int col_idx = b * patch_cols_per_batch + oy * output_width + ox;
                                int row_idx = ic * kernel_size * kernel_size + ky * kernel_size + kx;

                                img_tensor.at(b, ic, input_y, input_x) += cols_mapped(row_idx, col_idx);
                            }
                        }
                    }
                }
            }
        }
        return img_tensor;
    }
