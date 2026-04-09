/**
 * @file Conv2d_utils.cpp
 * @brief Implementation details for `Conv2d` (index caching, im2col/col2im helpers).
 */

#include "nn/layers/Conv2d.hpp"

// ============ Index Caching & Computation ============

template <typename Backend>
auto Conv2dImpl<Backend>::get_or_compute_indices(int batch_size, int input_height, int input_width) const
    -> const std::vector<Conv2dUtils::PatchIndices>&
{
    auto key = std::make_tuple(batch_size, input_height, input_width);

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = index_cache_.find(key);
        if (it != index_cache_.end()) [[likely]] // LCOV_EXCL_LINE
        {
            return it->second; // LCOV_EXCL_LINE
        }
    }

    // Compute indices if not in cache
    auto indices = compute_indices(batch_size, input_height, input_width);

    std::lock_guard<std::mutex> lock(cache_mutex_);
    return index_cache_[key] = std::move(indices);
} // LCOV_EXCL_LINE

template <typename Backend>
auto Conv2dImpl<Backend>::compute_indices(int batch_size, int input_height, int input_width) const
    -> std::vector<Conv2dUtils::PatchIndices>
{
    const int dilated_kernel_size = dilation_ * (kernel_size_ - 1) + 1;
    const int output_height = (input_height + 2 * padding_ - dilated_kernel_size) / stride_ + 1;
    const int output_width = (input_width + 2 * padding_ - dilated_kernel_size) / stride_ + 1;
    const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
    const int total_patches = batch_size * output_height * output_width; // Use actual batch_size

    std::vector<Conv2dUtils::PatchIndices> indices(total_patches);

// Precompute all indices
#pragma omp parallel for collapse(2) if (use_parallel_)
    for (int b = 0; b < batch_size; ++b) // Loop iterates up to actual batch_size
    {
        for (int oy = 0; oy < output_height; ++oy)
        {
            for (int ox = 0; ox < output_width; ++ox)
            {
                const int patch_idx = (b * output_height * output_width) + (oy * output_width) + ox;
                auto& patch = indices[patch_idx];
                patch.values.resize(patch_rows);
                patch.positions.resize(patch_rows);

                int elem_idx = 0;
                for (int ic = 0; ic < in_channels_; ++ic)
                {
                    for (int ky = 0; ky < kernel_size_; ++ky)
                    {
                        for (int kx = 0; kx < kernel_size_; ++kx)
                        {
                            const int input_y = oy * stride_ + ky * dilation_ - padding_;
                            const int input_x = ox * stride_ + kx * dilation_ - padding_;

                            // For im2col: store reference to value
                            // This will be filled during im2col_optimized
                            patch.values[elem_idx] = 0.0F;

                            // For col2im: store output position
                            const int result_row =
                                (((b * in_channels_) + ic) * input_height) + input_y;
                            const int result_col = input_x;
                            patch.positions[elem_idx] = {result_row, result_col};

                            ++elem_idx;
                        }
                    }
                }
            }
        }
    }
    return indices;
} // LCOV_EXCL_LINE

template <typename Backend>
void Conv2dImpl<Backend>::compute_indices_once(int batch_size, int input_height, int input_width) const
{
    if (!indices_computed_) [[unlikely]]
    {
        get_or_compute_indices(batch_size, input_height, input_width);
        indices_computed_ = true;
    }
} // LCOV_EXCL_LINE

// ============ Image-to-Column (im2col) Transformation ============

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
template <typename Backend>
void Conv2dImpl<Backend>::im2col_optimized(const typename Conv2dImpl<Backend>::Tensor& input,
    typename Conv2dImpl<Backend>::Tensor& output,
    int batch_size,
    int input_height,
    int input_width,
    int output_height,
    int output_width) const
{
    const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
    const int patch_cols_per_batch = output_height * output_width;

    const bool parallel_heavy = (batch_size * patch_cols_per_batch > 1000);
    if (use_parallel_ && parallel_heavy) [[likely]]
    {
// Parallel version for large problems
#pragma omp parallel for collapse(2) if (use_parallel_)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int p = 0; p < patch_cols_per_batch; ++p)
            {
                const int col_idx = (b * patch_cols_per_batch) + p;
                const int oy = p / output_width;
                const int ox = p % output_width;

                int elem_idx = 0;
                for (int ic = 0; ic < in_channels_; ++ic)
                {
                    for (int ky = 0; ky < kernel_size_; ++ky)
                    {
                        for (int kx = 0; kx < kernel_size_; ++kx)
                        {
                            const int input_y = oy * stride_ + ky * dilation_ - padding_;
                            const int input_x = ox * stride_ + kx * dilation_ - padding_;
                            float value = 0.0F;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 &&
                                input_x < input_width)
                            {
                                value = input.at(b, ic, input_y, input_x);
                            }
                            output.at(elem_idx, col_idx) = value;
                            elem_idx++;
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Sequential version for small problems
        for (int b = 0; b < batch_size; ++b)
        {
            for (int p = 0; p < patch_cols_per_batch; ++p)
            {
                const int col_idx = (b * patch_cols_per_batch) + p;
                const int oy = p / output_width;
                const int ox = p % output_width;

                int elem_idx = 0;
                for (int ic = 0; ic < in_channels_; ++ic)
                {
                    for (int ky = 0; ky < kernel_size_; ++ky)
                    {
                        for (int kx = 0; kx < kernel_size_; ++kx)
                        {
                            const int input_y = oy * stride_ + ky * dilation_ - padding_;
                            const int input_x = ox * stride_ + kx * dilation_ - padding_;
                            float value = 0.0F;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 &&
                                input_x < input_width)
                            {
                                value = input.at(b, ic, input_y, input_x);
                            }
                            output.at(elem_idx, col_idx) = value;
                            elem_idx++;
                        }
                    }
                }
            }
        }
    }
}

// ============ Column-to-Image (col2im) Transformation ============

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
template <typename Backend>
auto Conv2dImpl<Backend>::col2im_optimized(const typename Conv2dImpl<Backend>::Tensor& cols,
    int batch_size,
    int input_height,
    int input_width,
    int output_height,
    int output_width) const -> typename Conv2dImpl<Backend>::Tensor
{
    const int patch_cols_per_batch = output_height * output_width;

    // Use pre-allocated buffer
    auto& result = *col2im_buffer_;
    if (result.get_shape()[0] != batch_size || result.get_shape()[2] != input_height ||
        result.get_shape()[3] != input_width)
    {
        result = nn::Tensor(batch_size, in_channels_, input_height, input_width);
        result.setZero();
    }
    else
    {
        result.setZero();
    }

    const bool parallel_heavy = (batch_size * patch_cols_per_batch > 1000);
    if (use_parallel_ && parallel_heavy) [[likely]]
    {
// Parallel accumulation with improved loop structure for cache efficiency
#pragma omp parallel for collapse(2) if (use_parallel_)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int ic = 0; ic < in_channels_; ++ic)
            {
                // Process each input channel block separately to improve cache locality
                const int channel_row_offset = ic * kernel_size_ * kernel_size_;

                for (int oy = 0; oy < output_height; ++oy)
                {
                    for (int ox = 0; ox < output_width; ++ox)
                    {
                        const int p = (oy * output_width) + ox;
                        const int col_idx = (b * patch_cols_per_batch) + p;

                        for (int ky = 0; ky < kernel_size_; ++ky)
                        {
                            for (int kx = 0; kx < kernel_size_; ++kx)
                            {
                                const int input_y = oy * stride_ + ky * dilation_ - padding_;
                                const int input_x = ox * stride_ + kx * dilation_ - padding_;
                                const int elem_idx = channel_row_offset + (ky * kernel_size_) + kx;

                                if (input_y >= 0 && input_y < input_height && input_x >= 0 &&
                                    input_x < input_width)
                                {
                                    result.at(b, ic, input_y, input_x) += cols(elem_idx, col_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Sequential accumulation with improved loop structure
        for (int b = 0; b < batch_size; ++b)
        {
            for (int ic = 0; ic < in_channels_; ++ic)
            {
                const int channel_row_offset = ic * kernel_size_ * kernel_size_;

                for (int oy = 0; oy < output_height; ++oy)
                {
                    for (int ox = 0; ox < output_width; ++ox)
                    {
                        const int p = (oy * output_width) + ox;
                        const int col_idx = (b * patch_cols_per_batch) + p;

                        for (int ky = 0; ky < kernel_size_; ++ky)
                        {
                            for (int kx = 0; kx < kernel_size_; ++kx)
                            {
                                const int input_y = oy * stride_ + ky * dilation_ - padding_;
                                const int input_x = ox * stride_ + kx * dilation_ - padding_;
                                const int elem_idx = channel_row_offset + (ky * kernel_size_) + kx;

                                if (input_y >= 0 && input_y < input_height && input_x >= 0 &&
                                    input_x < input_width)
                                {
                                    result.at(b, ic, input_y, input_x) += cols(elem_idx, col_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

// ============ Bias Addition ============

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
template <typename Backend>
void Conv2dImpl<Backend>::add_bias_optimized(
    typename Conv2dImpl<Backend>::Tensor& matrix, const typename Conv2dImpl<Backend>::Tensor& bias, [[maybe_unused]] int num_cols) const
{
    // Support bias stored either as (out_channels, 1) or as (1, out_channels).
    // Use element accessors to avoid layout assumptions about Tensor backend storage.
    const auto bias_size = (bias.rows() == matrix.rows() && bias.cols() >= 1) ? bias.rows()
                           : (bias.cols() == matrix.rows() && bias.rows() >= 1)
                               ? bias.cols()
                               : std::min(matrix.rows(), bias.size());

    const auto n_rows = matrix.rows();
    const auto n_cols = matrix.cols();

    // Parallel version for large problems
    if (use_parallel_ && num_cols > 1000)
    {
#pragma omp parallel for if (use_parallel_)
        for (nn::Index i = 0; i < n_rows; ++i)
        {
            const nn::Index bias_idx = (bias_size > 0) ? (i % bias_size) : 0;
            const float bias_val = (bias.rows() == 1) ? bias.at(0, bias_idx) : bias.at(bias_idx, 0);
            for (nn::Index j = 0; j < n_cols; ++j)
            {
                matrix.at(i, j) += bias_val;
            }
        }
    }
    else
    {
        // Sequential version
        for (nn::Index i = 0; i < n_rows; ++i)
        {
            const nn::Index bias_idx = (bias_size > 0) ? (i % bias_size) : 0;
            const float bias_val = (bias.rows() == 1) ? bias.at(0, bias_idx) : bias.at(bias_idx, 0);
            for (nn::Index j = 0; j < n_cols; ++j)
            {
                matrix.at(i, j) += bias_val;
            }
        }
    }
}

// ============ Output Reshaping ============

template <typename Backend>
auto Conv2dImpl<Backend>::reshape_output_optimized(
    const typename Conv2dImpl<Backend>::Tensor& matrix, int batch_size, int output_height, int output_width) const
    -> typename Conv2dImpl<Backend>::Tensor
{
    // Create output tensor with correct shape
    nn::Tensor output(batch_size, out_channels_, output_height, output_width);

    const int patch_cols = output_height * output_width;

    // We need to copy from matrix (C, B*H*W) to output (B, C, H, W)
    // matrix(c, col_idx) corresponds to output(b, c, oy, ox)
    // where col_idx = b * (H*W) + oy * W + ox

    if (use_parallel_)
    {
#pragma omp parallel for collapse(2) if (use_parallel_)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int c = 0; c < out_channels_; ++c)
            {
                for (int oy = 0; oy < output_height; ++oy)
                {
                    for (int ox = 0; ox < output_width; ++ox)
                    {
                        const int col_idx = b * patch_cols + oy * output_width + ox;
                        output.at(b, c, oy, ox) = matrix.at(c, col_idx);
                    }
                }
            }
        }
    }
    else
    {
        for (int b = 0; b < batch_size; ++b)
        {
            for (int c = 0; c < out_channels_; ++c)
            {
                for (int oy = 0; oy < output_height; ++oy)
                {
                    for (int ox = 0; ox < output_width; ++ox)
                    {
                        const int col_idx = b * patch_cols + oy * output_width + ox;
                        output.at(b, c, oy, ox) = matrix.at(c, col_idx);
                    }
                }
            }
        }
    }

    // LCOV_EXCL_START
    return output;
}
// LCOV_EXCL_STOP

// Explicit instantiation: generate code for the Eigen (CPU) backend only.
template class Conv2dImpl<nn::EigenTensorBackend>;
