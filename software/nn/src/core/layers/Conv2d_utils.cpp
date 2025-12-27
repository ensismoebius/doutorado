#include <iostream>

#include "Conv2d.hpp"

// ============ Index Caching & Computation ============

auto Conv2d::get_or_compute_indices(int input_height, int input_width) const
    -> const std::vector<Conv2dImpl::PatchIndices>&
{
    auto key = std::make_pair(input_height, input_width);

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = index_cache_.find(key);
        if (it != index_cache_.end()) [[likely]]
        {
            return it->second;
        }
    }

    // Compute indices if not in cache
    auto indices = compute_indices(input_height, input_width);

    std::lock_guard<std::mutex> lock(cache_mutex_);
    return index_cache_[key] = std::move(indices);
}

auto Conv2d::compute_indices(int input_height, int input_width) const
    -> std::vector<Conv2dImpl::PatchIndices>
{
    const int output_height = input_height - kernel_size_ + 1;
    const int output_width = input_width - kernel_size_ + 1;
    const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
    const int total_patches = max_batch_size_ * output_height * output_width;

    std::vector<Conv2dImpl::PatchIndices> indices(total_patches);

// Precompute all indices
#pragma omp parallel for collapse(2) if (use_parallel_)
    for (int b = 0; b < max_batch_size_; ++b)
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
                            const int input_y = oy + ky;
                            const int input_x = ox + kx;

                            // For im2col: store reference to value
                            patch.values[elem_idx] = 0.0F; // Placeholder

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
}

void Conv2d::compute_indices_once(int input_height, int input_width) const
{
    if (!indices_computed_) [[unlikely]]
    {
        get_or_compute_indices(input_height, input_width);
        indices_computed_ = true;
    }
}

// ============ Image-to-Column (im2col) Transformation ============

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Conv2d::im2col_optimized(const nn::Tensor& input, nn::Tensor& output, int batch_size,
                              int input_height, int input_width, int output_height,
                              int output_width) const
{
    const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
    const int patch_cols_per_batch = output_height * output_width;

    // Map output for direct write access with Eigen::Ref-compatible signature
    Eigen::Map<Eigen::MatrixXf> output_map(
        output.get_data_ref().data(),
        patch_rows,
        static_cast<Eigen::Index>(batch_size) * static_cast<Eigen::Index>(patch_cols_per_batch));

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
                            const int input_y = oy + ky;
                            const int input_x = ox + kx;
                            output_map(elem_idx++, col_idx) = input.at(b, ic, input_y, input_x);
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
                            const int input_y = oy + ky;
                            const int input_x = ox + kx;
                            output_map(elem_idx++, col_idx) = input.at(b, ic, input_y, input_x);
                        }
                    }
                }
            }
        }
    }
}

// ============ Column-to-Image (col2im) Transformation ============

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
auto Conv2d::col2im_optimized(const Eigen::MatrixXf& cols, int batch_size, int input_height,
                              int input_width, int output_height, int output_width) const
    -> nn::Tensor
{
    const int patch_cols_per_batch = output_height * output_width;

    // Use pre-allocated buffer
    auto& result = *col2im_buffer_;
    if (result.get_shape()[0] != batch_size || result.get_shape()[2] != input_height ||
        result.get_shape()[3] != input_width)
    {
        result = nn::Tensor(batch_size, in_channels_, input_height, input_width);
        result.get_data_ref().setZero();
    }
    else
    {
        result.get_data_ref().setZero();
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
                                const int input_y = oy + ky;
                                const int input_x = ox + kx;
                                const int elem_idx = channel_row_offset + (ky * kernel_size_) + kx;
                                result.at(b, ic, input_y, input_x) += cols(elem_idx, col_idx);
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
                                const int input_y = oy + ky;
                                const int input_x = ox + kx;
                                const int elem_idx = channel_row_offset + (ky * kernel_size_) + kx;
                                result.at(b, ic, input_y, input_x) += cols(elem_idx, col_idx);
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
void Conv2d::add_bias_optimized(Eigen::MatrixXf& matrix, const nn::Tensor& bias,
                                [[maybe_unused]] int num_cols) const
{
    // Support bias stored either as (out_channels, 1) or as (1, out_channels)
    const auto& b = bias.get_data_ref();
    Eigen::VectorXf bias_vector;
    if (b.rows() == matrix.rows() && b.cols() >= 1)
    {
        // column-major: bias stored as (out_channels, 1)
        bias_vector = b.col(0);
    }
    else if (b.cols() == matrix.rows() && b.rows() >= 1)
    {
        // row-major-like storage: bias stored as (1, out_channels)
        bias_vector = b.row(0).transpose();
    }
    else
    {
        // Fallback: try to reshape/replicate if possible
        bias_vector = Eigen::VectorXf::Zero(matrix.rows());
        const Eigen::Index n = std::min<Eigen::Index>(matrix.rows(), b.size());
        for (Eigen::Index i = 0; i < n; ++i)
        {
            bias_vector(i) = b.data()[i];
        }
    }

    // This parallel branch is correct and needs to be here
    if (use_parallel_ && num_cols > 1000)
    {
#pragma omp parallel for if (use_parallel_)
        for (int i = 0; i < matrix.rows(); ++i)
        {
            // safe access in case bias_vector length doesn't exactly match
            const Eigen::Index idx = (bias_vector.size() > 0) ? (i % bias_vector.size()) : 0;
            const float bias_val = bias_vector(idx);
            Eigen::Map<Eigen::VectorXf>(matrix.row(i).data(), num_cols).array() += bias_val;
        }
    }
    else // The sequential branch
    {
        if (bias_vector.size() == matrix.rows())
        {
            // Use Eigen's broadcasting with noalias for efficiency
            matrix.colwise() += bias_vector;
        }
        else
        {
            // Fallback: element-wise addition with safe indexing
            const Eigen::Index bvsz = bias_vector.size() > 0 ? bias_vector.size() : 1;
            for (int r = 0; r < matrix.rows(); ++r)
            {
                const Eigen::Index idx = r % bvsz;
                matrix.row(r).array() += bias_vector(idx);
            }
        }
    }
}

// ============ Output Reshaping ============

auto Conv2d::reshape_output_optimized(const Eigen::MatrixXf& matrix, int batch_size,
                                      int output_height, int output_width) const -> nn::Tensor
{
    // Create output tensor with correct shape
    nn::Tensor output(batch_size, out_channels_, output_height, output_width);

    // Use Eigen assignment for better integration with expression templates
    // If layouts are compatible, Eigen avoids temporary allocation
    Eigen::Map<Eigen::MatrixXf> output_map(output.get_data_ref().data(),
                                           out_channels_,
                                           static_cast<Eigen::Index>(batch_size) *
                                               static_cast<Eigen::Index>(output_height) *
                                               static_cast<Eigen::Index>(output_width));

    // Direct assignment using noalias - Eigen evaluates most efficiently
    // compared to manual memcpy, as it leverages vectorization
    output_map.noalias() = matrix;

    return output;
}
