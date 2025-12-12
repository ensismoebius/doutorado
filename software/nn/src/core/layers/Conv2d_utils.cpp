#include <cstring>

#include "Conv2d.hpp"

// ============ Index Caching & Computation ============

const std::vector<Conv2dImpl::PatchIndices>& Conv2d::get_or_compute_indices(int input_height,
                                                                            int input_width) const
{
    auto key = std::make_pair(input_height, input_width);

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = index_cache_.find(key);
        if (it != index_cache_.end())
        {
            return it->second;
        }
    }

    // Compute indices if not in cache
    auto indices = compute_indices(input_height, input_width);

    std::lock_guard<std::mutex> lock(cache_mutex_);
    return index_cache_[key] = std::move(indices);
}

std::vector<Conv2dImpl::PatchIndices> Conv2d::compute_indices(int input_height,
                                                              int input_width) const
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
    if (!indices_computed_)
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

    // Map output for direct write access
    Eigen::Map<Eigen::MatrixXf> output_map(
        output.get_data_ref().data(),
        patch_rows,
        static_cast<Eigen::Index>(batch_size) * static_cast<Eigen::Index>(patch_cols_per_batch));

    if (use_parallel_ && batch_size * patch_cols_per_batch > 1000)
    {
// Parallel version for large problems
#pragma omp parallel for collapse(2) if (use_parallel_)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int p = 0; p < patch_cols_per_batch; ++p)
            {
                const int col_idx = (b * patch_cols_per_batch) + p;
                const int oy = p / output_width; // Current output y
                const int ox = p % output_width; // Current output x

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
                const int oy = p / output_width; // Current output y
                const int ox = p % output_width; // Current output x

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
nn::Tensor Conv2d::col2im_optimized(const Eigen::MatrixXf& cols, int batch_size, int input_height,
                                    int input_width, int output_height, int output_width) const
{
    const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
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

    // Get precomputed indices
    const auto& indices_vec = get_or_compute_indices(input_height, input_width);

    if (use_parallel_ && batch_size * patch_cols_per_batch > 1000)
    {
// Parallel accumulation
#pragma omp parallel for collapse(2) if (use_parallel_)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int p = 0; p < patch_cols_per_batch; ++p)
            {
                const int col_idx = (b * patch_cols_per_batch) + p;
                const auto& patch_positions = indices_vec[(b * patch_cols_per_batch) + p].positions;

                for (int r = 0; r < patch_rows; ++r)
                {
                    const float value = cols(r, col_idx);
                    // Add to appropriate position using precomputed mapping
                    const auto& pos = patch_positions[r];

                    // We need to map pos.first (linearized index) back to 4D coordinates
                    // The original compute_indices creates a linearized index result_row and
                    // result_col result_row = (b * in_channels_ + ic) * input_height + input_y;
                    // result_col = input_x;
                    // This means pos.first is a combination of b, ic, input_y.
                    // We need to reverse this to get b_orig, ic_orig, input_y_orig
                    // input_x_orig is pos.second

                    const int linearized_b_ic_y = pos.first;
                    const int x_orig = pos.second; // input_x

                    const int y_orig = linearized_b_ic_y % input_height;
                    const int b_ic = linearized_b_ic_y / input_height;
                    const int ic_orig = b_ic % in_channels_;
                    const int b_orig = b_ic / in_channels_;

#pragma omp atomic
                    result.at(b_orig, ic_orig, y_orig, x_orig) += value;
                }
            }
        }
    }
    else
    {
        // Sequential accumulation
        for (int b = 0; b < batch_size; ++b)
        {
            for (int p = 0; p < patch_cols_per_batch; ++p)
            {
                const int col_idx = (b * patch_cols_per_batch) + p;
                const auto& patch_positions = indices_vec[(b * patch_cols_per_batch) + p].positions;

                for (int r = 0; r < patch_rows; ++r)
                {
                    const float value = cols(r, col_idx);
                    const auto& pos = patch_positions[r];

                    const int linearized_b_ic_y = pos.first;
                    const int x_orig = pos.second; // input_x

                    const int y_orig = linearized_b_ic_y % input_height;
                    const int b_ic = linearized_b_ic_y / input_height;
                    const int ic_orig = b_ic % in_channels_;
                    const int b_orig = b_ic / in_channels_;

                    result.at(b_orig, ic_orig, y_orig, x_orig) += value;
                }
            }
        }
    }

    return result;
}

// ============ Bias Addition ============

void Conv2d::add_bias_optimized(Eigen::MatrixXf& matrix, const nn::Tensor& bias, int num_cols) const
{
    const Eigen::VectorXf bias_vector = bias.get_data_ref().col(0);

    if (use_parallel_ && num_cols > 1000)
    {
#pragma omp parallel for if (use_parallel_)
        for (int i = 0; i < matrix.rows(); ++i)
        {
            const float bias_val = bias_vector(i);
            Eigen::VectorXf::Map(matrix.row(i).data(), num_cols).array() += bias_val;
        }
    }
    else
    {
        // Use Eigen's broadcasting with noalias for efficiency
        matrix.noalias() += bias_vector * Eigen::RowVectorXf::Ones(num_cols);
    }
}

// ============ Output Reshaping ============

nn::Tensor Conv2d::reshape_output_optimized(const Eigen::MatrixXf& matrix, int batch_size,
                                            int output_height, int output_width) const
{
    const int total_elements = batch_size * out_channels_ * output_height * output_width;

    // Create output tensor with correct shape
    nn::Tensor output(batch_size, out_channels_, output_height, output_width);

    // Direct memory mapping for efficient copy
    Eigen::Map<Eigen::VectorXf> output_map(output.get_data_ref().data(), total_elements);

    Eigen::Map<const Eigen::VectorXf> input_map(matrix.data(), total_elements);

    // Single memcpy for all data (fastest possible)
    if (total_elements > 0)
    {
        std::memcpy(output_map.data(), input_map.data(), total_elements * sizeof(float));
    }

    return output;
}
