#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../tensor/Tensor.hpp"
#include "Module.hpp"

class EIGEN_ALIGN16 Conv2d : public Module
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Conv2d(int in_channels, int out_channels, int kernel_size, int max_batch_size = 64,
           bool use_parallel = true)
        : in_channels_(in_channels),
          out_channels_(out_channels),
          kernel_size_(kernel_size),
          max_batch_size_(max_batch_size),
          use_parallel_(use_parallel),
          weights_(nn::Tensor(static_cast<Eigen::Index>(kernel_size) * kernel_size * in_channels,
                              out_channels)),
          bias_(nn::Tensor(out_channels, 1)),
          im2col_buffer_(
              std::make_unique<nn::Tensor>(in_channels * kernel_size * kernel_size, max_batch_size * 256 * 256)),
          col2im_buffer_(std::make_unique<nn::Tensor>(max_batch_size, in_channels, 512, 512)),
          grad_output_buffer_(std::make_unique<nn::Tensor>())
    {
        // Initialize weights with He initialization
        initialize_weights_he();

        // Warm up indices computation with default size
        constexpr int DEFAULT_SIZE = 32;
        compute_indices_once(DEFAULT_SIZE, DEFAULT_SIZE);
    }

    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        const auto batch_size = static_cast<int>(input.get_shape()[0]);
        const auto input_height = static_cast<int>(input.get_shape()[2]);
        const auto input_width = static_cast<int>(input.get_shape()[3]);

        // Cache input for backward pass
        input_cache_ = input;

        // Compute output dimensions
        const int output_height = input_height - kernel_size_ + 1;
        const int output_width = input_width - kernel_size_ + 1;

        // Ensure indices are precomputed
        compute_indices_once(input_height, input_width);

        // 1. Perform optimized im2col
        const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
        const int patch_cols_per_batch = output_height * output_width;
        const int total_patch_cols = batch_size * patch_cols_per_batch;

        // Resize buffer if needed (rare after initial preallocation)
        if (im2col_buffer_->rows() < patch_rows || im2col_buffer_->cols() < total_patch_cols) [[unlikely]]
        {
            im2col_buffer_ = std::make_unique<nn::Tensor>(patch_rows, total_patch_cols);
        }

        auto& im2col_tensor = *im2col_buffer_;

        // Use optimized im2col with precomputed indices
        im2col_optimized(input,
                         im2col_tensor,
                         batch_size,
                         input_height,
                         input_width,
                         output_height,
                         output_width);

        // 2. Map to Eigen matrices for efficient operations
        Eigen::Map<const Eigen::MatrixXf> im2col_mapped(
            im2col_tensor.get_data_ref().data(), patch_rows, total_patch_cols);

        Eigen::Map<const Eigen::MatrixXf> weights_mapped(
            weights_.get_data_ref().data(), patch_rows, out_channels_);

        // 3. Perform matrix multiplication (main computation)
        // Use Eigen's lazy evaluation with noalias() to avoid temporary
        Eigen::MatrixXf output_2d(out_channels_, total_patch_cols);
        output_2d.noalias() = weights_mapped.transpose() * im2col_mapped;

        // 4. Add bias using optimized broadcasting
        add_bias_optimized(output_2d, bias_, total_patch_cols);

        // 5. Reshape output efficiently
        nn::Tensor output =
            reshape_output_optimized(output_2d, batch_size, output_height, output_width);

        return output;
    }

    // Backward pass: compute gradients w.r.t. input, weights and bias
    // Implements full backpropagation through convolution kernels.
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        const auto batch_size = static_cast<int>(input_cache_.get_shape()[0]);
        const auto input_height = static_cast<int>(input_cache_.get_shape()[2]);
        const auto input_width = static_cast<int>(input_cache_.get_shape()[3]);
        const int output_height = input_height - kernel_size_ + 1;
        const int output_width = input_width - kernel_size_ + 1;

        const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
        const int patch_cols_per_batch = output_height * output_width;
        const int total_patch_cols = batch_size * patch_cols_per_batch;

        // 1. Map grad_output to 2D matrix
        Eigen::Map<const Eigen::MatrixXf> grad_output_mapped(
            grad_output.get_data_ref().data(), out_channels_, total_patch_cols);

        // 2. Compute bias gradient (sum over all positions and batches)
        bias_.get_grad_ref().col(0) = grad_output_mapped.rowwise().sum();

        // 3. Get im2col of cached input
        auto& im2col_buffer = *im2col_buffer_;
        if (im2col_buffer.rows() < patch_rows || im2col_buffer.cols() < total_patch_cols) [[unlikely]]
        {
            im2col_buffer = nn::Tensor(patch_rows, total_patch_cols);
        }

        im2col_optimized(input_cache_,
                         im2col_buffer,
                         batch_size,
                         input_height,
                         input_width,
                         output_height,
                         output_width);

        Eigen::Map<const Eigen::MatrixXf> im2col_mapped(
            im2col_buffer.get_data_ref().data(), patch_rows, total_patch_cols);

        // 4. Compute weights gradient: dW = im2col_input * dY^T
        // Use parallelization if enabled
        Eigen::MatrixXf& weights_grad = weights_.get_grad_ref();

        if (use_parallel_ && total_patch_cols > 1000)
        {
// Parallel matrix multiplication for large problems
#pragma omp parallel for if (use_parallel_)
            for (int i = 0; i < patch_rows; ++i)
            {
                for (int j = 0; j < out_channels_; ++j)
                {
                    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
                    for (int k = 0; k < total_patch_cols; ++k)
                    {
                        sum += im2col_mapped(i, k) * grad_output_mapped(j, k);
                    }
                    weights_grad(i, j) = sum;
                }
            }
        }
        else
        {
            // Sequential for small problems
            weights_grad.noalias() = im2col_mapped * grad_output_mapped.transpose();
        }

        // 5. Map weights for input gradient computation
        Eigen::Map<const Eigen::MatrixXf> weights_mapped(
            weights_.get_data_ref().data(), patch_rows, out_channels_);

        // 6. Compute input gradient: dX_col = W * dY
        Eigen::MatrixXf d_input_col = weights_mapped * grad_output_mapped;

        // 7. Convert col2im (input gradient)
        nn::Tensor grad_input = col2im_optimized(
            d_input_col, batch_size, input_height, input_width, output_height, output_width);

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

    // Configuration getters
    bool is_parallel_enabled() const
    {
        return use_parallel_;
    }
    void set_parallel_enabled(bool enabled)
    {
        use_parallel_ = enabled;
    }

   private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int max_batch_size_;
    bool use_parallel_;
    mutable bool indices_computed_ = false;

    nn::Tensor weights_;
    nn::Tensor bias_;
    nn::Tensor input_cache_;

    // Pre-allocated buffers (avoid dynamic allocation in hot paths)
    std::unique_ptr<nn::Tensor> im2col_buffer_;
    std::unique_ptr<nn::Tensor> col2im_buffer_;
    std::unique_ptr<nn::Tensor> grad_output_buffer_;

    // Hash function for pair<int, int> for use in unordered_map
    struct PairHash
    {
        template <typename T, typename U>
        std::size_t operator()(const std::pair<T, U>& p) const
        {
            return std::hash<T>()(p.first) ^ (std::hash<U>()(p.second) << 1);
        }
    };

    /**
     * @brief Precompute im2col indices for given input dimensions
     */
    struct PatchIndices
    {
        std::vector<float> values;                  // For im2col: float values
        std::vector<std::pair<int, int>> positions; // For col2im: (row, col) positions
    };

    using IndexCache = std::unordered_map<std::pair<int, int>, std::vector<PatchIndices>, PairHash>;
    mutable IndexCache index_cache_;
    mutable std::mutex cache_mutex_;

    const std::vector<PatchIndices>& get_or_compute_indices(int input_height, int input_width) const
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

    std::vector<PatchIndices> compute_indices(int input_height, int input_width) const
    {
        const int output_height = input_height - kernel_size_ + 1;
        const int output_width = input_width - kernel_size_ + 1;
        const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
        const int total_patches = max_batch_size_ * output_height * output_width;

        std::vector<PatchIndices> indices(total_patches);

// Precompute all indices
#pragma omp parallel for collapse(2) if (use_parallel_)
        for (int b = 0; b < max_batch_size_; ++b)
        {
            for (int oy = 0; oy < output_height; ++oy)
            {
                for (int ox = 0; ox < output_width; ++ox)
                {
                    const int patch_idx =
                        (b * output_height * output_width) + (oy * output_width) + ox;
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

    void compute_indices_once(int input_height, int input_width) const
    {
        if (!indices_computed_)
        {
            get_or_compute_indices(input_height, input_width);
            indices_computed_ = true;
        }
    }

    /**
     * @brief Optimized im2col using precomputed indices and block operations
     */
    void im2col_optimized(const nn::Tensor& input, nn::Tensor& output, int batch_size,
                          int input_height, int input_width, int output_height,
                          int output_width) const
    {
        const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
        const int patch_cols_per_batch = output_height * output_width;

        // Map output for direct write access
        Eigen::Map<Eigen::MatrixXf> output_map(output.get_data_ref().data(),
                                               patch_rows,
                                               static_cast<Eigen::Index>(batch_size) *
                                                   static_cast<Eigen::Index>(patch_cols_per_batch));

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

    /**
     * @brief Optimized col2im using precomputed indices
     */
    nn::Tensor col2im_optimized(const Eigen::MatrixXf& cols, int batch_size, int input_height,
                                int input_width, int output_height, int output_width) const
    {
        const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
        const int patch_cols_per_batch = output_height * output_width;

        // Use pre-allocated buffer
        auto& result = *col2im_buffer_;
        if (result.get_shape()[0] != batch_size || result.get_shape()[2] != input_height ||
            result.get_shape()[3] != input_width) [[unlikely]]
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
                    const auto& patch_positions =
                        indices_vec[(b * patch_cols_per_batch) + p].positions;

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
                    const auto& patch_positions =
                        indices_vec[(b * patch_cols_per_batch) + p].positions;

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

    /**
     * @brief Optimized bias addition using SIMD-friendly operations
     */
    void add_bias_optimized(Eigen::MatrixXf& matrix, const nn::Tensor& bias, int num_cols) const
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

    /**
     * @brief Efficient output reshaping without nested loops
     */
    nn::Tensor reshape_output_optimized(const Eigen::MatrixXf& matrix, int batch_size,
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

    /**
     * @brief Initialize weights using He initialization for ReLU networks
     */
    void initialize_weights_he()
    {
        const int fan_in = in_channels_ * kernel_size_ * kernel_size_;
        const float stddev = std::sqrt(2.0F / static_cast<float>(fan_in));

        Eigen::Map<Eigen::MatrixXf> weights_map(
            weights_.get_data_ref().data(), weights_.rows(), weights_.cols());

        // Generate random normal distribution
        weights_map = Eigen::MatrixXf::Random(weights_.rows(), weights_.cols()) * stddev;

        // Initialize bias to small positive values
        bias_.get_data_ref().setConstant(0.01F);
    }
};
