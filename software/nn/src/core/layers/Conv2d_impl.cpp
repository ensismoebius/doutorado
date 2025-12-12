#include "Conv2d.hpp"

// ============ Constructor ============

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int max_batch_size,
               bool use_parallel)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      max_batch_size_(max_batch_size),
      use_parallel_(use_parallel),
      weights_(nn::Tensor(static_cast<Eigen::Index>(kernel_size) * kernel_size * in_channels,
                          out_channels)),
      bias_(nn::Tensor(1, out_channels))
{
    // Initialize weights and bias
    // Pre-allocate buffers with maximum expected size
    const int max_patch_rows = in_channels * kernel_size * kernel_size;
    const int max_output_pixels = 256 * 256; // Max 256x256 output

    im2col_buffer_ =
        std::make_unique<nn::Tensor>(max_patch_rows, max_batch_size * max_output_pixels);

    col2im_buffer_ =
        std::make_unique<nn::Tensor>(max_batch_size, in_channels, 512, 512); // Max 512x512 input

    grad_output_buffer_ = std::make_unique<nn::Tensor>();

    // Initialize weights with He initialization
    initialize_weights_he();

    // Warm up indices computation with default size
    constexpr int DEFAULT_SIZE = 32;
    compute_indices_once(DEFAULT_SIZE, DEFAULT_SIZE);
}

// ============ Forward & Backward Passes ============

auto Conv2d::forward(const nn::Tensor& input) -> nn::Tensor
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
    const bool need_resize =
        (im2col_buffer_->rows() < patch_rows || im2col_buffer_->cols() < total_patch_cols);
    if (need_resize) [[unlikely]]
    {
        im2col_buffer_ = std::make_unique<nn::Tensor>(patch_rows, total_patch_cols);
    }

    auto& im2col_tensor = *im2col_buffer_;

    // Use optimized im2col with precomputed indices
    im2col_optimized(
        input, im2col_tensor, batch_size, input_height, input_width, output_height, output_width);

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
    add_bias_optimized(output_2d, bias_);

    // 5. Reshape output efficiently
    nn::Tensor output =
        reshape_output_optimized(output_2d, batch_size, output_height, output_width);

    return output;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
auto Conv2d::backward(const nn::Tensor& grad_output) -> nn::Tensor
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
    if (im2col_buffer.rows() < patch_rows || im2col_buffer.cols() < total_patch_cols)
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

    const bool large_mm = (total_patch_cols > 1000);
    if (use_parallel_ && large_mm) [[likely]]
    {
// Parallel matrix multiplication for large problems
#pragma omp parallel for if (use_parallel_)
        for (int i = 0; i < patch_rows; ++i)
        {
            for (int j = 0; j < out_channels_; ++j)
            {
                float sum = 0.0F;
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

// ============ Getters ============

auto Conv2d::get_weights() const -> const nn::Tensor&
{
    return weights_;
}

auto Conv2d::get_weights() -> nn::Tensor&
{
    return weights_;
}

auto Conv2d::get_bias() const -> const nn::Tensor&
{
    return bias_;
}

auto Conv2d::get_bias() -> nn::Tensor&
{
    return bias_;
}

// ============ Setters ============

void Conv2d::set_weights(const nn::Tensor& weights)
{
    weights_ = weights;
}

void Conv2d::set_bias(const nn::Tensor& bias)
{
    bias_ = bias;
}

bool Conv2d::is_parallel_enabled() const
{
    return use_parallel_;
}

void Conv2d::set_parallel_enabled(bool enabled)
{
    use_parallel_ = enabled;
}

// ============ Initialization ============

void Conv2d::initialize_weights_he()
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
