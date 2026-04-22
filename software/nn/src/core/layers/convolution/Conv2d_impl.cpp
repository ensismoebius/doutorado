/**
 * @file Conv2d_impl.cpp
 * @brief Implementation of the `Conv2d` layer forward/backward paths.
 */

#include <cmath>
#include <random>

#include "nn/layers/convolution/Conv2d.hpp"

constexpr int DEFAULT_SIZE = 32;
constexpr int MAX_IMAGE_SIZE = 256;
constexpr int COL2IM_SIZE = 512;

// ============ Constructor ============

template <typename Backend>
Conv2dImpl<Backend>::Conv2dImpl(
    int in_channels, int out_channels, int kernel_size, int max_batch_size, bool use_parallel)
    : Conv2dImpl<Backend>(
          in_channels, out_channels, kernel_size, 1, 0, 1, use_parallel, max_batch_size)
{
}

template <typename Backend>
Conv2dImpl<Backend>::Conv2dImpl(int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
    : Conv2dImpl<Backend>(in_channels, out_channels, kernel_size, stride, padding, dilation, true, 64)
{
}

template <typename Backend>
Conv2dImpl<Backend>::Conv2dImpl(int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool use_parallel,
    int max_batch_size)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      max_batch_size_(max_batch_size),
      use_parallel_(use_parallel),
      weights_(
          nn::Tensor(static_cast<size_t>(kernel_size) * kernel_size * in_channels, out_channels)),
      bias_(nn::Tensor(1, out_channels)), // Bias should be 1 x out_channels
      im2col_buffer_(std::make_unique<nn::Tensor>(in_channels * kernel_size * kernel_size,
          max_batch_size * MAX_IMAGE_SIZE * MAX_IMAGE_SIZE)),
      col2im_buffer_(
          std::make_unique<nn::Tensor>(max_batch_size, in_channels, COL2IM_SIZE, COL2IM_SIZE)),
      grad_output_buffer_(std::make_unique<nn::Tensor>(max_batch_size,
          out_channels,
          DEFAULT_SIZE - kernel_size + 1,
          DEFAULT_SIZE - kernel_size + 1))
{
    // Initialize weights with He initialization
    initialize_weights_he();

    // Warm up indices computation with default size
    compute_indices_once(max_batch_size_, DEFAULT_SIZE, DEFAULT_SIZE);
}

// ============ Forward & Backward Passes ============

template <typename Backend>
auto Conv2dImpl<Backend>::forward(const typename Conv2dImpl<Backend>::Tensor& input,
    bool requires_grad) -> typename Conv2dImpl<Backend>::Tensor
{
    // Validate input tensor shape
    const auto& shape = input.get_shape();
    if (shape.size() != 4)
    {
        throw std::invalid_argument(
            "Conv2d input must be 4D (batch, channels, height, width), got " +
            std::to_string(shape.size()) + "D tensor");
    }

    const auto in_ch = static_cast<int>(shape[1]);
    if (in_ch != in_channels_)
    {
        throw std::invalid_argument("Conv2d input channels (" + std::to_string(in_ch) +
                                    ") do not match expected in_channels (" +
                                    std::to_string(in_channels_) + ")");
    }

    const auto batch_size = static_cast<int>(input.get_shape()[0]);
    const auto input_height = static_cast<int>(input.get_shape()[2]);
    const auto input_width = static_cast<int>(input.get_shape()[3]);

    // Cache input for backward pass only if gradients are required
    if (requires_grad)
    {
        input_cache_ = input;
    }

    // Compute output dimensions with stride and padding
    const int dilated_kernel_size = dilation_ * (kernel_size_ - 1) + 1;
    const int output_height = (input_height + 2 * padding_ - dilated_kernel_size) / stride_ + 1;
    const int output_width = (input_width + 2 * padding_ - dilated_kernel_size) / stride_ + 1;

    // Validate output dimensions
    if (output_height <= 0 || output_width <= 0)
    {
        throw std::invalid_argument(
            "Conv2d: input spatial dimensions (" + std::to_string(input_height) + "x" +
            std::to_string(input_width) + ") are too small for kernel size " +
            std::to_string(kernel_size_) + " with stride=" + std::to_string(stride_) +
            ", padding=" + std::to_string(padding_) + ", dilation=" + std::to_string(dilation_));
    }

    // Ensure indices are precomputed
    compute_indices_once(batch_size, input_height, input_width);

    // 1. Perform optimized im2col
    const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
    const int patch_cols_per_batch = output_height * output_width;
    const int total_patch_cols = batch_size * patch_cols_per_batch;

    // Resize buffer if needed (rare after initial preallocation)
    const bool need_resize = (im2col_buffer_->rows() < static_cast<nn::Index>(patch_rows) ||
                              im2col_buffer_->cols() < static_cast<nn::Index>(total_patch_cols));
    if (need_resize) [[unlikely]]
    {
        im2col_buffer_ = std::make_unique<nn::Tensor>(patch_rows, total_patch_cols);
    }

    auto& im2col_tensor = *im2col_buffer_;

    // Use optimized im2col with precomputed indices
    im2col_optimized(
        input, im2col_tensor, batch_size, input_height, input_width, output_height, output_width);

    // Use only the active part of the buffer
    auto im2col_active = im2col_tensor.block(0, 0, patch_rows, total_patch_cols);

    // 2. Perform matrix multiplication: output = weights^T * im2col
    // weights is (patch_rows x out_channels), im2col is (patch_rows x total_patch_cols)
    // We need weights^T * im2col = (out_channels x patch_rows) * (patch_rows x total_patch_cols)
    //                              = (out_channels x total_patch_cols)
    nn::Tensor weights_transposed = weights_.transpose();
    nn::Tensor output_2d = weights_transposed.matmul(im2col_active);

    // 3. Add bias using optimized broadcasting
    add_bias_optimized(output_2d, bias_, total_patch_cols);

    // 4. Reshape output efficiently
    nn::Tensor output =
        reshape_output_optimized(output_2d, batch_size, output_height, output_width);

    return output;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
template <typename Backend>
auto Conv2dImpl<Backend>::backward(const typename Conv2dImpl<Backend>::Tensor& grad_output) ->
    typename Conv2dImpl<Backend>::Tensor
{
    const auto batch_size = static_cast<int>(input_cache_.get_shape()[0]);
    const auto input_height = static_cast<int>(input_cache_.get_shape()[2]);
    const auto input_width = static_cast<int>(input_cache_.get_shape()[3]);
    const int dilated_kernel_size = dilation_ * (kernel_size_ - 1) + 1;
    const int output_height = (input_height + 2 * padding_ - dilated_kernel_size) / stride_ + 1;
    const int output_width = (input_width + 2 * padding_ - dilated_kernel_size) / stride_ + 1;

    const int patch_rows = in_channels_ * kernel_size_ * kernel_size_;
    const int patch_cols_per_batch = output_height * output_width;
    const int total_patch_cols = batch_size * patch_cols_per_batch;

    // Reshape grad_output to 2D (C, B*H*W)
    // We need to manually copy because grad_output is (B, C, H, W)
    nn::Tensor grad_output_2d(out_channels_, total_patch_cols);

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
                        const int col_idx = b * patch_cols_per_batch + oy * output_width + ox;
                        grad_output_2d.at(c, col_idx) = grad_output.at(b, c, oy, ox);
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
                        const int col_idx = b * patch_cols_per_batch + oy * output_width + ox;
                        grad_output_2d.at(c, col_idx) = grad_output.at(b, c, oy, ox);
                    }
                }
            }
        }
    }

    // 1. Compute bias gradient (sum over all positions and batches)
    // grad_output_2d.sum_rows() returns (C, 1), but bias is (1, C)
    bias_.set_grad(grad_output_2d.sum_rows().transpose());

    // 3. Get im2col of cached input
    auto& im2col_buffer = *im2col_buffer_;
    if (im2col_buffer.rows() < static_cast<nn::Index>(patch_rows) ||
        im2col_buffer.cols() < static_cast<nn::Index>(total_patch_cols))
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

    // Use only the active part of the buffer
    auto im2col_active = im2col_buffer.block(0, 0, patch_rows, total_patch_cols);

    // 3. Compute weights gradient: dW = im2col_input * dY^T
    // im2col_buffer is (patch_rows, total_patch_cols), grad_output is (out_channels,
    // total_patch_cols)
    nn::Tensor grad_output_transposed = grad_output_2d.transpose();
    weights_.set_grad(im2col_active.matmul(grad_output_transposed));

    // 4. Compute input gradient: dX_col = W * dY
    // weights_ is (patch_rows, out_channels), grad_output is (out_channels, total_patch_cols)
    nn::Tensor d_input_col = weights_.matmul(grad_output_2d);

    // 5. Convert col2im (input gradient)
    nn::Tensor grad_input = col2im_optimized(
        d_input_col, batch_size, input_height, input_width, output_height, output_width);

    return grad_input;
}

// ============ Getters ============

template <typename Backend>
auto Conv2dImpl<Backend>::get_weights() const -> const typename Conv2dImpl<Backend>::Tensor&
{
    return weights_;
}

template <typename Backend>
auto Conv2dImpl<Backend>::get_weights() -> typename Conv2dImpl<Backend>::Tensor&
{
    return weights_;
}

template <typename Backend>
auto Conv2dImpl<Backend>::get_bias() const -> const typename Conv2dImpl<Backend>::Tensor&
{
    return bias_;
}

template <typename Backend>
auto Conv2dImpl<Backend>::get_bias() -> typename Conv2dImpl<Backend>::Tensor&
{
    return bias_;
}

// ============ Setters ============

template <typename Backend>
void Conv2dImpl<Backend>::set_weights(const typename Conv2dImpl<Backend>::Tensor& weights)
{
    weights_ = weights;
}

template <typename Backend>
void Conv2dImpl<Backend>::set_bias(const typename Conv2dImpl<Backend>::Tensor& bias)
{
    bias_ = bias;
}

template <typename Backend>
bool Conv2dImpl<Backend>::is_parallel_enabled() const
{
    return use_parallel_;
}

template <typename Backend>
void Conv2dImpl<Backend>::set_parallel_enabled(bool enabled)
{
    use_parallel_ = enabled;
}

// ============ Initialization ============

template <typename Backend>
void Conv2dImpl<Backend>::initialize_weights_he()
{
    const int fan_in = in_channels_ * kernel_size_ * kernel_size_;
    const float stddev = std::sqrt(2.0F / static_cast<float>(fan_in));

    // Generate random normal distribution using C++ <random>
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0F, stddev);

    float* weights_data = weights_.mutable_data_ptr();
    const auto total_weights = weights_.rows() * weights_.cols();
    for (nn::Index i = 0; i < total_weights; ++i)
    {
        weights_data[i] = dist(gen);
    }

    // Initialize bias to small positive values using Tensor backend
    float* bias_data = bias_.mutable_data_ptr();
    const auto total_bias = bias_.rows() * bias_.cols();
    for (nn::Index i = 0; i < total_bias; ++i)
    {
        bias_data[i] = 0.01F;
    }
}

// Explicit instantiation: generate code for the Eigen (CPU) backend only.
// A GPU backend implementation would add its own explicit instantiation here or in a
// separate translation unit after providing the required algorithm specialisation.
template class Conv2dImpl<nn::EigenTensorBackend>;
