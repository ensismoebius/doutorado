#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <vector>

// Branch prediction hints: use compiler built-ins when available

#include "../tensor/Tensor.hpp"
#include "Conv2d_utils.hpp"
#include "Module.hpp"

/**
 * @file Conv2d.hpp
 * @brief 2D Convolution Layer for Neural Networks
 *
 * Implements an efficient 2D convolution layer using im2col/col2im operations
 * with optional OpenMP parallelization for forward and backward passes.
 *
 * Key features:
 * - Optimized im2col/col2im for efficient matrix operations
 * - Index caching to avoid redundant computations
 * - Optional parallel execution using OpenMP
 * - He initialization for weight parameters
 * - Full backpropagation support
 *
 * For implementation details, see Conv2d_impl.cpp and Conv2d_utils.cpp.
 */

class EIGEN_ALIGN16 Conv2d : public Module
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor for Conv2d layer
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Size of the convolution kernel (square kernels only)
     * @param max_batch_size Maximum batch size for buffer pre-allocation (default: 64)
     * @param use_parallel Enable OpenMP parallelization (default: true)
     */
    Conv2d(int in_channels, int out_channels, int kernel_size, int max_batch_size = 64,
           bool use_parallel = true);

    /**
     * @brief Forward pass: compute convolution output
     * @param input Input tensor of shape (batch, channels, height, width)
     * @return Output tensor with shape (batch, out_channels, out_height, out_width)
     */
    auto forward(const nn::Tensor& input) -> nn::Tensor override;

    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient of loss w.r.t. output
     * @return Gradient of loss w.r.t. input
     */
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override;

    // ============ Getters ============

    /**
     * @brief
     *
     */
    [[nodiscard]] auto get_weights() const -> const nn::Tensor&;

    /**
     * @brief Get mutable reference to weight parameters
     */
    auto get_weights() -> nn::Tensor&;

    /**
     * @brief Get const reference to bias parameters
     */
    [[nodiscard]] const nn::Tensor& get_bias() const;

    /**
     * @brief Get mutable reference to bias parameters
     */
    auto get_bias() -> nn::Tensor&;

    // ============ Setters ============

    /**
     * @brief Set weight parameters
     */
    void set_weights(const nn::Tensor& weights);

    /**
     * @brief Set bias parameters
     */
    void set_bias(const nn::Tensor& bias);

    /**
     * @brief Query if parallelization is enabled
     */
    [[nodiscard]] auto is_parallel_enabled() const -> bool;

    /**
     * @brief Enable or disable OpenMP parallelization
     */
    void set_parallel_enabled(bool enabled);

   private:
    // ============ Layer Parameters ============
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int max_batch_size_;
    bool use_parallel_;

    nn::Tensor weights_;
    nn::Tensor bias_;
    nn::Tensor input_cache_;

    // ============ Pre-allocated Buffers ============
    std::unique_ptr<nn::Tensor> im2col_buffer_;
    std::unique_ptr<nn::Tensor> col2im_buffer_;
    std::unique_ptr<nn::Tensor> grad_output_buffer_;

    // ============ Index Caching ============
    mutable Conv2dImpl::IndexCache index_cache_;
    mutable bool indices_computed_ = false;
    mutable std::mutex cache_mutex_;

    // ============ Private Helper Methods ============

    /**
     * @brief Get or compute precomputed indices for given input dimensions
     */
    auto get_or_compute_indices(int input_height, int input_width) const
        -> const std::vector<Conv2dImpl::PatchIndices>&;

    /**
     * @brief Precompute all im2col/col2im indices for efficient reuse
     */
    auto compute_indices(int input_height, int input_width) const
        -> std::vector<Conv2dImpl::PatchIndices>;

    /**
     * @brief Ensure indices are computed for given dimensions (one-time initialization)
     */
    void compute_indices_once(int input_height, int input_width) const;

    /**
     * @brief Convert input to column matrix (im2col transformation)
     *
     * Rearranges image patches into columns for efficient matrix multiplication.
     */
    void im2col_optimized(const nn::Tensor& input, nn::Tensor& output, int batch_size,
                          int input_height, int input_width, int output_height,
                          int output_width) const;

    /**
     * @brief Convert column matrix back to image (col2im transformation)
     *
     * Reconstructs spatial dimensions from column format, accumulating overlapping patches.
     */
    auto col2im_optimized(const nn::Tensor& cols, int batch_size, int input_height, int input_width,
                          int output_height, int output_width) const -> nn::Tensor;

    /**
     * @brief Add bias to output matrix with optimized broadcasting
     */
    void add_bias_optimized(Eigen::MatrixXf& matrix, const nn::Tensor& bias, int num_cols) const;

    /**
     * @brief Reshape output matrix to 4D tensor without data copy
     */
    auto reshape_output_optimized(const nn::Tensor& matrix, int batch_size, int output_height,
                                  int output_width) const -> nn::Tensor;

    /**
     * @brief Initialize weights using He initialization for ReLU networks
     */
    void initialize_weights_he();
};
