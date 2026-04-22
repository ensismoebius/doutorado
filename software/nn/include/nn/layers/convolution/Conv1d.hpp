#ifndef NN_LAYERS_CONV1D_HPP
#define NN_LAYERS_CONV1D_HPP

#include <memory>
#include <mutex>
#include <vector>

#include "nn/layers/base/Module.hpp"
#include "nn/logging/Logger.hpp"

/**
 * @file Conv1d.hpp
 * @brief 1D Convolution Layer for Neural Networks
 *
 * Implements a 1D convolution layer for processing sequential data such as
 * time series, audio waveforms, or encoded features over time.
 *
 * Key features:
 * - 1D convolution for temporal/sequential data
 * - He initialization for weight parameters
 * - Full backpropagation support
 * - Configurable stride and padding
 *
 * Input shape: (batch, in_channels, length)
 * Output shape: (batch, out_channels, length_out)
 *   where length_out = floor((length + 2*padding - kernel_size) / stride) + 1
 */

template <typename Backend>
class Conv1dImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;

    Tensor weights_;
    Tensor bias_;
    Tensor input_cache_;

   public:
    /**
     * @brief Constructor for Conv1d layer (simplified API)
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Size of the convolution kernel
     * @param stride Stride for the convolution (default: 1)
     * @param padding Padding for the convolution (default: 0)
     * @param dilation Dilation for the convolution (default: 1)
     */
    Conv1dImpl(int in_channels,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        int dilation);

    /**
     * @brief Forward pass: compute convolution output
     * @param input Input tensor of shape (batch, in_channels, length)
     * @param requires_grad Whether gradients will be computed (default: true)
     * @return Output tensor with shape (batch, out_channels, length_out)
     */
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;

    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient of loss w.r.t. output
     * @return Gradient of loss w.r.t. input
     */
    auto backward(const Tensor& grad_output) -> Tensor override;

    // ============ Getters ============

    [[nodiscard]] auto get_weights() const -> const Tensor&;
    auto get_weights() -> Tensor&;
    [[nodiscard]] const Tensor& get_bias() const;
    auto get_bias() -> Tensor&;

    // ============ Setters ============

    void set_weights(const Tensor& weights);
    void set_bias(const Tensor& bias);

   private:
    void initialize_weights_he();

    auto compute_output_length(int input_length) const -> int;
};

#endif // NN_LAYERS_CONV1D_HPP
