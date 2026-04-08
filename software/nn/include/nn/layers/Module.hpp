#ifndef MODULE_HPP
#define MODULE_HPP

#include <map>
#include <span>
#include <string>

#include "nn/device/Device.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file Module.hpp
 * @brief Core interface for all differentiable building blocks ("layers") in this project.
 *
 * This project follows a small PyTorch-like convention:
 * - A `Module` transforms an input tensor in `forward()`.
 * - Training computes gradients by calling `backward()` on the same module graph.
 * - Some modules are stateful (e.g., spiking neurons) and must be reset between
 *   independent sequences/batches using `reset_state()`.
 *
 * Design note:
 * - The library keeps the interface intentionally minimal to make experiments readable.
 * - Optimizers operate on raw pointers returned by `params()`.
 */
struct Module
{
    Module() = default;

    /**
     * @brief Copy contructor of an Module object (enabled)
     * Used whe copying an object from an variable to another
     * @param copy
     */
    Module(const Module& otherObjectReference) = default;

    /**
     * @brief The '=' operator copies an object from an variable to another one
     *
     * @param otherObjectReference
     * @return Module&
     */
    auto operator=(const Module&) -> Module& = default;

    /**
     * @brief Move constructor of an Optimizer object (disabled)
     * Used when moving an object from an variable to another
     * @param otherObjectReference
     */
    Module(Module&& otherObjectReference) = delete;

    /**
     * @brief Disabled move operation from an objecto to another
     *
     * @param otherObjectReference
     * @return Module&
     */
    auto operator=(Module&&) -> Module& = delete;

    /**
     * @brief Forward pass: computes the module output given an input tensor.
     *
     * @param input Input activations (shape depends on the module).
     * @param requires_grad When true, the module is allowed to cache any state needed
     *        for backpropagation (e.g., inputs, membrane histories). When false, the
     *        module should avoid caching to reduce memory.
     * @return Output activations.
     *
     * Invariant: if `requires_grad` is true, a subsequent call to `backward()` is
     * expected to produce meaningful gradients for the module parameters.
     */
    virtual auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor = 0;

    /**
     * @brief Templated forward pass for backend-polymorphic execution.
     *
     * This allows forwarding tensors with different backends (Eigen, OpenCL, etc.)
     * without CPU->GPU copies. The default implementation converts to nn::Tensor
     * (Eigen backend). Override for GPU-native implementations.
     *
     * @tparam Backend Tensor backend type (e.g., EigenTensorBackend, OpenCLTensorBackend)
     * @param input Input tensor with any backend
     * @param requires_grad Enable gradient caching
     * @return Output tensor (default returns Eigen backend)
     */
    template <typename Backend>
    auto forward(nn::TensorImpl<Backend>& input, bool requires_grad = true) -> nn::Tensor
    {
        return forward(static_cast<const nn::Tensor&>(input), requires_grad);
    }

    /**
     * @brief Backward pass: propagates gradient from module output to module input.
     *
     * @param grad_output Gradient of the loss with respect to this module's output.
     * @return Gradient of the loss with respect to this module's input.
     *
     * Expected usage: `backward()` is called in reverse topological order of the model.
     */
    virtual auto backward(const nn::Tensor& grad_output) -> nn::Tensor = 0;

    /**
     * @brief Sets the module in training mode.
     *
     * @param on true for training mode, false for evaluation mode.
     */
    virtual void train(bool on) {};

    /**
     * @brief Resets stateful internal variables (e.g., LIF membrane potential).
     *
     * This is crucial for spiking/recurrent modules: if you feed multiple independent
     * sequences, failing to reset can contaminate one sample with state from the previous
     * one and break the i.i.d. assumption of SGD/Adam.
     */
    virtual void reset_state() {};

    /**
     * @brief Returns a non-owning view of trainable parameters owned by this module.
     *
     * The returned `std::span` is a non-owning view and MUST reference storage
     * that is owned by the `Module` (or by the caller). Implementations that
     * previously returned freshly-constructed `std::vector<nn::Tensor*>` must
     * instead expose storage (e.g. a persistent `std::vector<nn::Tensor*>` member)
     * whose lifetime outlives the span.
     *
     * Convention: if a module has no trainable parameters, return an empty span.
     */
    virtual auto params() -> std::span<nn::Tensor*>
    {
        return std::span<nn::Tensor*>{};
    }

    /**
     * @brief Return a map of parameter name -> Tensor for this module.
     *
     * Default implementation returns an empty map. Layers that expose named
     * parameters (e.g., `Linear`) should override this to allow saving/loading
     * state via `state_dict()` / `load_state_dict()`.
     */
    virtual auto state_dict() const -> std::map<std::string, nn::Tensor>
    {
        return {};
    }

    /**
     * @brief Load parameters from a state dictionary produced by `state_dict()`.
     *
     * Default implementation is a no-op. Implementations should copy shapes and
     * values from the provided tensors into their parameter members.
     */
    virtual void load_state_dict(const std::map<std::string, nn::Tensor>&) {}

    /**
     * @brief Move the module to the specified compute device (PyTorch-like).
     *
     * The default implementation lazily starts the device runtime for OpenCL
     * devices and returns `*this` for method chaining. Derived modules that
     * carry device-resident buffers may override to perform actual data transfer.
     *
     * @param device Target device (see `nn::Device::from_string`).
     * @return Reference to `*this` for chaining.
     */
    virtual auto to(const nn::Device& device) -> Module&
    {
        nn::DeviceRuntime::ensure_runtime(device);
        return *this;
    }

    /**
     * @brief Destroy the Module object
     */
    virtual ~Module() = default;
};

#endif
