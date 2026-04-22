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
 * - Optimizers operate on raw pointers returned by `params()`. Parameters are always
 *   stored as CPU-resident `nn::Tensor` regardless of `Backend` so that existing
 *   optimizers keep working without modification.
 *
 * Backend polymorphism:
 * - `Module` is templated on `Backend` (no default — callers must name it explicitly).
 * - `forward()` and `backward()` operate on `TensorImpl<Backend>`, making the
 *   interface agnostic to any specific backend (Eigen, OpenCL, CUDA, …).
 * - Adding a new backend never requires new virtual methods; instantiate
 *   `Module<NewBackend>` and derive from it instead.
 */
template <typename Backend>
struct Module
{
    /// Tensor type for this module's compute backend.
    using Tensor = nn::TensorImpl<Backend>;
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
     * @param input Input activations in the module's backend tensor type.
     * @param requires_grad When true, the module caches any state needed for
     *        backpropagation. When false, caching is skipped to reduce memory.
     * @return Output activations in the same backend tensor type.
     *
     * Invariant: if `requires_grad` is true, a subsequent call to `backward()` is
     * expected to produce meaningful gradients for the module parameters.
     */
    virtual auto forward(const Tensor& input, bool requires_grad = true) -> Tensor = 0;

    /**
     * @brief Backward pass: propagates gradient from module output to module input.
     *
     * @param grad_output Gradient of the loss w.r.t. this module's output, in the
     *        same backend tensor type as used by `forward()`.
     * @return Gradient of the loss w.r.t. this module's input.
     *
     * Expected usage: `backward()` is called in reverse topological order of the model.
     */
    virtual auto backward(const Tensor& grad_output) -> Tensor = 0;

    /**
     * @brief Sets the module in training mode.
     *
     * @param on true for training mode, false for evaluation mode.
     */
    virtual void train(bool on) {};

    /**
     * @brief Set training mode (PyTorch-style convenience method).
     *
     * Call this at the start of each training epoch.
     * Equivalent to train(true).
     */
    void train() { train(true); }

    /**
     * @brief Set evaluation mode (PyTorch-style convenience method).
     *
     * Call this before validation or inference.
     * Equivalent to train(false).
     */
    void eval() { train(false); }

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
