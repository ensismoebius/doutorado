#ifndef MODULE_HPP
#define MODULE_HPP

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
     * @brief Returns pointers to trainable parameters owned by this module.
     *
     * Parameters are returned as raw pointers so optimizers can update them in-place.
     * Convention: if a module has no trainable parameters, it returns an empty vector.
     */
    virtual auto params() -> std::vector<nn::Tensor*>
    {
        return {};
    }

    /**
     * @brief Destroy the Module object
     */
    virtual ~Module() = default;
};

#endif
