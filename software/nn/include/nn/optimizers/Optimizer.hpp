#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <span>

#include "nn/tensor/Tensor.hpp"

/**
 * @file Optimizer.hpp
 * @brief Minimal optimizer interface for updating trainable parameters in-place.
 *
 * How it fits:
 * - A model exposes its trainable tensors via `Module::params()`.
 * - An `Optimizer` receives those pointers and updates `param->data` using `param->grad()`.
 *
 * Design choices:
 * - Parameters are passed as `std::span<nn::Tensor*>` to avoid unnecessary allocations.
 * - `attach()` lets an optimizer allocate per-parameter state (e.g., Adam moments) once.
 */

struct Optimizer
{
    /**
     * @brief Default contructor of the Optimizer object
     *
     */
    Optimizer() = default;

    /**
     * @brief Copy contructor of an Optimizer object (enabled)
     * Used whe copying an object from an variable to another
     * @param otherObjectReference
     */
    Optimizer(const Optimizer& otherObjectReference) = default;

    /**
     * @brief The '=' operator copies an object from an variable to another one
     *
     * @param otherObjectReference
     * @return Optimizer&
     */
    auto operator=(const Optimizer& otherObjectReference) -> Optimizer& = default;

    /**
     * @brief Move constructor of an Optimizer object (disabled)
     * Used when moving an object from an variable to another
     * @param otherObjectReference
     */
    Optimizer(Optimizer&& otherObjectReference) = delete;

    /**
     * @brief Disabled move operation from an objecto to another
     *
     * @param otherObjectReference
     * @return Optimizer&
     */
    auto operator=(Optimizer&& otherObjectReference) -> Optimizer& = delete;

    /**
     * @brief Apply one parameter update step.
     *
     * Precondition: each tensor in `params` has a valid gradient (usually set by backward()).
     */
    virtual auto step(std::span<nn::Tensor*> params) -> void = 0;

    /**
     * @brief Set all parameter gradients to zero before the next backward pass.
     */
    virtual auto zero_grad(std::span<nn::Tensor*> params) -> void = 0;

    /**
     * @brief Optional hook for optimizers that need per-parameter state.
     *
     * Example: Adam stores first/second moments m/v with the same shape as each parameter.
     * Call this after building the model and before training.
     */
    virtual auto attach(std::span<nn::Tensor*> params) -> void {}
    virtual ~Optimizer() = default;
};

#endif // OPTIMIZER_HPP
