#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <map>
#include <span>
#include <vector>

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
     * Convenience no-arg overload: step using attached parameters (if any).
     * Calls the `step(std::span<...>)` virtual and dispatches to concrete
     * implementations.
     */
    virtual auto step() -> void
    {
        if (!attached_params_.empty())
        {
            step(std::span<nn::Tensor*>{attached_params_.data(), attached_params_.size()});
        }
        else
        {
            throw std::runtime_error("Optimizer::step() called with no attached parameters");
        }
    }

    /**
     * @brief Set all parameter gradients to zero before the next backward pass.
     */
    virtual auto zero_grad(std::span<nn::Tensor*> params) -> void = 0;

    /**
     * Convenience no-arg overload for zeroing attached parameters' gradients.
     */
    virtual auto zero_grad() -> void
    {
        if (!attached_params_.empty())
        {
            zero_grad(std::span<nn::Tensor*>{attached_params_.data(), attached_params_.size()});
        }
        else
        {
            throw std::runtime_error("Optimizer::zero_grad() called with no attached parameters");
        }
    }

    /**
     * @brief Optional hook for optimizers that need per-parameter state.
     *
     * Example: Adam stores first/second moments m/v with the same shape as each parameter.
     * Call this after building the model and before training.
     */
    virtual auto attach(std::span<nn::Tensor*> params) -> void {}

    // Stored copy of the last attached parameters (optional). Concrete optimizers
    // may still override `attach()` but should call `Optimizer::attach(params)`
    // to preserve this storage for no-arg convenience methods.
    std::vector<nn::Tensor*> attached_params_;

    virtual ~Optimizer() = default;
    /**
     * @brief Return optimizer internal state as a map of name->Tensor.
     * Default: empty. Concrete optimizers may override to expose moments or counters.
     */
    virtual auto state_dict() const -> std::map<std::string, nn::Tensor>
    {
        return {};
    }

    /**
     * @brief Load optimizer internal state from a map produced by `state_dict()`.
     */
    virtual void load_state_dict(const std::map<std::string, nn::Tensor>&) {}
};

#endif // OPTIMIZER_HPP
