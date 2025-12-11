#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <span>

#include "../tensor/Tensor.hpp"

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

    virtual auto step(std::span<nn::Tensor*> params) -> void = 0;
    virtual auto zero_grad(std::span<nn::Tensor*> params) -> void = 0;
    virtual ~Optimizer() = default;
};

#endif // OPTIMIZER_HPP
