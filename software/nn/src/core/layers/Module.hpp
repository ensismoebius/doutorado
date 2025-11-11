#ifndef MODULE_HPP
#define MODULE_HPP

#include "../tensor/Tensor.hpp"

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
     * @brief forward propaga o sinal de entrada
     *
     * @param input
     * @return Tensor
     */
    virtual auto forward(const Tensor& input) -> Tensor = 0;

    /**
     * @brief backward propaga o gradiente (
     *
     * @param grad_output
     * @return Tensor
     */
    virtual auto backward(const Tensor& grad_output) -> Tensor = 0;

    /**
     * @brief Destroy the Module object
     */
    virtual ~Module() = default;
};

#endif
