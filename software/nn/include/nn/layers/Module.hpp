#ifndef MODULE_HPP
#define MODULE_HPP

#include "nn/tensor/Tensor.hpp"
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
    virtual auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor = 0;

    /**
     * @brief backward propaga o gradiente (
     *
     * @param grad_output
     * @return Tensor
     */
    virtual auto backward(const nn::Tensor& grad_output) -> nn::Tensor = 0;

    /**
     * @brief Sets the module in training mode.
     *
     * @param on true for training mode, false for evaluation mode.
     */
    virtual void train(bool on) {};

    /**
     * @brief Returns a vector of pointers to the trainable parameters (weights, biases) of the
     * module.
     * @return std::vector<Tensor*>
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
