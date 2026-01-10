#pragma once
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "nn/layers/Module.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file Sequential.hpp
 * @brief A lightweight PyTorch-like container that composes multiple `Module`s.
 *
 * Why this exists:
 * - Most experiments in this repo build models as stacks (Linear → LIF → Linear → ...).
 * - `Sequential` provides that composition with a tiny API surface.
 *
 * How it fits:
 * - Higher-level models (autoencoders, ResNets, etc.) typically own one or more `Sequential`s
 *   and delegate `forward()` / `backward()` to them.
 */

// A PyTorch-like Sequential container for C++
struct Sequential : Module
{
    std::vector<std::shared_ptr<Module>> layers;
    std::vector<nn::Tensor> outputs; // Optional cache of intermediate activations (per layer).

    Sequential() = default;

    // PyTorch-like constructor: Sequential({layer1, layer2, ...})
    Sequential(std::initializer_list<std::shared_ptr<Module>> init_layers) : layers(init_layers) {}

    // Constructor from a vector of layers
    explicit Sequential(const std::vector<std::shared_ptr<Module>>& init_layers)
        : layers(init_layers)
    {
    }

    // Add a layer (PyTorch: .add_module)
    void add_module(const std::shared_ptr<Module>& module)
    {
        layers.push_back(module);
    }

    // Operator[] for layer access (PyTorch: __getitem__)
    auto operator[](size_t idx)
    {
        return layers.at(idx);
    }
    auto operator[](size_t idx) const
    {
        return layers.at(idx);
    }

    // Forward pass
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        if (layers.empty())
        {
            throw std::runtime_error("Sequential: cannot forward with empty layer list");
        }

        // Cache intermediates so debugging/visualization can inspect per-layer outputs.
        // Note: this cache is not strictly required for backprop because each layer may
        // manage its own caches; it is primarily a convenience.
        outputs.clear();
        nn::Tensor temp_input = input;
        for (auto& layer : layers) [[likely]]
        {
            temp_input = layer->forward(temp_input, requires_grad);
            outputs.emplace_back(temp_input);
        }
        return temp_input;
    }

    // Backward pass
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // Reverse-order gradient propagation (chain rule).
        // Each layer is responsible for producing gradients for its own parameters.
        nn::Tensor grad = grad_output;
        for (size_t i = layers.size(); i-- > 0;) [[likely]]
        {
            grad = layers[i]->backward(grad);
        }
        return grad;
    }

    // Number of layers (PyTorch: __len__)
    [[nodiscard]] auto size() const
    {
        return layers.size();
    }

    // Returns all trainable parameters (weights and biases) from all layers
    [[nodiscard]] auto params() -> std::vector<nn::Tensor*> override
    {
        std::vector<nn::Tensor*> parameters;
        parameters.reserve(layers.size() * 2); // Reserve space for weights and biases

        for (auto& layer : layers) [[likely]]
        {
            // Concatenate per-layer params into one flat list for optimizers.
            auto layer_params = layer->params();
            parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
        }
        return parameters;
    }

    // Set training mode for all layers
    void train(bool on) override
    {
        for (auto& layer : layers) [[likely]]
        {
            layer->train(on);
        }
    }
};
