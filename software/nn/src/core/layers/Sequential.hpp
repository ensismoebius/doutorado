#pragma once
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "Module.hpp"
#include "core/tensor/Tensor.hpp"

// A PyTorch-like Sequential container for C++
struct Sequential : Module
{
    std::vector<std::shared_ptr<Module>> layers;
    std::vector<nn::Tensor> outputs; // output cache

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
