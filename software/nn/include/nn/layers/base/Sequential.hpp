#ifndef NN_LAYERS_SEQUENTIAL_HPP
#define NN_LAYERS_SEQUENTIAL_HPP
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "nn/layers/base/Module.hpp"
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
 *
 * Backend polymorphism:
 * - `SequentialImpl<Backend>` works with any backend tensor type.
* - The convenience alias `Sequential = SequentialImpl<Backend>` (in
* `nn/layers/Layers.hpp`) preserves backward compatibility with all existing call sites.
 */

/// Backend-parameterized Sequential container. All layers in the container must share
/// the same `Backend`; use a cross-backend bridge node for mixed-backend pipelines.
template <typename Backend>
struct SequentialImpl : Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

    std::vector<std::shared_ptr<Module<Backend>>> layers;
    /// Optional cache of intermediate activations (per layer) for debugging/visualization.
    std::vector<Tensor> outputs;
    // Owned concatenation of parameter pointers from child layers. This storage
    // must remain valid while the returned span is used by callers.
    std::vector<nn::Tensor*> param_ptrs_;

    SequentialImpl() = default;

    // PyTorch-like constructor: Sequential({layer1, layer2, ...})
    SequentialImpl(std::initializer_list<std::shared_ptr<Module<Backend>>> init_layers)
        : layers(init_layers)
    {
    }

    // Constructor from a vector of layers
    explicit SequentialImpl(const std::vector<std::shared_ptr<Module<Backend>>>& init_layers)
        : layers(init_layers)
    {
    }

    // Add a layer (PyTorch: .add_module)
    void add_module(const std::shared_ptr<Module<Backend>>& module)
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
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (layers.empty())
        {
            throw std::runtime_error("Sequential: cannot forward with empty layer list");
        }

        // Cache intermediates so debugging/visualization can inspect per-layer outputs.
        // Note: this cache is not strictly required for backprop because each layer may
        // manage its own caches; it is primarily a convenience.
        outputs.clear();
        Tensor temp_input = input;
        for (auto& layer : layers) [[likely]]
        {
            temp_input = layer->forward(temp_input, requires_grad);
            outputs.emplace_back(temp_input);
        }
        return temp_input;
    }

    // Backward pass
    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // Reverse-order gradient propagation (chain rule).
        Tensor grad = grad_output;
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
    [[nodiscard]] auto params() -> std::span<nn::Tensor*> override
    {
        param_ptrs_.clear();
        for (auto& layer : layers) [[likely]]
        {
            auto layer_params = layer->params();
            param_ptrs_.insert(param_ptrs_.end(), layer_params.begin(), layer_params.end());
        }
        return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, nn::Tensor> override
    {
        std::map<std::string, nn::Tensor> out;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            auto d = layers[i]->state_dict();
            for (const auto& kv : d)
            {
                out[std::to_string(i) + "." + kv.first] = kv.second;
            }
        }
        return out;
    }

    void load_state_dict(const std::map<std::string, nn::Tensor>& sd) override
    {
        // Dispatch keys of form "<idx>.<name>" to child modules
        for (const auto& kv : sd)
        {
            const std::string& key = kv.first;
            auto pos = key.find('.');
            if (pos == std::string::npos) continue;
            auto idx_str = key.substr(0, pos);
            auto name = key.substr(pos + 1);
            size_t idx = static_cast<size_t>(std::stoul(idx_str));
            if (idx >= layers.size()) continue;
            // Create a small map and call child load
            std::map<std::string, nn::Tensor> child_map{{name, kv.second}};
            layers[idx]->load_state_dict(child_map);
        }
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

#endif // NN_LAYERS_SEQUENTIAL_HPP
