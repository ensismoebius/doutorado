#pragma once
#include "Module.hpp"
#include "tensor/Tensor.hpp"
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

// A PyTorch-like Sequential container for C++
struct Sequential : Module {

  std::vector<std::shared_ptr<Module>> layers;
  std::vector<Tensor> outputs; // output cache

  Sequential() = default;

  // PyTorch-like constructor: Sequential({layer1, layer2, ...})
  Sequential(std::initializer_list<std::shared_ptr<Module>> init_layers) : layers(init_layers) {}

  // Add a layer (PyTorch: .add_module)
  void add_module(const std::shared_ptr<Module> &module) {
    layers.push_back(module);
  }

  // Operator[] for layer access (PyTorch: __getitem__)
  auto operator[](size_t idx) {
    return layers.at(idx);
  }
  auto operator[](size_t idx) const {
    return layers.at(idx);
  }

  // Forward pass
  auto forward(const Tensor &input) -> Tensor override {
    outputs.clear();
    Tensor temp_input = input;
    for (auto &layer : layers) {
      temp_input = layer->forward(temp_input);
      outputs.emplace_back(temp_input);
    }
    return temp_input;
  }

  // Backward pass
  auto backward(const Tensor &grad_output) -> Tensor override {
    Tensor grad = grad_output;
    for (size_t i = layers.size(); i-- > 0;) {
      grad = layers[i]->backward(grad);
    }
    return grad;
  }

  // Number of layers (PyTorch: __len__)
  [[nodiscard]] auto size() const {
    return layers.size();
  }
};
