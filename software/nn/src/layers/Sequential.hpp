#include "Module.hpp"
#include "tensor/Tensor.hpp"
#include <cstddef>

struct Sequential : Module {

  std::vector<std::shared_ptr<Module>> layers;

  std::vector<Tensor> outputs; // para cache de saída

  auto add(const std::shared_ptr<Module> &module) -> void {
    layers.push_back(module);
  }

  auto forward(const Tensor &input) -> Tensor override {
    outputs.clear();
    Tensor temp_input = input;
    for (auto &layer : layers) {
      temp_input = layer->forward(temp_input);
      outputs.emplace_back(temp_input);
    }
    return temp_input;
  }

  auto backward(const Tensor &grad_output) -> Tensor override {
    Tensor grad = grad_output;
    for (size_t i = layers.size() - 1; i >= 0; --i) {
      grad = layers[i]->backward(grad);
    }
    return grad;
  }
};
