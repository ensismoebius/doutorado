#ifndef LINEAR_CPP
#define LINEAR_CPP

#include <Eigen/Dense>
#include <cmath>
#include <cnpy.h>
#include <random>
#include <span>

#include "../tensor/Tensor.hpp"

struct Linear {
  int in_features;             // número de entradas (features de entrada do tensor)
  int out_features;            // número de saídas (neurônios ou unidades na camada)
  Eigen::MatrixXf weight;      // matriz de pesos com dimensão [out_features x in_features]
  Eigen::VectorXf bias;        // vetor de bias com dimensão [out_features]
  Eigen::MatrixXf input_cache; // armazena a entrada da camada para uso no backpropagation

  Eigen::MatrixXf grad_weight;
  Eigen::VectorXf grad_bias;

  /**
   * @brief Inicializa pesos e bias com base no número de entradas e saídas
   *
   * @param in_f Número de entradas
   * @param out_f Número de saídas
   */
  Linear(int in_features, int out_features) : in_features(in_features), out_features(out_features), weight(out_features, in_features), bias(out_features), grad_weight(out_features, in_features), grad_bias(out_features) {

    // Inicialização Xavier uniforme (uniforme em [-limite, +limite])
    // limite segundo Xavier
    float limit = std::sqrt(6.0F / static_cast<float>(in_features + out_features));
    // distribuição uniforme
    std::uniform_real_distribution<float> dist(-limit, limit);

    // inicializa Mersenne Twister com a semente
    std::mt19937 gen(std::random_device{}());

    // Inicializa pesos com valores aleatórios da distribuição
    weight = Eigen::MatrixXf(out_features, in_features).unaryExpr([&](float) { return dist(gen); });

    // Inicializa bias também com valores aleatórios da mesma distribuição
    bias = Eigen::VectorXf(out_features).unaryExpr([&](float) { return dist(gen); });

    // Inicializa os gradientes como matrizes de zeros
    grad_weight.setZero();
    grad_bias.setZero();
  }

  auto save_weights(const std::string &prefix) const -> void {
    cnpy::npy_save(prefix + "_weights.npy", weight.data(), {(size_t)weight.rows(), (size_t)weight.cols()}, "w");
    cnpy::npy_save(prefix + "_bias.npy", bias.data(), {(size_t)bias.size()}, "w");
  }

  auto load_weights(const std::string &prefix) -> void {
    auto loadedWeights = cnpy::npy_load(prefix + "_weights.npy");
    auto loadedBias = cnpy::npy_load(prefix + "_bias.npy");

    // Convert an array pointer to an iterable
    std::span<const float> b_span(loadedBias.data<float>(), loadedBias.num_vals);
    std::span<const float> w_span(loadedWeights.data<float>(), loadedWeights.num_vals);

    // std::cout << "\n\n\n\nBias before\n" << bias;
    // std::cout << "\nWeight before\n" << weight;

    // Reconstruct the bias and the weights
    bias = Eigen::VectorXf(loadedBias.shape[0]);
    weight = Eigen::MatrixXf(loadedWeights.shape[0], loadedWeights.shape[1]);

    for (int i = 0; i < bias.size(); ++i) {
      bias(i) = b_span[i];
    }

    for (int i = 0; i < weight.size(); ++i) {
      weight(i) = w_span[i];
    }

    // std::cout << "\n\nBias after\n" << bias;
    // std::cout << "\nWeight after\n" << weight;
    // std::cout << "\n\n\n\n";
  }

  /**
   * @brief Realiza a propagação direta (forward pass) da
   * entrada pela camada linear.
   * Aplica a transformação afim: saída = entrada * W^T + b
   * @param input Tensor de saída [batch_size x out_features]
   * @return Tensor de saída [batch_size x out_features]
   */
  auto forward(const Tensor &input) -> Tensor {
    input_cache = input.data; // salva para o backward

    Eigen::MatrixXf output = (input.data * weight.transpose()).rowwise() + bias.transpose();
    return {output};
  }

  /**
   * @brief Realiza a propagação reversa (backward pass) da derivada da perda
   * em relação à saída da camada.Calcula os gradientes da perda em relação aos
   * pesos, bias e à entrada da camada
   *
   * @param grad_output gradiente da perda em relação à saída da camada [batch_size x out_features]
   * @return gradiente da perda em relação à entrada da camada [batch_size x in_features]
   */
  auto backward(const Tensor &grad_output) -> Tensor {
    // grad_output: dL/dY, onde Y = Wx + b
    // dL/dW = dL/dY * dY/dW = grad_output.T * input
    grad_weight = grad_output.data.transpose() * input_cache;

    // dL/db = soma das derivadas por linha (batch)
    grad_bias = grad_output.data.colwise().sum();

    // dL/dX = dL/dY * dY/dX = grad_output * W
    Eigen::MatrixXf grad_input = grad_output.data * weight;

    return {grad_input};
  }
};

#endif