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
   * @param in_features Número de entradas
   * @param out_features Número de saídas
   */
  Linear(const int in_features, const int out_features) : in_features(in_features), out_features(out_features), weight(out_features, in_features), bias(out_features), grad_weight(out_features, in_features), grad_bias(out_features) {

    // Inicialização Xavier uniforme (uniforme em [-limite, +limite])
    // limite segundo Xavier
    float const limit = std::sqrt(6.0F / static_cast<float>(in_features + out_features));
    // distribuição uniforme
    std::uniform_real_distribution dist(-limit, limit);

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
    cnpy::npy_save(prefix + "_weights.npy", weight.data(), {static_cast<size_t>(weight.rows()), static_cast<size_t>(weight.cols())}, "w");
    cnpy::npy_save(prefix + "_bias.npy", bias.data(), {static_cast<size_t>(bias.size())}, "w");
  }

  auto load_weights(const std::string &prefix) -> void {
    auto loadedWeights = cnpy::npy_load(prefix + "_weights.npy");
    auto loadedBias = cnpy::npy_load(prefix + "_bias.npy");

    // Convert an array pointer to an iterable
    std::span<const float> const b_span(loadedBias.data<float>(), loadedBias.num_vals);
    std::span<const float> const w_span(loadedWeights.data<float>(), loadedWeights.num_vals);

    // Reconstruct the bias and the weights
    bias = Eigen::VectorXf(loadedBias.shape[0]);
    weight = Eigen::MatrixXf(loadedWeights.shape[0], loadedWeights.shape[1]);

    for (int i = 0; i < bias.size(); ++i) {
      bias(i) = b_span[i];
    }

    for (int i = 0; i < weight.size(); ++i) {
      weight(i) = w_span[i];
    }
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

    // Be x = input and y = output
    // y = x.w + b
    Eigen::MatrixXf const output = (input.data * weight.transpose()).rowwise() + bias.transpose();
    return {output};
  }

  /**
   * @brief Realiza a propagação reversa (backward pass) da derivada da perda
   * em relação à saída da camada.Calcula os gradientes da perda em relação aos
   * pesos, bias e à entrada da camada
   *
   * @param grad_previous gradiente da perda em relação à saída da camada [batch_size x out_features]
   * @return gradiente da perda em relação à entrada da camada [batch_size x in_features]
   */
  auto backward(const Tensor &grad_previous) -> Tensor {

    // Considerando que B, X, Y, Z e W são tensores.
    // Vamos representar a camada atual por um tensor Z = f(X) = WX + B.
    // Vamos representar a camada anterior por Y = g(Z) = (pode ser qualquer coisa).
    // Sendo assim sabemos que na fase do _forward_ as camadas f(X) e g(Z)
    // interagem da seguinte forma: Y = g(f(X))

    // Portanto para sabermos a direção do crescimento de W
    // quando há uma mudança em Y é necessário calcular
    // a derivada de Y em relação a W: dY/dW

    // Para que se obtenha isso precisamos recorrer à regra da cadeia:
    // dY/dW = dY/dZ * dZ/dW

    // Nessa camada dY/dZ = grad_previous então a expressão fica assim:
    // dY/dW = grad_previous * dZ/dW

    // dZ/dW = x * 1*W^0 + 0 = x

    // Portanto o gradiente da camada atual é:
    // dY/dW = grad_previous * x

    // Lembrando que:
    // x = input_cache, dY/dZ = grad_previous e dY/dW = grad_weight
    // Então dY/dW = dY/dZ * dZ/dW é igual a:
    // grad_weight = grad_previous.T * input_cache

    grad_weight = grad_previous.data.transpose() * input_cache;

    // Da mesma forma o gradiente em relação a B será expresso por
    // dY/db = dY/dZ * dZ/dB
    // dZ/dB = (WX + B)' = 0 + 1*B^0 = 1
    // Portanto:
    // dY/dB = grad_previous * 1 = grad_previous

    // Considerando que estamos processando mais de um
    // vetor de entrada (um batch), somamos as derivadas
    // por linha:
    grad_bias = grad_previous.data.colwise().sum();

    // Por fim a derivada de X será
    // dY/dX = dY/dZ * dZ/dX
    // dY/dX = grad_output * (WX + B)'
    // dY/dX = grad_output * W*1*X^0 + 0
    // dY/dX = grad_output * W
    Eigen::MatrixXf const grad_input = grad_previous.data * weight;

    return {grad_input};
  }
};

#endif