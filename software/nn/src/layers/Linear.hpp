#ifndef LINEAR_CPP
#define LINEAR_CPP

#include <cnpy.h>

#include "../tensor/Tensor.hpp"
#include "layers/Module.hpp"

/**
 * @brief Camada Linear (ou camada totalmente conectada)
 * Implementa uma camada linear que aplica a transformação afim:
 * saída = entrada * W^T + b
 *
 * Onde:
 * - entrada é o tensor de entrada
 * - weight é a matriz de pesos
 * - bias é o vetor de bias
 */
struct Linear : public Module {

  int in_features;             // número de entradas (features de entrada do tensor)
  int out_features;            // número de saídas (neurônios ou unidades na camada)
  Tensor weight;               // matriz de pesos com dimensão [out_features x in_features]
  Tensor bias;                 // vetor de bias com dimensão [out_features]
  Eigen::MatrixXf input_cache; // armazena a entrada da camada para uso no backpropagation

  /**
   * @brief Inicializa pesos e bias com base no número de entradas e saídas
   *
   * @param in_features Número de entradas
   * @param out_features Número de saídas
   */
  Linear(const int in_features, const int out_features)
      : in_features(in_features), out_features(out_features), weight(out_features, in_features),
        bias(out_features, 1) {}

#ifdef DEBUG
  // If DEBUG is defined then show the debug information
  auto debug(const Tensor &input) -> void {
    std::cout << "Linear layer forward:" << "\n";
    std::cout << "Input dims: " << input.data.rows() << "x" << input.data.cols() << "\n";
    std::cout << "Weight dims: " << weight.data.rows() << "x" << weight.data.cols() << "\n";
    std::cout << "Bias dims: " << bias.data.rows() << "x" << bias.data.cols() << "\n";
  }
#endif

  /**
   * @brief Realiza a propagação direta (forward pass) da
   * entrada pela camada linear.
   * Aplica a transformação afim: saída = entrada * W^T + b
   * @param input Tensor de saída [batch_size x out_features]
   * @return Tensor de saída [batch_size x out_features]
   */
  auto forward(const Tensor &input) -> Tensor override {
    input_cache = input.data; // salva para o backward

#ifdef DEBUG
    // If DEBUG is defined then show the debug information
    debug(input);
#endif

    // Be x = input and y = output
    // y = x.w + b
    // Ensure bias is broadcast as a row vector
    Eigen::MatrixXf const output =
        (input.data * weight.data.transpose()).rowwise() + bias.data.col(0).transpose();
    return {output};
  }

  /**
   * @brief Realiza a propagação reversa (backward pass) da derivada da perda
   * em relação à saída da camada.Calcula os gradientes da perda em relação aos
   * pesos, bias e à entrada da camada
   *
   * @param grad_previous gradiente da perda em relação à saída da camada [batch_size x
   * out_features]
   * @return gradiente da perda em relação à entrada da camada [batch_size x in_features]
   */
  auto backward(const Tensor &grad_previous) -> Tensor override {

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

    weight.grad = grad_previous.data.transpose() * input_cache;

    // Da mesma forma o gradiente em relação a B será expresso por
    // dY/db = dY/dZ * dZ/dB
    // dZ/dB = (WX + B)' = 0 + 1*B^0 = 1
    // Portanto:
    // dY/dB = grad_previous * 1 = grad_previous

    // Considerando que estamos processando mais de um
    // vetor de entrada (um batch), somamos as derivadas
    // por linha:
    bias.grad = grad_previous.data.colwise().sum().transpose();

    // Por fim a derivada de X será
    // dY/dX = dY/dZ * dZ/dX
    // dY/dX = grad_output * (WX + B)'
    // dY/dX = grad_output * W*1*X^0 + 0
    // dY/dX = grad_output * W
    Eigen::MatrixXf const grad_input = grad_previous.data * weight.data;

    return {grad_input};
  }
};

#endif