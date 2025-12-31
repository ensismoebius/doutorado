#ifndef LINEAR_CPP
#define LINEAR_CPP

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

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
struct Linear : public Module
{
    int in_features;        // número de entradas (features de entrada do tensor)
    int out_features;       // número de saídas (neurônios ou unidades na camada)
    nn::Tensor weight;      // matriz de pesos com dimensão [out_features x in_features]
    nn::Tensor bias;        // vetor de bias com dimensão [out_features]
    nn::Tensor input_cache; // armazena a entrada da camada para uso no backpropagation

    /**
     * @brief Inicializa pesos e bias com base no número de entradas e saídas
     *
     * @param in_features Número de entradas
     * @param out_features Número de saídas
     */
    Linear(const int in_features, const int out_features)
        : in_features(in_features),
          out_features(out_features),
          weight(nn::Tensor(out_features, in_features)),
          bias(nn::Tensor(out_features, 1))
    {
    }

#ifdef DEBUG
    // If DEBUG is defined then show the debug information
    auto debug(const nn::Tensor& input) -> void
    {
        std::cout << "Linear layer forward:" << "\n";
        std::cout << "Input dims: " << input.get_data_ref().rows() << "x"
                  << input.get_data_ref().cols() << "\n";
        std::cout << "Weight dims: " << weight.get_data_ref().rows() << "x"
                  << weight.get_data_ref().cols() << "\n";
        std::cout << "Bias dims: " << bias.get_data_ref().rows() << "x"
                  << bias.get_data_ref().cols() << "\n";
    }
#endif

    /**
     * @brief Realiza a propagação direta (forward pass) da
     * entrada pela camada linear.
     * Aplica a transformação afim: saída = entrada * W^T + b
     * @param input Tensor de saída [batch_size x out_features]
     * @return Tensor de saída [batch_size x out_features]
     */
    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        input_cache = input; // salva para o backward

#ifdef DEBUG
        // If DEBUG is defined then show the debug information
        debug(input);
#endif

        // Be x = input and y = output
        // y = x.w + b
        // Ensure bias is broadcast as a row vector
        auto weight_t = weight.transpose();
        auto intermediate = input.matmul(weight_t);

        // Add bias - broadcast bias across batch dimension
        // intermediate shape: [batch_size, out_features]
        // bias shape: [out_features, 1] -> needs to be broadcasted to [batch_size, out_features]
        Eigen::MatrixXf result_data = intermediate.get_data_ref();
        result_data.rowwise() += bias.get_data_ref().col(0).transpose();

        return nn::Tensor{result_data};
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
    auto backward(const nn::Tensor& grad_previous) -> nn::Tensor override
    {
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

        weight.set_grad(grad_previous.get_data_ref().transpose() * input_cache.get_data_ref());

        // Da mesma forma o gradiente em relação a B será expresso por
        // dY/db = dY/dZ * dZ/dB
        // dZ/dB = (WX + B)' = 0 + 1*B^0 = 1
        // Portanto:
        // dY/dB = grad_previous * 1 = grad_previous

        // Considerando que estamos processando mais de um
        // vetor de entrada (um batch), somamos as derivadas
        // por linha:
        bias.set_grad(grad_previous.get_data_ref().colwise().sum().transpose());

        // Por fim a derivada de X será
        // dY/dX = dY/dZ * dZ/dX
        // dY/dX = grad_output * (WX + B)'
        // dY/dX = grad_output * W*1*X^0 + 0
        // dY/dX = grad_output * W
        auto grad_input_tensor = grad_previous.matmul(weight);
        return grad_input_tensor;
    }
};

#endif