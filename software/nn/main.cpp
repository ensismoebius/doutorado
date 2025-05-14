#include <iostream>
#include <cmath>
#include "layers/ReLU.cpp"
#include "layers/Linear.cpp"
#include "util/vetorizationCheck.hpp"

float compute_mse_loss(const Tensor &prediction, const Tensor &target)
{
    Eigen::MatrixXf diff = prediction.data - target.data;
    return diff.array().square().mean();
}

Tensor compute_mse_grad(const Tensor &prediction, const Tensor &target)
{
    Eigen::MatrixXf grad = 2.0f * (prediction.data - target.data) / prediction.data.rows();
    return Tensor(grad);
}

int main()
{
    printVetorizationSupport();

    // Entrada x: 4 amostras, 2 features
    Tensor x(4, 2);
    x.data << 1, 0,
        0, 1,
        1, 1,
        0, 0;

    // Alvo y: saída esperada (4 amostras, 1 saída)
    Tensor y_target(4, 1);
    y_target.data << 1,
        1,
        0,
        0;

    // Camadas
    Linear linear1(2, 4); // primeira camada: input 2 → hidden 4
    ReLU relu;
    Linear linear2(4, 1); // segunda camada: hidden 4 → output 1

    float learning_rate = 0.1f;

    for (int epoch = 0; epoch < 100; ++epoch)
    {
        // Forward
        Tensor out1 = linear1.forward(x);
        Tensor out2 = relu.forward(out1);
        Tensor out3 = linear2.forward(out2);

        // Loss
        float loss = compute_mse_loss(out3, y_target);
        std::cout << "Epoch " << epoch << ", Loss = " << loss << std::endl;

        // Backward
        Tensor grad_loss = compute_mse_grad(out3, y_target);
        Tensor grad_linear2 = linear2.backward(grad_loss);
        Tensor grad_relu = relu.backward(grad_linear2);
        Tensor grad_linear1 = linear1.backward(grad_relu);

        // Atualização dos pesos
        linear2.weight -= learning_rate * linear2.grad_weight;
        linear2.bias -= learning_rate * linear2.grad_bias;

        linear1.weight -= learning_rate * linear1.grad_weight;
        linear1.bias -= learning_rate * linear1.grad_bias;
    }

    return 0;
}
