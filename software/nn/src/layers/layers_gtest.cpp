#include "layers/Leaky.hpp"
#include "layers/Linear.hpp"
#include "layers/ReLU.hpp"
#include "tensor/Tensor.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

// Teste para Linear
TEST(LinearLayerTest, ForwardSimple) {
    Linear linear(2, 1);
    // Define pesos e bias manualmente para teste determinístico
    linear.weight.data << 2.0f, 3.0f;
    linear.bias.data << 1.0f;
    Eigen::MatrixXf input(1, 2);
    input << 1.0f, 2.0f;
    Tensor in_tensor(input);
    Tensor out = linear.forward(in_tensor);
    // Esperado: (1*2 + 2*3) + 1 = 2 + 6 + 1 = 9
    ASSERT_FLOAT_EQ(out.data(0, 0), 9.0f);
}

// Teste para ReLU
TEST(ReLULayerTest, ForwardAndBackward) {
    ReLU relu;
    Eigen::MatrixXf input(2, 2);
    input << -1.0f, 2.0f, 0.0f, -3.0f;
    Tensor in_tensor(input);
    Tensor out = relu.forward(in_tensor);
    ASSERT_FLOAT_EQ(out.data(0, 0), 0.0f);
    ASSERT_FLOAT_EQ(out.data(0, 1), 2.0f);
    ASSERT_FLOAT_EQ(out.data(1, 0), 0.0f);
    ASSERT_FLOAT_EQ(out.data(1, 1), 0.0f);
    // Backward: grad_output ones
    Eigen::MatrixXf grad_out = Eigen::MatrixXf::Ones(2, 2);
    Tensor grad_tensor(grad_out);
    Tensor grad_in = relu.backward(grad_tensor);
    ASSERT_FLOAT_EQ(grad_in.data(0, 0), 0.0f);
    ASSERT_FLOAT_EQ(grad_in.data(0, 1), 1.0f);
    ASSERT_FLOAT_EQ(grad_in.data(1, 0), 0.0f);
    ASSERT_FLOAT_EQ(grad_in.data(1, 1), 0.0f);
}

// Teste para Leaky (LIF)
TEST(LeakyLayerTest, ForwardSpikeAndReset) {
    Leaky leaky(/*dt=*/1.0f, /*R=*/5.0f, /*C=*/1.0f, /*V_thresh=*/2.0f, /*reset_zero=*/true);
    Eigen::MatrixXf input(1, 1);
    input << 3.0f; // Acima do threshold
    Tensor in_tensor(input);
    Tensor out = leaky.forward(in_tensor);
    // Como input > threshold, deve gerar spike (1.0)
    ASSERT_FLOAT_EQ(out.data(0, 0), 1.0f);
    // Após spike, v_mem deve ser resetado para zero
    ASSERT_FLOAT_EQ(leaky.v_mem(0, 0), 0.0f);
}

// Teste para Leaky sem reset para zero
TEST(LeakyLayerTest, ForwardSpikeNoResetZero) {
    Leaky leaky(/*dt=*/1.0f, /*R=*/5.0f, /*C=*/1.0f, /*V_thresh=*/2.0f, /*reset_zero=*/false);
    Eigen::MatrixXf input(1, 1);
    input << 3.0f; // Acima do threshold
    Tensor in_tensor(input);
    Tensor out = leaky.forward(in_tensor);
    // Deve gerar spike
    ASSERT_FLOAT_EQ(out.data(0, 0), 1.0f);
    // v_mem deve ser reduzido pelo threshold
    ASSERT_FLOAT_EQ(leaky.v_mem(0, 0), 1.0f); // 3.0 - 2.0
}
