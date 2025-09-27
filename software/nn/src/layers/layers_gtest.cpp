#include "../initializers/xavier.hpp"
#include "layers/Leaky.hpp"
#include "layers/Linear.hpp"
#include "layers/MSELoss.hpp"
#include "layers/ReLU.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <memory>
// Teste para MSELoss
TEST(MSELossTest, ForwardAndBackward) {
  MSELoss mse;
  Eigen::MatrixXf pred(2, 1);
  pred << 1.0F, 2.0F;
  Eigen::MatrixXf target(2, 1);
  target << 0.0F, 2.0F;
  Tensor pred_tensor(pred);
  Tensor target_tensor(target);
  mse.set_target(target_tensor);
  Tensor loss = mse.forward(pred_tensor);
  ASSERT_NEAR(loss.data(0, 0), 0.5F, 1e-5F);
  Tensor grad = mse.backward(pred_tensor);
  ASSERT_NEAR(grad.data(0, 0), 1.0F, 1e-5F);
  ASSERT_NEAR(grad.data(1, 0), 0.0F, 1e-5F);
}

// Teste para Sequential
TEST(SequentialTest, ForwardAndBackward) {
  auto linear = std::make_shared<Linear>(2, 1);
  linear->weight.data << 1.0F, 2.0F;
  linear->bias.data << 0.0F;
  auto relu = std::make_shared<ReLU>();
  Sequential seq({linear, relu});
  Eigen::MatrixXf input(1, 2);
  input << -1.0F, 2.0F;
  Tensor in_tensor(input);
  Tensor out = seq.forward(in_tensor);
  ASSERT_NEAR(out.data(0, 0), 3.0F, 1e-5F);
  Eigen::MatrixXf grad_out(1, 1);
  grad_out << 1.0F;
  Tensor grad_tensor(grad_out);
  Tensor grad_in = seq.backward(grad_tensor);
  ASSERT_NEAR(grad_in.data(0, 0), 1.0F, 1e-5F);
  ASSERT_NEAR(grad_in.data(0, 1), 2.0F, 1e-5F);
}

// Teste para Linear
TEST(LinearLayerTest, ForwardSimple) {
  Linear linear(2, 1);
  // Define pesos e bias manualmente para teste determinístico
  linear.weight.data << 2.0F, 3.0F;
  linear.bias.data << 1.0F;
  Eigen::MatrixXf input(1, 2);
  input << 1.0F, 2.0F;
  Tensor in_tensor(input);
  Tensor out = linear.forward(in_tensor);
  // Esperado: (1*2 + 2*3) + 1 = 2 + 6 + 1 = 9
  ASSERT_FLOAT_EQ(out.data(0, 0), 9.0F);
}

// Teste para Leaky (LIF)
TEST(LeakyLayerTest, ForwardSpikeAndReset) {
  Leaky leaky(/*dt=*/1.0F, /*R=*/5.0F, /*C=*/1.0F, /*V_thresh=*/2.0F, /*reset_zero=*/true, 0.0F, std::make_shared<ExponentialSurrogate>());
  Eigen::MatrixXf input(1, 1);
  input << 3.0F; // Acima do threshold
  Tensor in_tensor(input);
  Tensor out = leaky.forward(in_tensor);
  // Como input > threshold, deve gerar spike (1.0)
  ASSERT_FLOAT_EQ(out.data(0, 0), 1.0F);
  // Após spike, v_mem deve ser resetado para zero
  ASSERT_FLOAT_EQ(leaky.v_mem(0, 0), 0.0F);
}

// Teste para Leaky sem reset para zero
TEST(LeakyLayerTest, ForwardSpikeNoResetZero) {
  Leaky leaky(/*dt=*/1.0F, /*R=*/5.0F, /*C=*/1.0F, /*V_thresh=*/2.0F, /*reset_zero=*/false, 0.0F, std::make_shared<ExponentialSurrogate>());
  Eigen::MatrixXf input(1, 1);
  input << 3.0F; // Acima do threshold
  Tensor in_tensor(input);
  Tensor out = leaky.forward(in_tensor);
  // Deve gerar spike
  ASSERT_FLOAT_EQ(out.data(0, 0), 1.0F);
  // v_mem deve ser reduzido pelo threshold
  ASSERT_FLOAT_EQ(leaky.v_mem(0, 0), 1.0F); // 3.0 - 2.0
}
