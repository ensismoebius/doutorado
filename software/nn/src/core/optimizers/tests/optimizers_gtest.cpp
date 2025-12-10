#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "core/tensor/Tensor.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/optimizers/SGD.hpp"
#include "core/optimizers/SGDMinimal.hpp"

TEST(SGDMinimalOptimizerTest, StepAndZeroGrad)
{
    Eigen::MatrixXf w_data = Eigen::MatrixXf::Ones(2, 2);
    nn::Tensor w(w_data);
    Eigen::MatrixXf b_data = Eigen::MatrixXf::Zero(2, 1);
    nn::Tensor b(b_data);
    std::vector<nn::Tensor*> params = {&w, &b};
    SGDMinimal sgd_minimal(0.01F);
    Eigen::MatrixXf w_grad = Eigen::MatrixXf::Ones(2, 2);
    w.set_grad(w_grad);
    Eigen::MatrixXf b_grad = Eigen::MatrixXf::Ones(2, 1);
    b.set_grad(b_grad);
    sgd_minimal.step(params);
    ASSERT_NE(w.get_data_ref()(0, 0), 1.0F);
    sgd_minimal.zero_grad(params);
    ASSERT_EQ(w.get_grad_ref()(0, 0), 0.0F);
}

TEST(AdamOptimizerTest, StepAndZeroGrad)
{
    Eigen::MatrixXf weights_data = Eigen::MatrixXf::Ones(2, 2);
    nn::Tensor weights(weights_data);
    Eigen::MatrixXf bias_data = Eigen::MatrixXf::Zero(2, 1);
    nn::Tensor bias(bias_data);
    std::vector<nn::Tensor*> params = {&weights, &bias};
    Adam adam(0.01F);
    adam.attach(params);
    Eigen::MatrixXf weights_grad = Eigen::MatrixXf::Ones(2, 2);
    weights.set_grad(weights_grad);
    Eigen::MatrixXf bias_grad = Eigen::MatrixXf::Ones(2, 1);
    bias.set_grad(bias_grad);
    adam.step(params);
    ASSERT_NE(weights.get_data_ref()(0, 0), 1.0F);
    adam.zero_grad(params);
    ASSERT_EQ(weights.get_grad_ref()(0, 0), 0.0F);
}

TEST(SGDOptimizerTest, StepAndZeroGrad)
{
    Eigen::MatrixXf weights_data = Eigen::MatrixXf::Ones(2, 2);
    nn::Tensor weights(weights_data);
    Eigen::MatrixXf bias_data = Eigen::MatrixXf::Zero(2, 1);
    nn::Tensor bias(bias_data);
    std::vector<nn::Tensor*> params = {&weights, &bias};
    SGD sgd(0.01F);
    sgd.attach(params);
    Eigen::MatrixXf weights_grad = Eigen::MatrixXf::Ones(2, 2);
    weights.set_grad(weights_grad);
    Eigen::MatrixXf bias_grad = Eigen::MatrixXf::Ones(2, 1);
    bias.set_grad(bias_grad);
    sgd.step(params);
    ASSERT_NE(weights.get_data_ref()(0, 0), 1.0F);
    sgd.zero_grad(params);
    ASSERT_EQ(weights.get_grad_ref()(0, 0), 0.0F);
}
