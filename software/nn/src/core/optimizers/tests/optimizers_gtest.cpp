#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

#include "core/tensor/Tensor.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/optimizers/SGD.hpp"
#include "core/optimizers/SGDMinimal.hpp"

// Test Fixture for common optimizer setup
class OptimizerTest : public ::testing::Test {
protected:
    // Common data for weights and bias
    Eigen::MatrixXf initial_weights_data = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf initial_bias_data = Eigen::MatrixXf::Zero(2, 1);
    Eigen::MatrixXf initial_weights_grad = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf initial_bias_grad = Eigen::MatrixXf::Ones(2, 1);

    nn::Tensor weights;
    nn::Tensor bias;
    std::vector<nn::Tensor*> params;

    OptimizerTest()
        : weights(initial_weights_data), // Initialize Tensor objects
          bias(initial_bias_data)
    {
        params.push_back(&weights);
        params.push_back(&bias);
        weights.set_grad(initial_weights_grad);
        bias.set_grad(initial_bias_grad);
    }

    // Helper to check if a tensor's data has changed from initial
    bool has_data_changed(const nn::Tensor& tensor, const Eigen::MatrixXf& initial_data) const {
        return !tensor.get_data_ref().isApprox(initial_data);
    }
};

TEST_F(OptimizerTest, SGDMinimalOptimizerStepAndZeroGrad)
{
    SGDMinimal sgd_minimal(0.01F);
    sgd_minimal.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    sgd_minimal.zero_grad(params);
    ASSERT_TRUE(weights.get_grad_ref().isZero(1e-6F)); // Use isZero for Eigen matrices
    ASSERT_TRUE(bias.get_grad_ref().isZero(1e-6F));
}

TEST_F(OptimizerTest, AdamOptimizerStepAndZeroGrad)
{
    Adam adam(0.01F);
    adam.attach(params); // Adam requires attaching parameters
    adam.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    adam.zero_grad(params);
    ASSERT_TRUE(weights.get_grad_ref().isZero(1e-6F));
    ASSERT_TRUE(bias.get_grad_ref().isZero(1e-6F));
}

TEST_F(OptimizerTest, SGDOptimizerStepAndZeroGrad)
{
    SGD sgd(0.01F);
    sgd.attach(params); // SGD with momentum requires attaching parameters
    sgd.step(params);
    ASSERT_TRUE(has_data_changed(weights, initial_weights_data));
    sgd.zero_grad(params);
    ASSERT_TRUE(weights.get_grad_ref().isZero(1e-6F));
    ASSERT_TRUE(bias.get_grad_ref().isZero(1e-6F));
}

// New test for empty parameters list
TEST(OptimizerEdgeCases, EmptyParamsList)
{
    std::vector<nn::Tensor*> empty_params;
    SGDMinimal sgd_minimal(0.01F);
    ASSERT_NO_THROW(sgd_minimal.step(empty_params));
    ASSERT_NO_THROW(sgd_minimal.zero_grad(empty_params));

    Adam adam(0.01F);
    ASSERT_NO_THROW(adam.attach(empty_params));
    ASSERT_NO_THROW(adam.step(empty_params));
    ASSERT_NO_THROW(adam.zero_grad(empty_params));

    SGD sgd(0.01F);
    ASSERT_NO_THROW(sgd.attach(empty_params));
    ASSERT_NO_THROW(sgd.step(empty_params));
    ASSERT_NO_THROW(sgd.zero_grad(empty_params));
}