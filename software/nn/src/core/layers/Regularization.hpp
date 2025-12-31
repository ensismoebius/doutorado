#ifndef REGULARIZATION_HPP
#define REGULARIZATION_HPP

// Include necessary headers for standard vectors and the Tensor class
#include <vector>

#include "../tensor/Tensor.hpp"

// Base class for regularization techniques
// Provides a common interface for computing regularization penalties and their gradients
class Regularization
{
   protected:
    // Regularization strength parameter (lambda)
    // Controls the amount of penalty applied to the model parameters
    float lambda_;

   public:
    // Constructor that initializes the regularization strength
    // @param lambda: The regularization coefficient (must be non-negative)
    explicit Regularization(float lambda) : lambda_(lambda) {}

    // Virtual destructor for proper polymorphic behavior
    virtual ~Regularization() = default;

    // Pure virtual method to compute the regularization penalty
    // @param params: Vector of pointers to tensors containing model parameters
    // @return: A scalar tensor containing the total regularization penalty
    virtual auto forward(const std::vector<nn::Tensor*>& params) -> nn::Tensor = 0;

    // Pure virtual method to compute and accumulate gradients of the regularization penalty
    // @param params: Vector of pointers to tensors containing model parameters
    //               Gradients are accumulated into the tensors' grad matrices
    virtual void backward(const std::vector<nn::Tensor*>& params) = 0;
};

// L1 regularization (Lasso regularization)
// Penalizes the sum of absolute values of parameters
// Encourages sparsity in the parameter values
class L1Regularization : public Regularization
{
   public:
    // Constructor that initializes the L1 regularization strength
    // @param lambda: The L1 regularization coefficient
    explicit L1Regularization(float lambda) : Regularization(lambda) {}

    // Computes the L1 regularization penalty: lambda * sum(|param|)
    // @param params: Vector of pointers to parameter tensors
    // @return: Scalar tensor containing the L1 penalty value
    auto forward(const std::vector<nn::Tensor*>& params) -> nn::Tensor override
    {
        // Initialize penalty accumulator
        float penalty = 0.0F;

        // Iterate through all parameter tensors
        for (const auto* param : params)
        {
            // Get reference to the parameter data matrix
            const auto& data = param->get_data_ref();

            // Add the sum of absolute values of this parameter tensor to the penalty
            penalty += data.array().abs().sum();
        }

        // Scale the total penalty by the regularization strength
        penalty *= lambda_;

        // Create a 1x1 tensor to hold the scalar penalty value
        nn::Tensor loss(1, 1);
        loss.at(0, 0) = penalty;

        // Return the penalty as a tensor
        return loss;
    }

    // Computes and accumulates gradients for L1 regularization: lambda * sign(param)
    // @param params: Vector of pointers to parameter tensors
    //               Gradients are added to the existing gradients in param->grad
    void backward(const std::vector<nn::Tensor*>& params) override
    {
        // Iterate through all parameter tensors
        for (auto* param : params)
        {
            // Get references to the parameter data and gradient matrices
            const auto& data = param->get_data_ref();
            auto& grad = param->get_grad_ref();

            // Accumulate L1 gradient: lambda * sign(data)
            // sign() returns -1 for negative, 0 for zero, +1 for positive values
            grad += lambda_ * data.array().sign().matrix();
        }
    }
};

// L2 regularization (Ridge regularization)
// Penalizes the sum of squared parameter values
// Encourages smaller parameter values and reduces overfitting
class L2Regularization : public Regularization
{
   public:
    // Constructor that initializes the L2 regularization strength
    // @param lambda: The L2 regularization coefficient
    explicit L2Regularization(float lambda) : Regularization(lambda) {}

    // Computes the L2 regularization penalty: lambda * sum(param^2)
    // @param params: Vector of pointers to parameter tensors
    // @return: Scalar tensor containing the L2 penalty value
    auto forward(const std::vector<nn::Tensor*>& params) -> nn::Tensor override
    {
        // Initialize penalty accumulator
        float penalty = 0.0F;

        // Iterate through all parameter tensors
        for (const auto* param : params)
        {
            // Get reference to the parameter data matrix
            const auto& data = param->get_data_ref();

            // Add the sum of squared values of this parameter tensor to the penalty
            penalty += data.array().square().sum();
        }

        // Scale the total penalty by the regularization strength
        penalty *= lambda_;

        // Create a 1x1 tensor to hold the scalar penalty value
        nn::Tensor loss(1, 1);
        loss.at(0, 0) = penalty;

        // Return the penalty as a tensor
        return loss;
    }

    // Computes and accumulates gradients for L2 regularization: 2 * lambda * param
    // @param params: Vector of pointers to parameter tensors
    //               Gradients are added to the existing gradients in param->grad
    void backward(const std::vector<nn::Tensor*>& params) override
    {
        // Iterate through all parameter tensors
        for (auto* param : params)
        {
            // Get references to the parameter data and gradient matrices
            const auto& data = param->get_data_ref();
            auto& grad = param->get_grad_ref();

            // Accumulate L2 gradient: 2 * lambda * data
            // The derivative of lambda * sum(x^2) with respect to x is 2 * lambda * x
            grad += 2.0F * lambda_ * data;
        }
    }
};

#endif // REGULARIZATION_HPP