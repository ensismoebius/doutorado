#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "Relu.cpp"
#include "Model.cpp"
#include "Linear.cpp"

// MSE Loss (Mean Squared Error)
class MSELoss {
public:
    // Forward pass to calculate the loss
    double forward(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
        Eigen::MatrixXd diff = predictions - targets;
        return diff.squaredNorm() / diff.rows();
    }

    // Backward pass to calculate gradient of the loss with respect to predictions
    Eigen::MatrixXd backward(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets) {
        Eigen::MatrixXd diff = predictions - targets;
        return (2.0 / diff.rows()) * diff;
    }
};

// Simple gradient descent optimizer
class SGD {
    double learning_rate;

public:
    explicit SGD(double lr) : learning_rate(lr) {}

    void step(Model& model) {
        auto params = model.parameters();
        auto grads = model.gradients();

        for (auto& [name, param] : params) {
            // Update parameters with gradient descent
            param.noalias() -= learning_rate * grads[name];
        }
    }
};

void train(Model& model, MSELoss& mse_loss, SGD& optimizer, 
           const std::vector<Eigen::MatrixXd>& inputs, 
           const std::vector<Eigen::MatrixXd>& targets, 
           int epochs) {
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            Eigen::MatrixXd output = model.forward(inputs[i]);

            // Compute loss
            double loss = mse_loss.forward(output, targets[i]);
            total_loss += loss;

            // Backward pass
            Eigen::MatrixXd grad_loss = mse_loss.backward(output, targets[i]);
            model.backward(grad_loss);

            // Update parameters
            optimizer.step(model);
        }

        // Display loss for this epoch
        std::cout << "Epoch " << epoch << " - Loss: " << total_loss / inputs.size() << std::endl;
    }
}

int main() {
    // Instantiate the model
    Model model({
        new Linear(4, 5),
        new ReLU(),
        new Linear(5, 3),
        new ReLU(),
        new Linear(3, 1)
    });

    // Instantiate MSE loss function
    MSELoss mse_loss;

    // Instantiate optimizer
    SGD optimizer(0.01);

    // Training data (inputs and targets)
    std::vector<Eigen::MatrixXd> inputs = {
        Eigen::MatrixXd::Random(1, 4),
        Eigen::MatrixXd::Random(1, 4),
        Eigen::MatrixXd::Random(1, 4)
    };
    std::vector<Eigen::MatrixXd> targets = {
        Eigen::MatrixXd::Random(1, 1),
        Eigen::MatrixXd::Random(1, 1),
        Eigen::MatrixXd::Random(1, 1)
    };

    // Train the model for 100 epochs
    train(model, mse_loss, optimizer, inputs, targets, 100);

    return 0;
}
