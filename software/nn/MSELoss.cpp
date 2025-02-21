#ifndef MSE_LOSS
#define MSE_LOSS

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

// Quadratic Loss (Mean Squared Error)
class MSELoss
{
public:
    // Forward pass to calculate the loss
    double forward(const MatrixXd &predictions, const MatrixXd &targets)
    {
        // Calculate Mean Squared Error: MSE = 1/N * ||predictions - targets||^2
        MatrixXd diff = predictions - targets;
        return diff.squaredNorm() / diff.rows();
    }

    // Backward pass to calculate gradient of the loss with respect to predictions
    MatrixXd backward(const MatrixXd &predictions, const MatrixXd &targets)
    {
        // Gradient of MSE with respect to predictions: dL/dy = 2/N * (predictions - targets)
        MatrixXd diff = predictions - targets;
        return (2.0 / diff.rows()) * diff;
    }
};

#endif