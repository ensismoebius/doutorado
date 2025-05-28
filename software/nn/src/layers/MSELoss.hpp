#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include "Module.hpp"
#include "../tensor/Tensor.hpp"
#include <Eigen/Dense>

class MSELoss : public Module {
public:
    Tensor last_input;
    Tensor last_target;

    MSELoss() = default;

    // Forward computes the loss value as a Tensor (scalar)
    auto forward(const Tensor &prediction) -> Tensor override {
        // Use last_target set by set_target
        Eigen::MatrixXf diff = prediction.data - last_target.data;
        float loss = diff.array().square().mean();
        return {Eigen::MatrixXf::Constant(1, 1, loss)};
    }

    // Set the target tensor for the loss
    void set_target(const Tensor &target) {
        last_target = target;
    }

    // Backward computes the gradient of the loss w.r.t. prediction
    auto backward(const Tensor &prediction) -> Tensor override {
        Eigen::MatrixXf grad = 2.0F * (prediction.data - last_target.data) / prediction.data.rows();
        return {grad};
    }
};

#endif // MSELOSS_HPP
