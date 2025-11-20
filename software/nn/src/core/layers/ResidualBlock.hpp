#pragma once

#include <memory>

#include "Linear.hpp"
#include "Module.hpp"
#include "ReLU.hpp"

// Simple residual block for MLP: x -> Linear -> ReLU -> Linear + x
struct ResidualBlock : public Module
{
    std::shared_ptr<Linear> fc1;
    std::shared_ptr<ReLU> act1;
    std::shared_ptr<Linear> fc2;

    ResidualBlock(int features)
    {
        fc1 = std::make_shared<Linear>(features, features);
        act1 = std::make_shared<ReLU>();
        fc2 = std::make_shared<Linear>(features, features);
    }

    auto forward(const Tensor& input) -> Tensor override
    {
        Tensor out = fc1->forward(input);
        out = act1->forward(out);
        out = fc2->forward(out);

        // Add skip connection: assume input and out shapes match (N x D)
        Eigen::MatrixXf res = out.get_data_ref() + input.get_data_ref();
        return Tensor{res};
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // Backprop through addition: gradient passes to both paths
        // grad w.r.t. fc2 output is grad_output
        Tensor grad_fc2 = fc2->backward(grad_output);

        // grad through ReLU and fc1: first compute grad for that branch
        Tensor grad_act = act1->backward(grad_fc2);
        Tensor grad_fc1 = fc1->backward(grad_act);

        // Total gradient w.r.t. input is grad from main branch (grad_fc2 propagated through fc2->)
        // + grad from skip (identity)
        Eigen::MatrixXf total = grad_fc1.get_data_ref() + grad_output.get_data_ref();
        return Tensor{total};
    }
};
