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

    explicit ResidualBlock(int features)
        : fc1(std::make_shared<Linear>(features, features)),
          act1(std::make_shared<ReLU>()),
          fc2(std::make_shared<Linear>(features, features))
    {
    }

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        nn::Tensor out = fc1->forward(input, requires_grad);
        out = act1->forward(out, requires_grad);
        out = fc2->forward(out, requires_grad);

        // Add skip connection: assume input and out shapes match (N x D)
        nn::Tensor res(out.rows(), out.cols());
        res = out.add(input);
        return res;
    }

    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // Backprop through addition: gradient passes to both paths
        // grad w.r.t. fc2 output is grad_output
        nn::Tensor grad_fc2 = fc2->backward(grad_output);

        // grad through ReLU and fc1: first compute grad for that branch
        nn::Tensor grad_act = act1->backward(grad_fc2);
        nn::Tensor grad_fc1 = fc1->backward(grad_act);

        // Total gradient w.r.t. input is grad from main branch (grad_fc2 propagated through fc2->)
        // + grad from skip (identity)
        nn::Tensor total(grad_fc1.rows(), grad_fc1.cols());
        total = grad_fc1.add(grad_output);
        return total;
    }
};
