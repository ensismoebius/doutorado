/**
 * @file ResidualBlock.hpp
 * @brief A small MLP-style residual block (Linear/ReLU/Linear + skip).
 *
 * This is used by the dense `SimpleResNet` model and follows the codebase’s
 * 2D tensor convention: (batch x features).
 */

#ifndef NN_LAYERS_RESIDUALBLOCK_HPP
#define NN_LAYERS_RESIDUALBLOCK_HPP

#include <memory>

#include "nn/layers/activations/ReLU.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/dense/Linear.hpp"

// Simple residual block for MLP: x -> Linear -> ReLU -> Linear + x
//
// This is the classic residual pattern from ResNets adapted to a 2D tensor
// convention (N x D). The skip connection improves gradient flow in deeper
// networks by providing an identity path.
template <typename Backend>
struct ResidualBlockImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;
    std::shared_ptr<LinearImpl<Backend>> fc1;
    std::shared_ptr<ReLUImpl<Backend>> act1;
    std::shared_ptr<LinearImpl<Backend>> fc2;

    explicit ResidualBlockImpl(int features)
        : fc1(std::make_shared<LinearImpl<Backend>>(features, features)),
          act1(std::make_shared<ReLUImpl<Backend>>()),
          fc2(std::make_shared<LinearImpl<Backend>>(features, features))
    {
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        Tensor out = fc1->forward(input, requires_grad);
        out = act1->forward(out, requires_grad);
        out = fc2->forward(out, requires_grad);

        // Add skip connection: assume input and out shapes match (N x D)
        // (No projection layer is provided here; callers must ensure feature dims match.)
        Tensor res(out.rows(), out.cols());
        res = out.add(input);
        return res;
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
        // This mirrors: d(x + f(x))/dx = I + df/dx.
        Tensor total(grad_fc1.rows(), grad_fc1.cols());
        total = grad_fc1.add(grad_output);
        return total;
    }
};

#endif // NN_LAYERS_RESIDUALBLOCK_HPP
