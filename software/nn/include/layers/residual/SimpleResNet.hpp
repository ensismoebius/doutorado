/**
 * @file SimpleResNet.hpp
 * @brief Dense (MLP) ResNet-like classifier built from `ResidualBlock`s.
 *
 * This is a convenience model used for quick experiments/demos, not an image ResNet.
 */

#ifndef NN_LAYERS_SIMPLERESNET_HPP
#define NN_LAYERS_SIMPLERESNET_HPP
#include <memory>
#include <vector>

#include "initializers/kaiming_snn.hpp"
#include "layers/activations/ReLU.hpp"
#include "layers/base/Module.hpp"
#include "layers/base/Sequential.hpp"
#include "layers/dense/Linear.hpp"
#include "layers/residual/ResidualBlock.hpp"
#include "tensor/Tensor.hpp"

// Simple ResNet-like model for classification
//
// This is a *dense* (MLP) residual network, not the image ResNet architecture.
// It uses `ResidualBlockImpl<Backend>` (Linear/ReLU/Linear + skip) stacked `depth` times.
//
// Design notes:
// - Internally it builds a `SequentialImpl<Backend>` and delegates `forward/backward/params/train`.
// - Weight initialization uses `kaimingSNNInitializer()` on the Linear layers.
//   This is applied only to known Linear modules (fc_in/fc_out and residual block
//   linears).
template <typename Backend>
class SimpleResNetImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    SimpleResNetImpl(int input_dim, int hidden_dim, int output_dim, int depth = 3)
    {
        // Build model: input -> Linear -> ReLU -> ResidualBlocks -> Linear(output)
        auto fc_in = std::make_shared<LinearImpl<Backend>>(input_dim, hidden_dim);
        layers_.push_back(fc_in);
        layers_.push_back(std::make_shared<ReLUImpl<Backend>>());

        for (int i = 0; i < depth; ++i)
        {
            auto rb = std::make_shared<ResidualBlockImpl<Backend>>(hidden_dim);
            layers_.push_back(rb);
        }

        auto fc_out = std::make_shared<LinearImpl<Backend>>(hidden_dim, output_dim);
        layers_.push_back(fc_out);

        model_ = std::make_unique<SequentialImpl<Backend>>(layers_);

        // Initialize weights
        kaimingSNNInitializer(fc_in);
        kaimingSNNInitializer(fc_out);
        for (auto& layer : layers_)
        {
            if (auto rb = std::dynamic_pointer_cast<ResidualBlockImpl<Backend>>(layer))
            {
                kaimingSNNInitializer(rb->fc1);
                kaimingSNNInitializer(rb->fc2);
            }
        }
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        return model_->forward(input, requires_grad);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        return model_->backward(grad_output);
    }

    auto params() -> std::span<Tensor*> override
    {
        return model_->params();
    }

    void train(bool on) override
    {
        model_->train(on);
    }

   private:
    std::vector<std::shared_ptr<Module<Backend>>> layers_;
    std::unique_ptr<SequentialImpl<Backend>> model_;
};
#endif // NN_LAYERS_SIMPLERESNET_HPP
