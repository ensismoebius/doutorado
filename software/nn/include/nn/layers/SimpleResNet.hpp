/**
 * @file SimpleResNet.hpp
 * @brief Dense (MLP) ResNet-like classifier built from `ResidualBlock`s.
 *
 * This is a convenience model used for quick experiments/demos, not an image ResNet.
 */

#pragma once
#include <memory>
#include <vector>

#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Module.hpp"
#include "nn/layers/ReLU.hpp"
#include "nn/layers/ResidualBlock.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/tensor/Tensor.hpp"

// Simple ResNet-like model for classification
//
// This is a *dense* (MLP) residual network, not the image ResNet architecture.
// It uses `ResidualBlock` (Linear/ReLU/Linear + skip) stacked `depth` times.
//
// Design notes:
// - Internally it builds a `Sequential` and delegates `forward/backward/params/train`.
// - Weight initialization uses `kaimingSNNInitializer()` on the Linear layers.
//   This is applied only to known Linear modules (fc_in/fc_out and residual block
//   linears).
class SimpleResNet : public Module
{
   public:
    SimpleResNet(int input_dim, int hidden_dim, int output_dim, int depth = 3)
    {
        // Build model: input -> Linear -> ReLU -> ResidualBlocks -> Linear(output)
        auto fc_in = std::make_shared<Linear>(input_dim, hidden_dim);
        layers_.push_back(fc_in);
        layers_.push_back(std::make_shared<ReLU>());

        for (int i = 0; i < depth; ++i)
        {
            auto rb = std::make_shared<ResidualBlock>(hidden_dim);
            layers_.push_back(rb);
        }

        auto fc_out = std::make_shared<Linear>(hidden_dim, output_dim);
        layers_.push_back(fc_out);

        model_ = std::make_unique<Sequential>(layers_);

        // Initialize weights
        kaimingSNNInitializer(fc_in);
        kaimingSNNInitializer(fc_out);
        for (auto& layer : layers_)
        {
            if (auto rb = std::dynamic_pointer_cast<ResidualBlock>(layer))
            {
                kaimingSNNInitializer(rb->fc1);
                kaimingSNNInitializer(rb->fc2);
            }
        }
    }

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        return model_->forward(input, requires_grad);
    }

    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        return model_->backward(grad_output);
    }

    auto params() -> std::vector<nn::Tensor*> override
    {
        return model_->params();
    }

    void train(bool on) override
    {
        model_->train(on);
    }

   private:
    std::vector<std::shared_ptr<Module>> layers_;
    std::unique_ptr<Sequential> model_;
};