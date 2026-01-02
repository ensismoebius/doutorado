#pragma once
#include <memory>
#include <vector>

#include "../initializers/kaiming_snn.hpp"
#include "Linear.hpp"
#include "Module.hpp"
#include "ReLU.hpp"
#include "ResidualBlock.hpp"
#include "Sequential.hpp"
#include "core/tensor/Tensor.hpp"

// Simple ResNet-like model for classification
class SimpleResNet : public Module
{
   public:
    SimpleResNet(int input_dim, int hidden_dim, int output_dim, int depth = 3)
        : input_dim_(input_dim), output_dim_(output_dim)
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
    int input_dim_;
    int output_dim_;
    std::vector<std::shared_ptr<Module>> layers_;
    std::unique_ptr<Sequential> model_;
};