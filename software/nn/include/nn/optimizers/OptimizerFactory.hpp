#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "nn/optimizers/Adam.hpp"
#include "nn/optimizers/Optimizer.hpp"
#include "nn/optimizers/SGD.hpp"

namespace nn::optimizers
{

struct OptimizerFactoryConfig
{
    std::string type = "adam";
    float learning_rate = 0.001F;
    float momentum = 0.0F;
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1e-8F;
};

class OptimizerFactory
{
   public:
    static auto create(const std::string& type,
        float learning_rate,
        float momentum = 0.0F,
        float adam_beta1 = 0.9F,
        float adam_beta2 = 0.999F,
        float adam_epsilon = 1e-8F) -> std::unique_ptr<::Optimizer>
    {
        const std::string token = normalize_token(type);

        if (token == "adam")
        {
            return std::make_unique<Adam>(learning_rate, adam_beta1, adam_beta2, adam_epsilon);
        }

        if (token == "sgd")
        {
            return std::make_unique<SGD>(learning_rate, momentum);
        }

        throw std::runtime_error("Unsupported optimizer type: " + type);
    }

    static auto create(const OptimizerFactoryConfig& config) -> std::unique_ptr<::Optimizer>
    {
        return create(config.type,
            config.learning_rate,
            config.momentum,
            config.adam_beta1,
            config.adam_beta2,
            config.adam_epsilon);
    }

   private:
    static auto normalize_token(const std::string& value) -> std::string
    {
        std::string token = value;
        std::transform(token.begin(),
            token.end(),
            token.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return token;
    }
};

} // namespace nn::optimizers
