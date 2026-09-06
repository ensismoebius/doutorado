#pragma once

#include <string>

/**
 * @file OptimizerFactoryConfig.hpp
 * @brief Config struct for OptimizerFactory (extracted from OptimizerFactory.hpp).
 */

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

} // namespace nn::optimizers
