#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include "optimizers/Adam.hpp"
#include "optimizers/Lion.hpp"
#include "optimizers/Optimizer.hpp"
#include "optimizers/SGD.hpp"
#include "optimizers/ScheduleFreeAdamW.hpp"

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

/**
 * @brief The reference/paper default learning rate for an optimizer token.
 *
 * These are NOT interchangeable. Each optimizer's usable lr follows from how it forms its
 * update: Lion's is `±lr` on every coordinate regardless of gradient magnitude, so its
 * usable lr is an order of magnitude below Adam's; Schedule-Free AdamW's averaging makes it
 * tolerate a slightly larger one. Reusing one optimizer's lr for another is the single
 * easiest way to produce a bogus ablation ("optimizer X is worse") that actually measures
 * the learning rate.
 *
 * Values are each optimizer's own published/reference default:
 *   adam                1e-3    (Kingma & Ba, ICLR 2015; torch.optim.Adam)
 *   sgd                 1e-2    (torch.optim.SGD)
 *   lion                1e-4    (Chen et al., NeurIPS 2023; lion-pytorch)
 *   schedule-free-adamw 2.5e-3  (Defazio et al., NeurIPS 2024; schedulefree)
 *
 * Single source of truth: E05Config resolves an unspecified `training.learning_rate` from
 * here, and the run summary records the value actually used.
 *
 * @throws std::runtime_error on an unknown token (same set as create()).
 */
inline auto reference_learning_rate(const std::string& type) -> float
{
    std::string token = type;
    std::transform(token.begin(),
        token.end(),
        token.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (token == "adam") return 1e-3F;
    if (token == "sgd") return 1e-2F;
    if (token == "lion") return 1e-4F;
    if (token == "schedule-free-adamw" || token == "schedulefree" || token == "schedule_free_adamw")
        return 2.5e-3F;

    throw std::runtime_error("reference_learning_rate: unsupported optimizer type: " + type +
                             " (supported: adam, sgd, lion, schedule-free-adamw)");
}

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

        // Each of the following keeps its paper/reference default hyperparameters except
        // for the learning rate, which the caller always supplies. Note the reference
        // defaults differ a lot between methods (Adam 1e-3, Lion 1e-4, Schedule-Free
        // 2.5e-3), so a single lr is NOT comparable across optimizers -- an ablation must
        // tune lr per optimizer to be meaningful.
        if (token == "lion")
        {
            return std::make_unique<Lion>(learning_rate);
        }

        if (token == "schedule-free-adamw" || token == "schedulefree" ||
            token == "schedule_free_adamw")
        {
            return std::make_unique<ScheduleFreeAdamW>(
                learning_rate, adam_beta1, adam_beta2, adam_epsilon);
        }

        throw std::runtime_error("Unsupported optimizer type: " + type +
                                 " (supported: adam, sgd, lion, schedule-free-adamw)");
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
