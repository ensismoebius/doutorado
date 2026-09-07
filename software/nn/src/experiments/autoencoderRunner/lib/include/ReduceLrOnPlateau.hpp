/**
 * @file src/experiments/autoencoderRunner/lib/include/ReduceLrOnPlateau.hpp
 * @brief ReduceLROnPlateauState + LR-plateau helpers (extracted from
 *        autoencoderRunner.cpp). Global scope, matching the scope the
 *        original anonymous namespace occupied (AutoencoderRunner itself is
 *        a global-scope class, not in namespace autoencoderRunner).
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>

#include "cli.hpp"
#include "logging/Logger.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/Optimizer.hpp"
#include "optimizers/SGD.hpp"

struct ReduceLROnPlateauState
{
    float best_val_loss = std::numeric_limits<float>::infinity();
    std::size_t bad_epochs = 0;
};

inline auto optimizer_learning_rate_ptr(Optimizer& optimizer) -> float*
{
    if (auto* adam = dynamic_cast<Adam*>(&optimizer))
    {
        return &adam->learning_rate;
    }
    if (auto* sgd = dynamic_cast<SGD*>(&optimizer))
    {
        return &sgd->learning_rate;
    }
    return nullptr;
}

inline auto apply_reduce_lr_on_plateau(
    Optimizer& optimizer, const Config& config, ReduceLROnPlateauState& state, float epoch_val_loss)
    -> float
{
    float* learning_rate = optimizer_learning_rate_ptr(optimizer);
    if (learning_rate == nullptr || !config.training_lr_plateau_enabled)
    {
        return learning_rate != nullptr ? *learning_rate : config.training_learning_rate;
    }

    const bool improved =
        epoch_val_loss <
        (state.best_val_loss - std::max(0.0F, config.training_lr_plateau_min_delta));
    if (improved)
    {
        state.best_val_loss = epoch_val_loss;
        state.bad_epochs = 0;
        return *learning_rate;
    }

    ++state.bad_epochs;
    if (state.bad_epochs < std::max<std::size_t>(1, config.training_lr_plateau_patience))
    {
        return *learning_rate;
    }

    const float factor = std::clamp(config.training_lr_plateau_factor, 0.0F, 1.0F);
    const float new_lr = std::max(1e-8F, (*learning_rate) * factor);
    if (new_lr < *learning_rate)
    {
        std::ostringstream lr_log;
        lr_log << "ReduceLROnPlateau: val loss plateau detected, lr " << *learning_rate << " -> "
               << new_lr;
        NN_LOG_INFO(lr_log.str());
        *learning_rate = new_lr;
    }
    state.bad_epochs = 0;
    return *learning_rate;
}
