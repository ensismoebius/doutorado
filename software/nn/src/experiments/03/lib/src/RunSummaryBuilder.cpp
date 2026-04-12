/**
 * @file src/experiments/03/lib/src/RunSummaryBuilder.cpp
 * @brief Implementation for Runsummarybuilder.
 *

 */

#include "RunSummaryBuilder.hpp"

#include "Experiment03Config.hpp"

namespace experiment03
{
auto build_run_summary(const Config& config,
    int exit_code,
    size_t total_samples,
    size_t processed_samples,
    size_t seen_batches,
    const std::vector<float>& epoch_mean_losses,
    const std::string& error_message) -> Summary
{
    Summary s{};
    s.profile_name = config.profile_name;
    s.dataset_type = dataset_type_to_string(config.dataset_type);
    s.autoencoder_type = autoencoder_type_to_string(config.autoencoder_type);
    s.optimizer_type = config.training_optimizer_type;
    s.optimizer_learning_rate = config.training_learning_rate;
    s.optimizer_momentum = config.training_optimizer_momentum;
    s.optimizer_adam_beta1 = config.training_optimizer_adam_beta1;
    s.optimizer_adam_beta2 = config.training_optimizer_adam_beta2;
    s.optimizer_adam_epsilon = config.training_optimizer_adam_epsilon;
    s.exit_code = exit_code;
    s.total_samples = total_samples;
    s.processed_samples = processed_samples;
    s.seen_batches = seen_batches;
    s.epoch_mean_losses = epoch_mean_losses;
    if (!error_message.empty()) s.error_message = error_message;
    return s;
}

} // namespace experiment03
