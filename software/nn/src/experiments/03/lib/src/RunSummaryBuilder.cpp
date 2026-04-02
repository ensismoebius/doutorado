#include "RunSummaryBuilder.hpp"


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
    s.exit_code = exit_code;
    s.total_samples = total_samples;
    s.processed_samples = processed_samples;
    s.seen_batches = seen_batches;
    s.epoch_mean_losses = epoch_mean_losses;
    if (!error_message.empty()) s.error_message = error_message;
    return s;
}

} // namespace experiment03
