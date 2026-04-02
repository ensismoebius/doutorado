#pragma once

#include <string>
#include <vector>

#include "ResultsWriter.hpp"

namespace experiment03
{
auto build_run_summary(const Config& config,
    int exit_code,
    size_t total_samples,
    size_t processed_samples,
    size_t seen_batches,
    const std::vector<float>& epoch_mean_losses,
    const std::string& error_message = "") -> Summary;

} // namespace experiment03
