#pragma once

#include <string>
#include <vector>

#include "cli.hpp"

namespace experiment03
{
struct Summary
{
    std::string profile_name;
    std::string dataset_type;
    std::string autoencoder_type;
    int exit_code = 0;
    std::size_t total_samples = 0;
    std::size_t processed_samples = 0;
    std::size_t seen_batches = 0;
    std::vector<float> epoch_mean_losses;
    std::string error_message;
};

// Write run summary to src/experiments/03/results or ./results when available.
// Returns true on success and fills out_path; otherwise returns false and sets out_error.
auto write_run_summary_json(const Summary& summary, std::string& out_path, std::string& out_error)
    -> bool;
} // namespace experiment03
