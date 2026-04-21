/**
 * @file src/experiments/03/lib/src/ResultsWriter.cpp
 * @brief Implementation for Resultswriter.
 *

 */

#include "ResultsWriter.hpp"

#include <filesystem>

#include "nlohmann/json.hpp"
#include "nn/io/ReportIO.hpp"

namespace experiment03
{
namespace
{
auto safe_profile_stem(const std::string& profile_name) -> std::string
{
    namespace fs = std::filesystem;
    fs::path p(profile_name);

    std::string stem = p.stem().string();
    if (stem.empty()) stem = profile_name;
    return nn::io::sanitize_stem(stem);
}
} // namespace

auto write_run_summary_json(const Summary& summary, std::string& out_path, std::string& out_error)
    -> bool
{
    namespace fs = std::filesystem;

    const fs::path source_results_dir =
        fs::path(__FILE__).parent_path().parent_path().parent_path() / "results";

    fs::path results_dir = source_results_dir;
    if (!fs::exists(results_dir))
    {
        results_dir = fs::path("results");
    }

    std::error_code ec;
    fs::create_directories(results_dir, ec);
    if (ec)
    {
        out_error = "failed to create results directory: " + ec.message();
        return false;
    }

    const std::string stem =
        nn::io::timestamp_now_compact_local() + "_" + safe_profile_stem(summary.profile_name);
    const fs::path out_file = results_dir / (stem + ".json");

    nlohmann::json payload;
    payload["profile"] = summary.profile_name;
    payload["dataset_type"] = summary.dataset_type;
    payload["autoencoder_type"] = summary.autoencoder_type;
    payload["loss_type"] = summary.loss_type;
    payload["optimizer"]["type"] = summary.optimizer_type;
    payload["optimizer"]["learning_rate"] = summary.optimizer_learning_rate;
    payload["optimizer"]["momentum"] = summary.optimizer_momentum;
    payload["optimizer"]["adam_beta1"] = summary.optimizer_adam_beta1;
    payload["optimizer"]["adam_beta2"] = summary.optimizer_adam_beta2;
    payload["optimizer"]["adam_epsilon"] = summary.optimizer_adam_epsilon;
    payload["optimizer"]["final_learning_rate"] = summary.optimizer_final_learning_rate;
    payload["exit_code"] = summary.exit_code;
    payload["total_samples"] = summary.total_samples;
    payload["processed_samples"] = summary.processed_samples;
    payload["seen_batches"] = summary.seen_batches;
    payload["epoch_mean_losses"] = summary.epoch_mean_losses;
    payload["kfold"]["n_splits"] = summary.kfold_n_splits;
    payload["kfold"]["fold_epoch_val_losses"] = summary.fold_epoch_val_losses;
    payload["kfold"]["fold_epoch_val_eeg_losses"] = summary.fold_epoch_val_eeg_losses;
    payload["kfold"]["fold_epoch_val_audio_losses"] = summary.fold_epoch_val_audio_losses;
    payload["kfold"]["fold_mean_val_losses"] = summary.fold_mean_val_losses;
    payload["kfold"]["mean_val_loss"] = summary.mean_val_loss;
    payload["error"] = summary.error_message;

    if (!nn::io::write_json_file(out_file, payload, 2, &out_error))
    {
        return false;
    }

    out_path = out_file.string();
    return true;
}

} // namespace experiment03
