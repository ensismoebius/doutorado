#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace meeting01
{

// One reconstruction-error record for a single window under one trained model. The
// identity fields come straight from FsddWindowDataset::WindowMetadata and are stable
// across every model, so downstream statistics can pair errors at the window,
// recording, or speaker level (recording-level is the paper's primary estimand).
struct PerWindowError
{
    std::string model;        // "lstm-ae" | "gru-ae" | "transformer-ae" | "snn-ae"
    std::string encoding;     // "direct" | "poisson" | "latency"
    std::string architecture; // "lstm"/"gru"/"transformer" or the SNN arch
    float v_th = 0.0f;
    float alpha = 0.0f;
    int run_id = 0;
    std::uint32_t seed = 0u;
    int cv_fold = -1;
    std::string split; // "val" | "test"

    int speaker_id = -1;
    int recording_id = -1;
    int window_id = -1;
    int source_window_index = -1;

    float mse = 0.0f;
    float mae = 0.0f;
};

// Writes `rows` -- every model, encoding and seed of one fold, collected over the whole
// fold -- to `<path>` as CSV, REPLACING any file already there: the file is one fold
// run's rows, never a mix of two runs. (03_meeting01_pca_mean_baselines.py then adds the
// mean / pca reference rows to it.)
void write_per_window_errors_csv(
    const std::filesystem::path& path, const std::vector<PerWindowError>& rows);

} // namespace meeting01
