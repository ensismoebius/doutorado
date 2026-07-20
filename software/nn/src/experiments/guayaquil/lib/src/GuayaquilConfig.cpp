#include "../include/E04Config.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace e04
{

void E04Config::validate() const
{
    std::ostringstream errors;
    bool has_error = false;

    // Experiment section
    if (experiment.repeats <= 0)
    {
        errors << "  - experiment.repeats must be > 0 (got " << experiment.repeats << ")\n";
        has_error = true;
    }

    if (experiment.seed == 0)
    {
        errors << "  - experiment.seed should not be 0 (got " << experiment.seed << ")\n";
        has_error = true;
    }

    if (experiment.run_tag.empty())
    {
        errors << "  - experiment.run_tag is empty\n";
        has_error = true;
    }

    // Dataset section
    if (dataset.window_size <= 0)
    {
        errors << "  - dataset.window_size must be > 0 (got " << dataset.window_size << ")\n";
        has_error = true;
    }

    if (model.lstm_frame_size <= 0)
    {
        errors << "  - model.lstm_frame_size must be > 0 (got " << model.lstm_frame_size << ")\n";
        has_error = true;
    }
    else if (dataset.window_size > 0 && (dataset.window_size % model.lstm_frame_size) != 0)
    {
        errors << "  - model.lstm_frame_size (" << model.lstm_frame_size
               << ") must divide dataset.window_size (" << dataset.window_size << ")\n";
        has_error = true;
    }

    if (dataset.max_loaded_train_samples <= 0)
    {
        errors << "  - dataset.max_loaded_train_samples must be > 0 (got "
               << dataset.max_loaded_train_samples << ")\n";
        has_error = true;
    }

    if (dataset.max_validation_samples <= 0)
    {
        errors << "  - dataset.max_validation_samples must be > 0 (got "
               << dataset.max_validation_samples << ")\n";
        has_error = true;
    }

    // Training section
    if (training.samples_per_batch <= 0)
    {
        errors << "  - training.samples_per_batch must be > 0 (got " << training.samples_per_batch
               << ")\n";
        has_error = true;
    }

    if (training.samples_per_batch > dataset.max_loaded_train_samples)
    {
        errors << "  - training.samples_per_batch (" << training.samples_per_batch
               << ") exceeds max_loaded_train_samples (" << dataset.max_loaded_train_samples
               << ")\n";
        has_error = true;
    }

    if (training.epochs <= 0)
    {
        errors << "  - training.epochs must be > 0 (got " << training.epochs << ")\n";
        has_error = true;
    }

    if (training.early_stop_patience < 0)
    {
        errors << "  - training.early_stop_patience must be >= 0 (got "
               << training.early_stop_patience << ")\n";
        has_error = true;
    }

    if (training.early_stop_patience >= training.epochs && training.early_stop_patience > 0)
    {
        errors << "  - training.early_stop_patience (" << training.early_stop_patience
               << ") should be < epochs (" << training.epochs << ") for effective early stopping\n";
        has_error = true;
    }

    if (training.learning_rate <= 0.0f || training.learning_rate > 0.1f)
    {
        errors << "  - training.learning_rate outside typical range [1e-8, 0.1] (got "
               << training.learning_rate << ")\n";
        has_error = true;
    }

    if (training.max_reconstruct_mean_deviation <= 0.0f)
    {
        errors << "  - training.max_reconstruct_mean_deviation must be > 0 (got "
               << training.max_reconstruct_mean_deviation << ")\n";
        has_error = true;
    }

    // Model section
    if (model.encoder_layer_spec.empty())
    {
        errors << "  - model.encoder_layer_spec is empty\n";
        has_error = true;
    }

    if (model.decoder_layer_spec.empty())
    {
        errors << "  - model.decoder_layer_spec is empty\n";
        has_error = true;
    }

    // Evaluation section
    if (evaluation.datasets.empty())
    {
        errors << "  - evaluation.datasets is empty\n";
        has_error = true;
    }

    if (evaluation.encodings.empty())
    {
        errors << "  - evaluation.encodings is empty\n";
        has_error = true;
    }

    // Validate encoding names
    const std::vector<std::string> valid_encodings = {"direct", "poisson", "latency"};
    for (const auto& enc : evaluation.encodings)
    {
        if (std::find(valid_encodings.begin(), valid_encodings.end(), enc) == valid_encodings.end())
        {
            errors << "  - evaluation.encodings contains unknown encoding: '" << enc << "'\n";
            has_error = true;
        }
    }

    // SNN-specific validation (if architectures specified)
    if (!evaluation.snn_architectures.empty())
    {
        if (evaluation.v_th_values.empty())
        {
            errors << "  - evaluation.snn_architectures non-empty but v_th_values is empty\n";
            has_error = true;
        }

        if (evaluation.alpha_values.empty())
        {
            errors << "  - evaluation.snn_architectures non-empty but alpha_values is empty\n";
            has_error = true;
        }

        // Validate v_th values are positive
        for (float vth : evaluation.v_th_values)
        {
            if (vth <= 0.0f)
            {
                errors << "  - v_th_values contains non-positive value: " << vth << "\n";
                has_error = true;
            }
        }

        // Validate alpha values are in (0, 1)
        for (float alpha : evaluation.alpha_values)
        {
            if (alpha <= 0.0f || alpha >= 1.0f)
            {
                errors << "  - alpha_values must be in (0, 1), got: " << alpha << "\n";
                has_error = true;
            }
        }

        // Validate architecture names
        const std::vector<std::string> valid_archs = {"dense", "conv1d", "recurrent"};
        for (const auto& arch : evaluation.snn_architectures)
        {
            if (std::find(valid_archs.begin(), valid_archs.end(), arch) == valid_archs.end())
            {
                errors << "  - evaluation.snn_architectures contains unknown architecture: '"
                       << arch << "'\n";
                has_error = true;
            }
        }
    }

    if (has_error)
    {
        throw std::invalid_argument("E04Config validation failed:\n" + errors.str());
    }
}

} // namespace e04
