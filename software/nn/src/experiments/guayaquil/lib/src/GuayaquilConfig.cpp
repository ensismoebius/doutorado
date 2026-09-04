#include "../include/GuayaquilConfig.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace guayaquil
{

namespace
{

// One checker per config section. They were one 197-line `validate()` with a
// cyclomatic complexity of 32: twenty independent guards in a single
// function, each appending its complaint to a shared buffer AND setting a
// shared flag.
//
// Each checker below appends exactly what the original appended, in the
// original order, so the accumulated message is unchanged -- validation
// still reports EVERY problem at once rather than stopping at the first,
// which is what makes a bad profile fixable in one pass.
//
// The `has_error` flag is gone: it was true exactly when the buffer was
// non-empty, and keeping the two in sync by hand at twenty call sites is a
// silent failure waiting to happen (append without setting the flag and the
// config validates "successfully" while the complaint goes nowhere).

void check_experiment(const GuayaquilConfig::Experiment& experiment, std::ostringstream& errors)
{
    if (experiment.repeats <= 0)
    {
        errors << "  - experiment.repeats must be > 0 (got " << experiment.repeats << ")\n";
    }

    if (experiment.seed == 0)
    {
        errors << "  - experiment.seed should not be 0 (got " << experiment.seed << ")\n";
    }

    if (experiment.run_tag.empty())
    {
        errors << "  - experiment.run_tag is empty\n";
    }
}

/// Takes the whole config: the window/frame-size rule is a relation BETWEEN
/// the dataset and the model, and splitting it across two checkers would
/// hide that.
void check_dataset(const GuayaquilConfig& config, std::ostringstream& errors)
{
    const auto& dataset = config.dataset;
    const auto& model = config.model;

    if (dataset.window_size <= 0)
    {
        errors << "  - dataset.window_size must be > 0 (got " << dataset.window_size << ")\n";
    }

    if (model.lstm_frame_size <= 0)
    {
        errors << "  - model.lstm_frame_size must be > 0 (got " << model.lstm_frame_size << ")\n";
    }
    else if (dataset.window_size > 0 && (dataset.window_size % model.lstm_frame_size) != 0)
    {
        errors << "  - model.lstm_frame_size (" << model.lstm_frame_size
               << ") must divide dataset.window_size (" << dataset.window_size << ")\n";
    }

    if (dataset.max_loaded_train_samples <= 0)
    {
        errors << "  - dataset.max_loaded_train_samples must be > 0 (got "
               << dataset.max_loaded_train_samples << ")\n";
    }

    if (dataset.max_validation_samples <= 0)
    {
        errors << "  - dataset.max_validation_samples must be > 0 (got "
               << dataset.max_validation_samples << ")\n";
    }
}

/// Also takes the whole config: batch size is checked against the dataset's
/// sample budget.
void check_training(const GuayaquilConfig& config, std::ostringstream& errors)
{
    const auto& training = config.training;
    const auto& dataset = config.dataset;

    if (training.samples_per_batch <= 0)
    {
        errors << "  - training.samples_per_batch must be > 0 (got " << training.samples_per_batch
               << ")\n";
    }

    if (training.samples_per_batch > dataset.max_loaded_train_samples)
    {
        errors << "  - training.samples_per_batch (" << training.samples_per_batch
               << ") exceeds max_loaded_train_samples (" << dataset.max_loaded_train_samples
               << ")\n";
    }

    if (training.epochs <= 0)
    {
        errors << "  - training.epochs must be > 0 (got " << training.epochs << ")\n";
    }

    if (training.early_stop_patience < 0)
    {
        errors << "  - training.early_stop_patience must be >= 0 (got "
               << training.early_stop_patience << ")\n";
    }

    if (training.early_stop_patience >= training.epochs && training.early_stop_patience > 0)
    {
        errors << "  - training.early_stop_patience (" << training.early_stop_patience
               << ") should be < epochs (" << training.epochs << ") for effective early stopping\n";
    }

    if (training.learning_rate <= 0.0f || training.learning_rate > 0.1f)
    {
        errors << "  - training.learning_rate outside typical range [1e-8, 0.1] (got "
               << training.learning_rate << ")\n";
    }

    if (training.max_reconstruct_mean_deviation <= 0.0f)
    {
        errors << "  - training.max_reconstruct_mean_deviation must be > 0 (got "
               << training.max_reconstruct_mean_deviation << ")\n";
    }
}

void check_model(const GuayaquilConfig::Model& model, std::ostringstream& errors)
{
    if (model.encoder_layer_spec.empty())
    {
        errors << "  - model.encoder_layer_spec is empty\n";
    }

    if (model.decoder_layer_spec.empty())
    {
        errors << "  - model.decoder_layer_spec is empty\n";
    }
}

void check_evaluation(const GuayaquilConfig::Evaluation& evaluation, std::ostringstream& errors)
{
    if (evaluation.datasets.empty())
    {
        errors << "  - evaluation.datasets is empty\n";
    }

    if (evaluation.encodings.empty())
    {
        errors << "  - evaluation.encodings is empty\n";
    }

    const std::vector<std::string> valid_encodings = {"direct", "poisson", "latency"};
    for (const auto& enc : evaluation.encodings)
    {
        if (std::find(valid_encodings.begin(), valid_encodings.end(), enc) == valid_encodings.end())
        {
            errors << "  - evaluation.encodings contains unknown encoding: '" << enc << "'\n";
        }
    }

    // The SNN knobs are only meaningful once an SNN architecture is asked
    // for; an empty list means this run is LSTM-only and the thresholds
    // below are legitimately unset.
    if (evaluation.snn_architectures.empty())
    {
        return;
    }

    if (evaluation.v_th_values.empty())
    {
        errors << "  - evaluation.snn_architectures non-empty but v_th_values is empty\n";
    }

    if (evaluation.alpha_values.empty())
    {
        errors << "  - evaluation.snn_architectures non-empty but alpha_values is empty\n";
    }

    for (float vth : evaluation.v_th_values)
    {
        if (vth <= 0.0f)
        {
            errors << "  - v_th_values contains non-positive value: " << vth << "\n";
        }
    }

    for (float alpha : evaluation.alpha_values)
    {
        if (alpha <= 0.0f || alpha >= 1.0f)
        {
            errors << "  - alpha_values must be in (0, 1), got: " << alpha << "\n";
        }
    }

    const std::vector<std::string> valid_archs = {"dense", "conv1d", "recurrent"};
    for (const auto& arch : evaluation.snn_architectures)
    {
        if (std::find(valid_archs.begin(), valid_archs.end(), arch) == valid_archs.end())
        {
            errors << "  - evaluation.snn_architectures contains unknown architecture: '" << arch
                   << "'\n";
        }
    }
}

} // namespace

void GuayaquilConfig::validate() const
{
    std::ostringstream errors;

    check_experiment(experiment, errors);
    check_dataset(*this, errors);
    check_training(*this, errors);
    check_model(model, errors);
    check_evaluation(evaluation, errors);

    // Every check appends to `errors` and nothing else writes to it, so a
    // non-empty buffer IS the failure condition. The `has_error` flag this
    // replaces had to be set by hand next to each of the twenty appends --
    // one forgotten `has_error = true` would have printed a complaint into
    // a buffer nobody ever looked at, and reported the config as valid.
    if (errors.tellp() != 0)
    {
        throw std::invalid_argument("GuayaquilConfig validation failed:\n" + errors.str());
    }
}

} // namespace guayaquil
