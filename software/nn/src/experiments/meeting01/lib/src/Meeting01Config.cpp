#include "../include/Meeting01Config.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace meeting01
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

void check_experiment(const Meeting01Config::Experiment& experiment, std::ostringstream& errors)
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
void check_dataset(const Meeting01Config& config, std::ostringstream& errors)
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

    // The pooled/shuffled split (cv_fold < 0) was removed 2026-09-23 -- it let the
    // same speaker/recording land in both train and validation (the leakage defect
    // a reviewer flagged as strong-reject on submission 71). Every profile now runs
    // nested leave-one-group-out, which uses every window of the speaker-disjoint
    // partitions rather than a fixed pooled sample budget, so cv_fold is required
    // and max_loaded_train_samples/max_validation_samples are no longer checked
    // here (they are accepted but ignored -- see dataset.loso_max_* for the real,
    // per-fold caps).
    if (dataset.cv_fold < 0)
    {
        errors << "  - dataset.cv_fold must be set (>= 0). Remedy: add cv_fold (and "
                  "cv_num_folds) to the profile's dataset block, or pass --cv-fold on "
                  "the CLI.\n";
    }

    if (dataset.cv_fold >= 0 && dataset.cv_fold >= dataset.cv_num_folds)
    {
        errors << "  - dataset.cv_fold (" << dataset.cv_fold << ") must be < dataset.cv_num_folds ("
               << dataset.cv_num_folds << ")\n";
    }

    if (dataset.cv_num_folds < 2)
    {
        errors << "  - dataset.cv_num_folds must be >= 2 (got " << dataset.cv_num_folds << ")\n";
    }
}

/// Also takes the whole config: batch size is checked against the dataset's
/// sample budget.
void check_training(const Meeting01Config& config, std::ostringstream& errors)
{
    const auto& training = config.training;

    if (training.samples_per_batch <= 0)
    {
        errors << "  - training.samples_per_batch must be > 0 (got " << training.samples_per_batch
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

void check_model(const Meeting01Config::Model& model, std::ostringstream& errors)
{
    if (model.encoder_layer_spec.empty())
    {
        errors << "  - model.encoder_layer_spec is empty\n";
    }

    if (model.decoder_layer_spec.empty())
    {
        errors << "  - model.decoder_layer_spec is empty\n";
    }

    // Rejected rather than clamped: at 1 step the membrane has no history, so `beta`
    // multiplies a zero-initialised v_mem and the alpha/v_th knobs stop doing anything,
    // while poisson/latency coding lose the axis they encode on. That failure is silent
    // -- training completes and reports a plausible MSE -- so it has to fail loudly here.
    if (model.time_steps < 2)
    {
        errors << "  - model.time_steps must be >= 2 (got " << model.time_steps
               << "); 1 step disables membrane dynamics and spike coding entirely\n";
    }

    if (model.denoising_noise_std < 0.0f)
    {
        errors << "  - model.denoising_noise_std must be >= 0 (got " << model.denoising_noise_std
               << ")\n";
    }
}

// Fields every family's GA block shares (search mechanics, not genome bounds) — one
// check reused by SNN/LSTM/GRU/Transformer instead of four copies.
template <typename T>
void check_ga_common(const std::string& prefix, const T& ga, std::ostringstream& errors)
{
    if (ga.population_size <= 0)
    {
        errors << "  - " << prefix << ".population_size must be > 0 (got " << ga.population_size
               << ")\n";
    }
    if (ga.generations < 0)
    {
        errors << "  - " << prefix << ".generations must be >= 0 (got " << ga.generations << ")\n";
    }
    // Breeding needs two distinct parents. population_size == 1 with generations >= 1
    // used to be accepted and then hang forever in tournament()'s rejection loop, which
    // on a cluster looks like a job that runs for days and produces nothing.
    if (ga.population_size < 2 && ga.generations >= 1)
    {
        errors << "  - " << prefix << ".population_size must be >= 2 when generations >= 1 (got "
               << ga.population_size << " with " << ga.generations
               << " generations); breeding needs two distinct parents\n";
    }
    if (ga.crossover_prob < 0.0 || ga.crossover_prob > 1.0)
    {
        errors << "  - " << prefix << ".crossover_prob must be in [0, 1] (got " << ga.crossover_prob
               << ")\n";
    }
    if (ga.mutation_prob < 0.0 || ga.mutation_prob > 1.0)
    {
        errors << "  - " << prefix << ".mutation_prob must be in [0, 1] (got " << ga.mutation_prob
               << ")\n";
    }
    if (ga.winner_seeds < 1)
    {
        errors << "  - " << prefix << ".winner_seeds must be >= 1 (got " << ga.winner_seeds
               << ")\n";
    }
    if (ga.tournament_k < 2)
    {
        errors << "  - " << prefix << ".tournament_k must be >= 2 (got " << ga.tournament_k
               << ")\n";
    }
}

void check_ga(const std::string& prefix, const Meeting01Config::Ga& ga, std::ostringstream& errors)
{
    check_ga_common(prefix, ga, errors);
    if (ga.min_layers < 1)
    {
        errors << "  - " << prefix << ".min_layers must be >= 1 (got " << ga.min_layers << ")\n";
    }
    if (ga.max_layers < ga.min_layers)
    {
        errors << "  - " << prefix << ".max_layers (" << ga.max_layers
               << ") must be >= min_layers (" << ga.min_layers << ")\n";
    }
    if (ga.min_width < 1)
    {
        errors << "  - " << prefix << ".min_width must be >= 1 (got " << ga.min_width << ")\n";
    }
    if (ga.max_width < ga.min_width)
    {
        errors << "  - " << prefix << ".max_width (" << ga.max_width << ") must be >= min_width ("
               << ga.min_width << ")\n";
    }
    if (ga.voltage_threshold_min <= 0.0f || ga.voltage_threshold_max < ga.voltage_threshold_min)
    {
        errors << "  - " << prefix
               << " voltage_threshold_min/max must satisfy 0 < min <= max (got min="
               << ga.voltage_threshold_min << ", max=" << ga.voltage_threshold_max << ")\n";
    }
    if (ga.alpha_min <= 0.0f || ga.alpha_max >= 1.0f || ga.alpha_max < ga.alpha_min)
    {
        errors << "  - " << prefix
               << " alpha_min/max must satisfy 0 < min <= max < 1 (got min=" << ga.alpha_min
               << ", max=" << ga.alpha_max << ")\n";
    }
}

void check_recurrent_ga(
    const std::string& prefix, const Meeting01Config::RecurrentGa& ga, std::ostringstream& errors)
{
    check_ga_common(prefix, ga, errors);
    if (ga.min_hidden < 1)
    {
        errors << "  - " << prefix << ".min_hidden must be >= 1 (got " << ga.min_hidden << ")\n";
    }
    if (ga.max_hidden < ga.min_hidden)
    {
        errors << "  - " << prefix << ".max_hidden (" << ga.max_hidden
               << ") must be >= min_hidden (" << ga.min_hidden << ")\n";
    }
    if (ga.min_layers < 1)
    {
        errors << "  - " << prefix << ".min_layers must be >= 1 (got " << ga.min_layers << ")\n";
    }
    if (ga.max_layers < ga.min_layers)
    {
        errors << "  - " << prefix << ".max_layers (" << ga.max_layers
               << ") must be >= min_layers (" << ga.min_layers << ")\n";
    }
}

void check_transformer_ga(
    const std::string& prefix, const Meeting01Config::TransformerGa& ga, std::ostringstream& errors)
{
    check_ga_common(prefix, ga, errors);
    if (ga.min_d_model < 1)
    {
        errors << "  - " << prefix << ".min_d_model must be >= 1 (got " << ga.min_d_model << ")\n";
    }
    if (ga.max_d_model < ga.min_d_model)
    {
        errors << "  - " << prefix << ".max_d_model (" << ga.max_d_model
               << ") must be >= min_d_model (" << ga.min_d_model << ")\n";
    }
    if (ga.head_choices.empty())
    {
        errors << "  - " << prefix << ".head_choices is empty (need >= 1 legal head count)\n";
    }
    for (int h : ga.head_choices)
    {
        if (h < 1)
        {
            errors << "  - " << prefix << ".head_choices contains a non-positive value (" << h
                   << ")\n";
        }
    }
    if (ga.min_layers < 1)
    {
        errors << "  - " << prefix << ".min_layers must be >= 1 (got " << ga.min_layers << ")\n";
    }
    if (ga.max_layers < ga.min_layers)
    {
        errors << "  - " << prefix << ".max_layers (" << ga.max_layers
               << ") must be >= min_layers (" << ga.min_layers << ")\n";
    }
    if (ga.min_d_ff < 1)
    {
        errors << "  - " << prefix << ".min_d_ff must be >= 1 (got " << ga.min_d_ff << ")\n";
    }
    if (ga.max_d_ff < ga.min_d_ff)
    {
        errors << "  - " << prefix << ".max_d_ff (" << ga.max_d_ff << ") must be >= min_d_ff ("
               << ga.min_d_ff << ")\n";
    }
}

void check_evaluation(const Meeting01Config::Evaluation& evaluation, std::ostringstream& errors)
{
    if (evaluation.datasets.empty())
    {
        errors << "  - evaluation.datasets is empty\n";
    }

    if (evaluation.encodings.empty())
    {
        errors << "  - evaluation.encodings is empty\n";
    }

    const std::vector<std::string> valid_baselines = {"lstm-ae", "gru-ae", "transformer-ae"};
    for (const auto& b : evaluation.baselines)
    {
        if (std::find(valid_baselines.begin(), valid_baselines.end(), b) == valid_baselines.end())
        {
            errors << "  - evaluation.baselines contains unknown family: '" << b << "'\n";
        }
    }
    if (evaluation.baselines.empty())
    {
        errors << "  - evaluation.baselines is empty (need at least one trained baseline family)\n";
    }

    const std::vector<std::string> valid_encodings = {"direct", "poisson", "latency"};
    for (const auto& enc : evaluation.encodings)
    {
        if (std::find(valid_encodings.begin(), valid_encodings.end(), enc) == valid_encodings.end())
        {
            errors << "  - evaluation.encodings contains unknown encoding: '" << enc << "'\n";
        }
    }

    // Every architecture-searched family (2026-09-22: LSTM-AE/GRU-AE/Transformer-AE
    // search their own shape now, not just the SNN) is checked only when that family
    // actually runs — evaluation.baselines lists the non-spiking families,
    // evaluation.snn_architectures being non-empty is what turns the SNN arm on.
    const bool has_lstm =
        std::find(evaluation.baselines.begin(), evaluation.baselines.end(), "lstm-ae") !=
        evaluation.baselines.end();
    const bool has_gru =
        std::find(evaluation.baselines.begin(), evaluation.baselines.end(), "gru-ae") !=
        evaluation.baselines.end();
    const bool has_transformer =
        std::find(evaluation.baselines.begin(), evaluation.baselines.end(), "transformer-ae") !=
        evaluation.baselines.end();

    if (has_lstm) check_recurrent_ga("evaluation.ga.lstm", evaluation.ga.lstm, errors);
    if (has_gru) check_recurrent_ga("evaluation.ga.gru", evaluation.ga.gru, errors);
    if (has_transformer)
        check_transformer_ga("evaluation.ga.transformer", evaluation.ga.transformer, errors);

    // The SNN knobs are only meaningful once an SNN architecture is asked
    // for; an empty list means this run has no SNN arm and the thresholds
    // below are legitimately unset.
    if (evaluation.snn_architectures.empty())
    {
        return;
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

    check_ga("evaluation.ga.snn", evaluation.ga.snn, errors);
}

} // namespace

void Meeting01Config::validate() const
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
    //
    // `view()` rather than `tellp()`: tellp() answers pos_type(-1) when the
    // stream is in a failed state, and -1 != 0, so that path would throw
    // with an EMPTY message -- a validation failure that names no field.
    // view() also reads the buffer without copying it into a std::string.
    if (!errors.view().empty())
    {
        throw std::invalid_argument("Meeting01Config validation failed:\n" + errors.str());
    }
}

} // namespace meeting01
