#include "../include/Meeting01Config.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "../include/Meeting01EegBand.hpp"

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
               << ") must divide dataset.window_size (" << dataset.window_size
               << "), so that a frame never straddles two simulation steps\n";
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

/// Every EEG dataset the profile runs must cut windows that hold more than its latent can carry
/// (Meeting01EegBand.hpp says why, with the Siena numbers). Only the datasets in
/// evaluation.datasets: a source listed but never run loads nothing.
void check_eeg_windows(const Meeting01Config& config, std::ostringstream& errors)
{
    for (const auto& name : config.evaluation.datasets)
    {
        if (!is_eeg_dataset(name)) continue;
        const auto src = config.dataset.resolve(name);
        if (src.sample_rate <= 0)
        {
            errors << "  - dataset.sources[" << name
                   << "].sample_rate must be set: it is the rate every EEG recording is "
                      "converted to, so it fixes how long a window lasts\n";
            continue;
        }
        const int latent = src.latent_dim > 0 ? src.latent_dim : config.model.latent_dim;
        if (latent <= 0 || src.window_size <= 0)
            continue; // derived latent / bad size: reported elsewhere
        const double dof = eeg_window_degrees_of_freedom(src.window_size, src.sample_rate);
        if (dof <= static_cast<double>(latent))
        {
            errors << "  - dataset.sources[" << name << "]: a " << src.window_size
                   << "-sample window at " << src.sample_rate << " Hz, band-limited to "
                   << kEegBandLowHz << "-" << kEegBandHighHz << " Hz, holds about " << dof
                   << " independent values, not more than latent_dim " << latent
                   << ", so the bottleneck can pass it whole (PCA reconstructs it exactly). "
                      "Remedy: a lower sample_rate, which makes the window last longer (160 Hz "
                      "gives 1.6 s and "
                   << eeg_window_degrees_of_freedom(src.window_size, 160.0)
                   << " values), or a smaller latent_dim\n";
        }
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

    // Not >= 1: a skill of 1 is a perfect reconstruction, so patience would never start.
    if (training.lr_warmup_steps < 0)
        errors << "  - training.lr_warmup_steps must be >= 0 (got " << training.lr_warmup_steps
               << "); 0 turns the warm-up off\n";
    if (!(training.early_stop_min_skill >= 0.0f && training.early_stop_min_skill < 1.0f))
    {
        errors << "  - training.early_stop_min_skill must be in [0, 1) (got "
               << training.early_stop_min_skill
               << "); it is the share of the mean predictor's monitor loss a run must remove "
                  "before patience counts\n";
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

    if (model.recurrent_time_steps < 1)
    {
        errors << "  - model.recurrent_time_steps must be >= 1 (got " << model.recurrent_time_steps
               << "); it is the number of steps the LSTM/GRU/Transformer baselines read per "
                  "window (1 = the window once, 16 = repeated like the SNN)\n";
    }

    const bool known_pool = model.snn_time_pool == "none" || model.snn_time_pool == "output" ||
                            model.snn_time_pool == "latent";
    if (!known_pool)
    {
        errors << "  - model.snn_time_pool must be \"none\", \"output\" or \"latent\" (got \""
               << model.snn_time_pool << "\")\n";
    }
    if (model.snn_decoder_neuron != "lif" && model.snn_decoder_neuron != "analog")
    {
        errors << "  - model.snn_decoder_neuron must be \"lif\" or \"analog\" (got \""
               << model.snn_decoder_neuron << "\")\n";
    }
    // After a latent average the decoder sees one row per window, and a LifBPTT there would
    // have no time axis to unroll (it throws on B rows that are not a multiple of T).
    if (model.snn_time_pool == "latent" && model.snn_decoder_neuron != "analog")
    {
        errors << "  - model.snn_time_pool \"latent\" runs the decoder once per window, so it "
                  "needs model.snn_decoder_neuron \"analog\" (got \""
               << model.snn_decoder_neuron << "\")\n";
    }

    if (!model.initializer.empty() && model.initializer != "kaiming" &&
        model.initializer != "xavier")
    {
        errors << "  - model.initializer must be \"kaiming\" or \"xavier\" (got \""
               << model.initializer << "\")\n";
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
    if (ga.voltage_threshold_min < 1e-3f || ga.voltage_threshold_max < ga.voltage_threshold_min)
    {
        errors << "  - " << prefix
               << " voltage_threshold_min/max must satisfy 0.001 <= min <= max (got min="
               << ga.voltage_threshold_min << ", max=" << ga.voltage_threshold_max << ")\n";
    }
    if (ga.alpha_min < 1e-3f || ga.alpha_max > 0.999f || ga.alpha_max < ga.alpha_min)
    {
        errors << "  - " << prefix
               << " alpha_min/max must satisfy 0.001 <= min <= max <= 0.999 (got min="
               << ga.alpha_min << ", max=" << ga.alpha_max << ")\n";
    }
    const bool fixed_t = ga.time_steps_min == 0 && ga.time_steps_max == 0;
    if (!fixed_t && (ga.time_steps_min < 2 || ga.time_steps_max < ga.time_steps_min))
    {
        errors << "  - " << prefix
               << " time_steps_min/max must both be 0 (T fixed at model.time_steps) or satisfy "
                  "2 <= min <= max (got min="
               << ga.time_steps_min << ", max=" << ga.time_steps_max
               << "). One step leaves alpha and the poisson/latency codes without a time axis.\n";
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

    const std::vector<std::string> valid_baselines = {
        "lstm-ae", "gru-ae", "transformer-ae", "ann-ae"};
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
    // "ann-ae" is not searched: it is the matched control of the SNN arm -- the SNN GA winner's
    // layer widths with a leaky-ReLU neuron instead of a LIF one -- so without an SNN arm there
    // is no architecture for it to copy.
    if (std::find(evaluation.baselines.begin(), evaluation.baselines.end(), "ann-ae") !=
            evaluation.baselines.end() &&
        evaluation.snn_architectures.empty())
    {
        errors << "  - evaluation.baselines has 'ann-ae' (the matched ANN control of the SNN "
                  "winner) but evaluation.snn_architectures is empty, so there is no SNN "
                  "winner to copy; add an SNN architecture or drop 'ann-ae'\n";
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
    check_eeg_windows(*this, errors);
    check_training(*this, errors);
    check_model(model, errors);
    check_evaluation(evaluation, errors);

    // ann-ae is the SNN-AE with one neuron changed. With model.initializer empty each builder
    // keeps its own init (kaiming in the SNN one, xavier in the ANN one), so the control would
    // differ in its starting weights too -- on eegmmidb that alone moved a twin's test MSE
    // (measured 2026-11-01, .wiki/Experiments/Meeting01.md).
    if (std::find(evaluation.baselines.begin(), evaluation.baselines.end(), "ann-ae") !=
            evaluation.baselines.end() &&
        model.initializer.empty())
    {
        errors << "  - evaluation.baselines has 'ann-ae' but model.initializer is empty, so it "
                  "would start from xavier weights and the SNN-AE from kaiming ones; set "
                  "model.initializer to \"kaiming\" or \"xavier\" so the two arms differ in "
                  "the neuron only\n";
    }

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
