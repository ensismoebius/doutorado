#include "ThesisConfig.hpp"

#include <algorithm>
#include <concepts>
#include <fstream>
#include <optional>
#include <stdexcept>

#include "optimizers/OptimizerFactory.hpp"

namespace thesis
{

auto ThesisConfig::Training::effective_learning_rate() const -> float
{
    // Explicit profile value wins; otherwise fall back to the chosen optimizer's own
    // reference default, so a profile that names an optimizer without naming an lr still
    // trains at a rate that makes sense FOR THAT optimizer (they differ by ~10x).
    return learning_rate.value_or(nn::optimizers::reference_learning_rate(optimizer_type));
}

namespace
{

// One checker per config section. They were one 241-line `validate()` with a
// cyclomatic complexity of 47 -- a single function nobody could hold in their
// head, in which a missing check was invisible. Each function below fails
// exactly as the original did, in the original order, with the original
// message: callers (and the 1937 profile-audit tests) cannot tell the
// difference. File-local on purpose, so the public header stays untouched.

void validate_experiment(const ThesisConfig::Experiment& experiment)
{
    if (experiment.run_tag.empty())
        throw std::invalid_argument("ThesisConfig: experiment.run_tag is required");
    if (experiment.seed == 0u)
        throw std::invalid_argument("ThesisConfig: experiment.seed must be non-zero");
    if (experiment.repeats <= 0)
        throw std::invalid_argument("ThesisConfig: experiment.repeats must be > 0");
}

void validate_dataset(const ThesisConfig::Dataset& dataset)
{
    if (dataset.root.empty()) throw std::invalid_argument("ThesisConfig: dataset.root is required");

    auto valid_modality =
        dataset.modality == "voice" || dataset.modality == "eeg" || dataset.modality == "fused";
    if (!valid_modality)
        throw std::invalid_argument("ThesisConfig: dataset.modality must be voice/eeg/fused");

    auto valid_fusion_mode = dataset.fusion_mode == "early" || dataset.fusion_mode == "late";
    if (!valid_fusion_mode)
        throw std::invalid_argument("ThesisConfig: dataset.fusion_mode must be early/late");
}

void validate_handcrafted(
    const ThesisConfig::FeatureExtraction& feature_extraction, const ThesisConfig::Dataset& dataset)
{
    if (feature_extraction.handcrafted.transform != "dtwpt")
        throw std::invalid_argument("ThesisConfig: handcrafted.transform must be dtwpt");

    const auto valid_scale = feature_extraction.handcrafted.scale == "bark" ||
                             feature_extraction.handcrafted.scale == "mel" ||
                             feature_extraction.handcrafted.scale == "lfcc";
    if (!valid_scale)
        throw std::invalid_argument("ThesisConfig: handcrafted.scale must be bark/mel/lfcc");

    // Bark and Mel are cochlear scales: they model the frequency resolution of human
    // HEARING. There is no physiological basis for applying them to EEG, which is not
    // sound. They are rejected for modality=eeg on that principle
    // (.wiki/Guides/Engineering-Fixes-Log.md D6).
    //
    // Empirically they were also inert here, which is what exposed the problem: for EEG
    // all three scales produced bit-identical d_truth in 46/46 wavelet x category groups
    // (the only apparent exceptions were daub32 at ~1e-6, i.e. float noise from that
    // profile's individual re-run). The mechanism is group_by_scale()'s normalization by
    // the signal's own Nyquist: Bark spans ~24 Barks over the audible range and n_bands
    // is 24, so for voice the factor is ~0.97 (a no-op — the bin IS the Bark number),
    // but for EEG's 512 Hz Nyquist it is ~4.96, stretching the curve 5x. That stretch
    // makes the mapping injective, so each sub-band lands in its own bin and the
    // grouping degenerates to exactly lfcc's one-group-per-sub-band. "Bark" for EEG was
    // therefore never Bark — just a linearly rescaled pseudo-scale identical to linear.
    //
    // Note this makes the grid deliberately asymmetric: voice sweeps all three scales
    // (where they genuinely differ), EEG uses lfcc only. Phase 00 handcrafted is thus
    // 138 voice + 46 eeg, not 138 x 2.
    //
    // modality=fused is intentionally NOT covered: its voice half legitimately uses
    // bark/mel. In late fusion the EEG half degenerates to lfcc as above (harmless but
    // mislabeled); in early fusion the concatenated signal runs at the voice rate.
    if (dataset.modality == "eeg" && feature_extraction.handcrafted.scale != "lfcc")
        throw std::invalid_argument(
            "ThesisConfig: handcrafted.scale must be lfcc for modality=eeg — bark/mel are "
            "cochlear (hearing) scales with no physiological basis for EEG, and are "
            "provably degenerate to lfcc there (see .wiki/Guides/Engineering-Fixes-Log.md D6)");

    // Mother wavelets with coefficient traits in include/wavelet/Types.hpp.
    static const std::vector<std::string> valid_wavelets = {"haar",
        "daub4",
        "daub6",
        "daub8",
        "daub10",
        "daub12",
        "daub14",
        "daub16",
        "daub18",
        "daub20",
        "daub22",
        "daub24",
        "daub26",
        "daub28",
        "daub30",
        "daub32",
        "daub34",
        "daub36",
        "daub38",
        "daub40",
        "daub42",
        "daub44",
        "daub46"};
    if (std::find(valid_wavelets.begin(),
            valid_wavelets.end(),
            feature_extraction.handcrafted.wavelet) == valid_wavelets.end())
        throw std::invalid_argument(
            "ThesisConfig: handcrafted.wavelet must be haar or daubN (even N in [4,46])");

    if (feature_extraction.handcrafted.descriptors.empty())
        throw std::invalid_argument("ThesisConfig: handcrafted.descriptors must not be empty");

    if (feature_extraction.handcrafted.dtwpt_level < 1)
        throw std::invalid_argument("ThesisConfig: handcrafted.dtwpt_level must be >= 1");
}

void validate_autoencoder(const ThesisConfig::FeatureExtraction& feature_extraction)
{
    const auto& ae_model = feature_extraction.autoencoder.model;
    if (ae_model != "snn-ae" && ae_model != "ann-ae" && ae_model != "lstm-ae")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.model must be snn-ae, ann-ae, or lstm-ae");
    if (feature_extraction.autoencoder.encoder_layer_spec.empty())
        throw std::invalid_argument("ThesisConfig: autoencoder.encoder_layer_spec is required");
    if (feature_extraction.autoencoder.decoder_layer_spec.empty())
        throw std::invalid_argument("ThesisConfig: autoencoder.decoder_layer_spec is required");
    const auto& ae_enc = feature_extraction.autoencoder.encoding;
    if (ae_enc != "poisson" && ae_enc != "latency" && ae_enc != "direct")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.encoding must be poisson, latency, or direct");

    // Reconstruction losses the AE training path can instantiate. CrossEntropy is
    // classification and is deliberately absent.
    const auto& ae_loss = feature_extraction.autoencoder.ae_loss_type;
    const bool known_loss =
        ae_loss == "mse" || ae_loss == "mae" || ae_loss == "spikecount" || ae_loss == "spiketime";
    if (!known_loss)
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.ae_loss_type must be mse, mae, spikecount, or "
            "spiketime");

    const bool spike_loss = (ae_loss == "spikecount" || ae_loss == "spiketime");

    // Spike losses need spikes: only the spiking model produces them, and only a
    // spiking encoding puts information in them.
    if (spike_loss && ae_model != "snn-ae")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.ae_loss_type=" + ae_loss +
            " requires model=snn-ae (ann-ae/lstm-ae emit continuous values, not spikes)");
    if (spike_loss && ae_enc == "direct")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.ae_loss_type=" + ae_loss +
            " is invalid for encoding=direct — direct is analog (no spikes); use mse or mae");

    // Encoding<->loss invariant (.wiki/Concepts/Spike-Encoding.md): the wrong spike
    // loss does not merely underperform, it destroys the gradient signal.
    //   SpikeTimeLoss sees only the FIRST spike -> blind to rate/count information.
    //   SpikeCountLoss treats absent spikes as count 0 -> wrong gradient for late spikes.
    if (ae_loss == "spiketime" && ae_enc != "latency")
        throw std::invalid_argument(
            "ThesisConfig: ae_loss_type=spiketime requires encoding=latency "
            "(first-spike timing carries the information); got encoding=" +
            ae_enc);
    if (ae_loss == "spikecount" && ae_enc != "poisson")
        throw std::invalid_argument(
            "ThesisConfig: ae_loss_type=spikecount requires encoding=poisson "
            "(rate/count carries the information); got encoding=" +
            ae_enc);
    if (feature_extraction.autoencoder.time_steps < 1)
        throw std::invalid_argument("ThesisConfig: autoencoder.time_steps must be >= 1");
    if (feature_extraction.autoencoder.firing_rate_reg_lambda < 0.0f)
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.firing_rate_reg_lambda must be >= 0");
    if (feature_extraction.autoencoder.firing_rate_min < 0.0f ||
        feature_extraction.autoencoder.firing_rate_max > 1.0f ||
        feature_extraction.autoencoder.firing_rate_min >
            feature_extraction.autoencoder.firing_rate_max)
        throw std::invalid_argument(
            "ThesisConfig: require 0 <= autoencoder.firing_rate_min <= "
            "autoencoder.firing_rate_max <= 1");
}

void validate_classifier(const ThesisConfig::Classifier& classifier)
{
    if (classifier.type != "rnn" && classifier.type != "dsnn")
        throw std::invalid_argument("ThesisConfig: classifier.type must be rnn or dsnn");

    auto valid_text_mode =
        classifier.text_mode == "dependent" || classifier.text_mode == "independent";
    if (!valid_text_mode)
        throw std::invalid_argument(
            "ThesisConfig: classifier.text_mode must be dependent/independent");

    // layer_spec is only meaningful when the classifier actually runs (Phase 01).
    // Phase 00 profiles (classifier.enabled=false) stop after paraconsistent
    // ranking and may omit it.
    if (classifier.enabled)
    {
        if (classifier.layer_spec.size() < 3)
            throw std::invalid_argument(
                "ThesisConfig: classifier.layer_spec must have at least 3 items: "
                "linear:H:act, residual:D, linear:N_speakers:identity");

        const auto starts_with = [](const std::string& s, const std::string& p)
        { return s.rfind(p, 0) == 0; };

        if (!starts_with(classifier.layer_spec.front(), "linear:"))
            throw std::invalid_argument(
                "ThesisConfig: first classifier.layer_spec item must start with linear:");

        const bool has_residual = std::any_of(classifier.layer_spec.begin(),
            classifier.layer_spec.end(),
            [&](const std::string& s) { return starts_with(s, "residual:"); });
        if (!has_residual)
            throw std::invalid_argument(
                "ThesisConfig: classifier.layer_spec must include residual:D item");

        if (!starts_with(classifier.layer_spec.back(), "linear:N_speakers:"))
            throw std::invalid_argument(
                "ThesisConfig: last classifier.layer_spec item must start with linear:N_speakers:");
    }
}

void validate_training(const ThesisConfig::Training& training)
{
    if (training.epochs <= 0)
        throw std::invalid_argument("ThesisConfig: training.epochs must be > 0");
    if (training.learning_rate.has_value() && *training.learning_rate <= 0.0f)
        throw std::invalid_argument("ThesisConfig: training.learning_rate must be > 0");
    if (training.samples_per_batch <= 0)
        throw std::invalid_argument("ThesisConfig: training.samples_per_batch must be > 0");
    if (training.early_stop_patience < 0)
        throw std::invalid_argument("ThesisConfig: training.early_stop_patience must be >= 0");
    if (training.k_folds < 2)
        throw std::invalid_argument("ThesisConfig: training.k_folds must be >= 2");
    if (training.optimizer_type != "adam" && training.optimizer_type != "sgd" &&
        training.optimizer_type != "lion" && training.optimizer_type != "schedule-free-adamw")
        throw std::invalid_argument(
            "ThesisConfig: training.optimizer_type must be one of "
            "adam, sgd, lion, schedule-free-adamw");
    if (training.optimizer_momentum < 0.0f || training.optimizer_momentum >= 1.0f)
        throw std::invalid_argument("ThesisConfig: require 0 <= training.optimizer_momentum < 1");
    if (training.gradient_clip_norm < 0.0f)
        throw std::invalid_argument(
            "ThesisConfig: training.gradient_clip_norm must be >= 0 "
            "(0 = off, the default)");
    if (training.weight_decay < 0.0f)
        throw std::invalid_argument("ThesisConfig: training.weight_decay must be >= 0");
    if (training.firing_rate_reg_lambda < 0.0f)
        throw std::invalid_argument("ThesisConfig: training.firing_rate_reg_lambda must be >= 0");
    if (training.firing_rate_min < 0.0f || training.firing_rate_max > 1.0f ||
        training.firing_rate_min > training.firing_rate_max)
        throw std::invalid_argument(
            "ThesisConfig: require 0 <= firing_rate_min <= firing_rate_max <= 1");
    if (training.batch_normalization != "none" &&
        training.batch_normalization != "threshold-dependent")
        throw std::invalid_argument(
            "ThesisConfig: training.batch_normalization must be none or threshold-dependent");
    if (training.tdbn_alpha <= 0.0f)
        throw std::invalid_argument("ThesisConfig: training.tdbn_alpha must be > 0");
}

} // namespace

void ThesisConfig::validate() const
{
    validate_experiment(experiment);
    validate_dataset(dataset);

    const bool valid_strategy = feature_extraction.strategy == "handcrafted" ||
                                feature_extraction.strategy == "autoencoder";
    if (!valid_strategy)
        throw std::invalid_argument(
            "ThesisConfig: feature_extraction.strategy must be handcrafted or autoencoder");

    if (feature_extraction.strategy == "handcrafted")
        validate_handcrafted(feature_extraction, dataset);
    else
        validate_autoencoder(feature_extraction);

    validate_classifier(classifier);
    validate_training(training);
}

namespace
{

// Every parser below is the same sentence repeated: "if the key is there,
// read it into this member". Written out per key it was 57 near-identical
// lines whose only variable content -- key and destination -- was buried in
// the boilerplate, and whose eight functions the duplication detector saw
// as one function copied eight times. Said once, the parsers become what
// they are: a table mapping JSON keys to members.
//
// An absent key leaves the member at the struct's own initializer, which is
// what the hand-written version did too. That is a documented default, not
// a fallback: nothing here recovers from an error by guessing (CLAUDE.md).
template <typename T>
concept OptionalMember = std::same_as<T, std::optional<typename T::value_type>>;

template <typename T>
void assign_if_present(const nlohmann::json& j, const char* key, T& out)
{
    if (!j.contains(key)) return;
    // An `std::optional` member holds "unset" as a distinct state, so its
    // value is read as the UNDERLYING type and wrapped. Reading it as an
    // optional would make a JSON null indistinguishable from an absent key,
    // and this function's whole contract is that those differ.
    if constexpr (OptionalMember<T>)
        out = j.at(key).template get<typename T::value_type>();
    else
        out = j.at(key).template get<T>();
}

// One parser per config section, mirroring the validators above: parse the
// section, then check it. They were one 138-line `from_json` whose
// cyclomatic complexity was 58 -- one branch per optional key, all in a
// single function, where a key assigned to the wrong member was invisible.
//
// Absent keys still leave the member at its declared default (the struct's
// own initializer), exactly as before: that is a documented default, not a
// fallback (CLAUDE.md's no-fallback rule bans *recovering from an error* by
// guessing, which nothing here does).

void parse_experiment(const nlohmann::json& e, ThesisConfig& cfg)
{
    assign_if_present(e, "run_tag", cfg.experiment.run_tag);
    assign_if_present(e, "seed", cfg.experiment.seed);
    assign_if_present(e, "repeats", cfg.experiment.repeats);
    assign_if_present(e, "seed_deterministic", cfg.experiment.seed_deterministic);
}

void parse_numerics(const nlohmann::json& n, ThesisConfig& cfg)
{
    assign_if_present(n, "exact_activations", cfg.numerics.exact_activations);
}

void parse_dataset(const nlohmann::json& d, ThesisConfig& cfg)
{
    assign_if_present(d, "root", cfg.dataset.root);
    assign_if_present(d, "results_dir", cfg.dataset.results_dir);
    assign_if_present(d, "modality", cfg.dataset.modality);
    assign_if_present(d, "fusion_mode", cfg.dataset.fusion_mode);
    assign_if_present(d, "max_samples", cfg.dataset.max_samples);
}

void parse_handcrafted(const nlohmann::json& hc, ThesisConfig& cfg)
{
    assign_if_present(hc, "transform", cfg.feature_extraction.handcrafted.transform);
    assign_if_present(hc, "scale", cfg.feature_extraction.handcrafted.scale);
    assign_if_present(hc, "descriptors", cfg.feature_extraction.handcrafted.descriptors);
    assign_if_present(hc, "dtwpt_level", cfg.feature_extraction.handcrafted.dtwpt_level);
    assign_if_present(hc, "wavelet", cfg.feature_extraction.handcrafted.wavelet);
    assign_if_present(hc, "cepstral", cfg.feature_extraction.handcrafted.cepstral);
}

void parse_autoencoder(const nlohmann::json& ae, ThesisConfig& cfg)
{
    assign_if_present(ae, "model", cfg.feature_extraction.autoencoder.model);
    assign_if_present(
        ae, "encoder_layer_spec", cfg.feature_extraction.autoencoder.encoder_layer_spec);
    assign_if_present(
        ae, "decoder_layer_spec", cfg.feature_extraction.autoencoder.decoder_layer_spec);
    assign_if_present(ae, "encoding", cfg.feature_extraction.autoencoder.encoding);
    assign_if_present(ae, "ae_loss_type", cfg.feature_extraction.autoencoder.ae_loss_type);
    assign_if_present(ae, "time_steps", cfg.feature_extraction.autoencoder.time_steps);
    assign_if_present(
        ae, "voltage_threshold", cfg.feature_extraction.autoencoder.voltage_threshold);
    assign_if_present(
        ae, "firing_rate_reg_lambda", cfg.feature_extraction.autoencoder.firing_rate_reg_lambda);
    assign_if_present(ae, "firing_rate_min", cfg.feature_extraction.autoencoder.firing_rate_min);
    assign_if_present(ae, "firing_rate_max", cfg.feature_extraction.autoencoder.firing_rate_max);
}

void parse_feature_extraction(const nlohmann::json& fe, ThesisConfig& cfg)
{
    assign_if_present(fe, "strategy", cfg.feature_extraction.strategy);

    if (fe.contains("handcrafted")) parse_handcrafted(fe["handcrafted"], cfg);
    if (fe.contains("autoencoder")) parse_autoencoder(fe["autoencoder"], cfg);
}

void parse_paraconsistent(const nlohmann::json& p, ThesisConfig& cfg)
{
    assign_if_present(p, "enabled", cfg.paraconsistent.enabled);
}

void parse_classifier(const nlohmann::json& c, ThesisConfig& cfg)
{
    assign_if_present(c, "type", cfg.classifier.type);
    assign_if_present(c, "layer_spec", cfg.classifier.layer_spec);
    assign_if_present(c, "text_mode", cfg.classifier.text_mode);
    assign_if_present(c, "enabled", cfg.classifier.enabled);
}

void parse_training(const nlohmann::json& t, ThesisConfig& cfg)
{
    assign_if_present(t, "epochs", cfg.training.epochs);
    assign_if_present(t, "learning_rate", cfg.training.learning_rate);
    assign_if_present(t, "samples_per_batch", cfg.training.samples_per_batch);
    assign_if_present(t, "early_stop_patience", cfg.training.early_stop_patience);
    assign_if_present(t, "k_folds", cfg.training.k_folds);
    assign_if_present(t, "nested_cv", cfg.training.nested_cv);
    assign_if_present(t, "standardize_features", cfg.training.standardize_features);
    assign_if_present(t, "optimizer_type", cfg.training.optimizer_type);
    assign_if_present(t, "optimizer_momentum", cfg.training.optimizer_momentum);
    assign_if_present(t, "gradient_clip_norm", cfg.training.gradient_clip_norm);
    assign_if_present(t, "weight_decay", cfg.training.weight_decay);
    assign_if_present(t, "firing_rate_reg_lambda", cfg.training.firing_rate_reg_lambda);
    assign_if_present(t, "firing_rate_min", cfg.training.firing_rate_min);
    assign_if_present(t, "firing_rate_max", cfg.training.firing_rate_max);
    assign_if_present(t, "batch_normalization", cfg.training.batch_normalization);
    assign_if_present(t, "tdbn_alpha", cfg.training.tdbn_alpha);
}

} // namespace

ThesisConfig ThesisConfig::from_json(const nlohmann::json& j)
{
    ThesisConfig cfg;

    if (j.contains("experiment")) parse_experiment(j["experiment"], cfg);
    if (j.contains("numerics")) parse_numerics(j["numerics"], cfg);
    if (j.contains("dataset")) parse_dataset(j["dataset"], cfg);
    if (j.contains("feature_extraction")) parse_feature_extraction(j["feature_extraction"], cfg);
    if (j.contains("paraconsistent")) parse_paraconsistent(j["paraconsistent"], cfg);
    if (j.contains("classifier")) parse_classifier(j["classifier"], cfg);
    if (j.contains("training")) parse_training(j["training"], cfg);

    return cfg;
}

ThesisConfig ThesisConfig::from_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("ThesisConfig: cannot open " + path);
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

} // namespace thesis
