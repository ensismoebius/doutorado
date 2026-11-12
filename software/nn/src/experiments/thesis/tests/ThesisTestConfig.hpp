#pragma once

// Configs built in code for the thesis tests. ThesisConfig has no defaults (Required<T>), so
// a test declares every key its run reads — these helpers declare them once, with the values
// the tests used to inherit from the old struct defaults, and each test overrides what it
// is about. Each helper declares exactly the keys of one kind of run, so a config that
// passes through validate() stays valid.

#include <cstdint>
#include <string>
#include <vector>

#include "../lib/include/ThesisConfig.hpp"

namespace thesis::test
{

/// Phase-00-shaped run (no classifier), every always-read key declared; strategy unset.
inline ThesisConfig phase00(const std::string& modality)
{
    ThesisConfig c;
    c.experiment.run_tag = "test";
    c.experiment.seed = 42u;
    c.experiment.repeats = 1;
    c.dataset.root = "/nonexistent";
    c.dataset.results_dir = "results/thesis";
    c.dataset.modality = modality;
    c.dataset.max_samples = 0;
    if (modality == "fused") c.dataset.fusion_mode = "late";
    if (modality != "voice") c.dataset.eeg_condition = "pronounced";
    c.paraconsistent.enabled = true;
    c.paraconsistent.speaker_subsets = false;
    c.classifier.enabled = false;
    return c;
}

/// The training keys both trainings read (adam: no momentum key).
inline void set_training(ThesisConfig& c, int epochs, float learning_rate, int samples_per_batch)
{
    c.training.epochs = epochs;
    c.training.learning_rate = learning_rate;
    c.training.optimizer_type = "adam";
    c.training.gradient_clip_norm = 0.0f;
    c.training.samples_per_batch = samples_per_batch;
}

/// Autoencoder strategy. snn-ae gets encoding/threshold/firing-rate keys (time_steps only
/// for a temporal code); ann-ae gets none of them.
inline void set_autoencoder(ThesisConfig& c,
    const std::string& model,
    std::vector<std::string> encoder,
    std::vector<std::string> decoder,
    const std::string& ae_loss_type)
{
    c.feature_extraction.strategy = "autoencoder";
    auto& ae = c.feature_extraction.autoencoder;
    ae.model = model;
    ae.encoder_layer_spec = std::move(encoder);
    if (model != "lstm-ae") ae.decoder_layer_spec = std::move(decoder);
    ae.ae_loss_type = ae_loss_type;
}

/// The snn-ae keys. time_steps is declared only for a temporal code; the band only when
/// lambda > 0.
inline void set_snn(ThesisConfig& c,
    const std::string& encoding,
    int time_steps,
    float voltage_threshold,
    float firing_rate_reg_lambda,
    float firing_rate_min,
    float firing_rate_max)
{
    auto& ae = c.feature_extraction.autoencoder;
    ae.encoding = encoding;
    if (encoding != "direct") ae.time_steps = time_steps;
    ae.voltage_threshold = voltage_threshold;
    ae.firing_rate_reg_lambda = firing_rate_reg_lambda;
    if (firing_rate_reg_lambda > 0.0f)
    {
        ae.firing_rate_min = firing_rate_min;
        ae.firing_rate_max = firing_rate_max;
    }
}

/// The handcrafted keys, at the values the old struct defaults had.
inline void set_handcrafted(ThesisConfig& c)
{
    c.feature_extraction.strategy = "handcrafted";
    auto& hc = c.feature_extraction.handcrafted;
    hc.transform = "dtwpt";
    hc.scale = "lfcc";
    hc.descriptors = std::vector<std::string>{"energy", "zcr", "entropy", "teager"};
    hc.dtwpt_level = 4;
    hc.wavelet = "daub4";
    hc.cepstral = false;
}

/// Phase-01-shaped classifier (flat CV), with every classifier training key.
inline void set_classifier(ThesisConfig& c, const std::string& type)
{
    c.classifier.enabled = true;
    c.classifier.type = type;
    c.classifier.text_mode = "dependent";
    c.training.k_folds = 3;
    c.training.nested_cv = false;
    c.training.standardize_features = true;
    if (c.dataset.modality.get() != "eeg") c.degradation.enabled = false;
    if (type == "cosine") return; // no network: no layer_spec, no training keys
    c.classifier.layer_spec =
        std::vector<std::string>{"linear:16:relu", "residual:2", "linear:N_speakers:identity"};
    c.training.weight_decay = 0.0f;
    if (type == "dsnn")
    {
        c.training.firing_rate_reg_lambda = 0.0f;
        c.training.batch_normalization = "none";
    }
}

} // namespace thesis::test
