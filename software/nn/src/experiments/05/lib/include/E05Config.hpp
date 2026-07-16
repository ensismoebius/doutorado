#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace e05
{

struct E05Config
{
    struct Experiment
    {
        std::string run_tag;
        std::uint32_t seed = 42u;
        int repeats = 1;
        bool seed_deterministic = false;
    };

    struct Dataset
    {
        std::string root;
        std::string results_dir = "results";
        std::string modality = "fused"; // "voice" | "eeg" | "fused"
        int max_samples = 0;            // 0 = unlimited (for debug: set to small number)

        /// Only meaningful when modality == "fused". Ignored otherwise.
        ///   "early" → concatenate the raw voice+EEG signals into one signal
        ///             before feature extraction (single extractor run).
        ///   "late"  → run feature extraction independently per signal, then
        ///             concatenate the resulting feature vectors.
        std::string fusion_mode = "late"; // "early" | "late"
    };

    struct HandcraftedConfig
    {
        std::string transform = "dtwpt"; // "dtwpt" | "lfcc" | "mfcc"
        std::string scale = "lfcc";      // "bark" | "mel" | "lfcc"
        std::vector<std::string> descriptors = {"energy", "zcr", "entropy", "teager"};
        int dtwpt_level = 4;

        /// Mother wavelet for the DTWPT filter. One of the tags with coefficient
        /// traits in include/wavelet/Types.hpp: "haar" or "daubN" for even
        /// N in [4, 46]. Swept in Phase 00 (feature-vector selection).
        std::string wavelet = "daub4";

        /// Category 2 (cepstral) features. When false, per-band energies are used
        /// directly (Category 1: linear/mel/bark-band energies). When true, a
        /// log + DCT-II is applied across the band energies, yielding cepstral
        /// coefficients (LFCC/MFCC/BFCC for scale = lfcc/mel/bark).
        bool cepstral = false;
    };

    struct AutoencoderConfig
    {
        std::string model = "lstm-ae"; // "lstm-ae" | "snn-ae" | "ann-ae"
        std::vector<std::string> encoder_layer_spec;
        std::vector<std::string> decoder_layer_spec;

        /// Temporal spike code for snn-ae (ignored by ann-ae / lstm-ae):
        /// "poisson" (rate), "latency" (first-spike time), or "direct" (analog,
        /// no spikes). Each sample is expanded into `time_steps` frames and the
        /// per-sample feature is the mean latent spike-rate over them.
        std::string encoding = "poisson";
        int time_steps = 16;

        /// Encoder LIF firing threshold (snn-ae). Spike frames are low-amplitude,
        /// so the default is well below the LIF default (1.0) to ensure the
        /// encoder neurons actually fire; otherwise the latent collapses to zero.
        float voltage_threshold = 0.2f;

        /// Firing-rate regularization for the snn-ae encoder. 0 = disabled.
        /// Pushes each encoder Lif layer's mean firing rate into
        /// [firing_rate_min, firing_rate_max], preventing dead-latent collapse
        /// (rate->0) as documented in fixme.md D1. Mirrors the dsnn classifier's
        /// Training::firing_rate_reg_lambda mechanism. Ignored by ann-ae/lstm-ae.
        float firing_rate_reg_lambda = 0.0f;
        float firing_rate_min = 0.05f;
        float firing_rate_max = 0.80f;
    };

    struct FeatureExtraction
    {
        std::string strategy = "handcrafted"; // "handcrafted" | "autoencoder"
        HandcraftedConfig handcrafted;
        AutoencoderConfig autoencoder;
    };

    struct Paraconsistent
    {
        bool enabled = true;
    };

    struct Classifier
    {
        std::string type = "rnn"; // "rnn" | "dsnn"
        std::vector<std::string> layer_spec;
        std::string text_mode = "dependent"; // "dependent" | "independent"

        /// Phase gate. When false, the pipeline stops after paraconsistent
        /// ranking (Phase 00: feature-vector construction / EPC selection) and
        /// never trains a classifier; layer_spec is then not required. When true
        /// (Phase 01: authentication) the classifier runs as usual.
        bool enabled = true;
    };

    struct Training
    {
        int epochs = 50;
        float learning_rate = 1e-3f;
        int samples_per_batch = 32;
        int early_stop_patience = 10;
        int k_folds = 5;
        bool nested_cv = true;

        /// Per-feature z-score standardization of the classifier input. Statistics
        /// (mean, std per dimension) are fit on the TRAINING rows of each fold only
        /// and applied to train and test, preventing test-set leakage. Default on.
        bool standardize_features = true;

        // ── Regularization ───────────────────────────────────────────────────
        /// Decoupled L2 weight decay (AdamW). 0 = disabled. Applies to both the
        /// rnn (ResNet) and dsnn classifiers; only 2-D weight matrices are
        /// decayed (biases and SNN R/C/V_th scalars are excluded by the optimizer).
        float weight_decay = 0.0f;

        /// Firing-rate regularization weight for the dsnn classifier. 0 = disabled.
        /// Pushes each spiking layer's mean firing rate into [firing_rate_min,
        /// firing_rate_max], preventing dead (rate→0) and bursting (rate→1)
        /// neurons. Ignored by the rnn classifier (no spiking layers).
        float firing_rate_reg_lambda = 0.0f;
        float firing_rate_min = 0.05f; ///< Lower band edge (dead-neuron guard).
        float firing_rate_max = 0.80f; ///< Upper band edge (burst guard).

        /// Batch normalization for the dsnn classifier:
        ///   "none"                → no normalization
        ///   "threshold-dependent" → tdBN (Zheng et al., AAAI 2021): a tdBN layer
        ///        is inserted after each Linear and before each LIF, normalizing the
        ///        pre-spike current over batch+time and rescaling to N(0,(α·V_th)²).
        /// Ignored by the rnn classifier (no spiking layers).
        std::string batch_normalization = "none";
        float tdbn_alpha = 1.0f; ///< α for tdBN (target std = α·V_th; paper default 1).
    };

    Experiment experiment;
    Dataset dataset;
    FeatureExtraction feature_extraction;
    Paraconsistent paraconsistent;
    Classifier classifier;
    Training training;

    void validate() const;

    static E05Config from_json(const nlohmann::json& j);
    static E05Config from_file(const std::string& path);
};

} // namespace e05
