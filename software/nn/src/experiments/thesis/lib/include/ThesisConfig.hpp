#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace thesis
{

struct ThesisConfig
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
        std::string results_dir = "results/thesis"; // Thesis = thesis experiment
        std::string modality = "fused";             // "voice" | "eeg" | "fused"
        int max_samples = 0; // 0 = unlimited (for debug: set to small number)

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
        /// (rate->0) as documented in .wiki/Guides/Engineering-Fixes-Log.md D1. Mirrors the dsnn
        /// classifier's Training::firing_rate_reg_lambda mechanism. Ignored by ann-ae/lstm-ae.
        float firing_rate_reg_lambda = 0.0f;
        float firing_rate_min = 0.05f;
        float firing_rate_max = 0.80f;
    };

    /// Numerical fidelity of activation functions.
    ///
    /// The LSTM uses rational approximations from FastActivations.hpp by default in this
    /// codebase's history — rat_sig(x)=0.5+x/(2(1+|x|)) and rat_tanh(x)=x/(1+|x|), i.e.
    /// softsign-based gates chosen for speed. They are NOT close to the real thing:
    /// |tanh - rat_tanh| reaches 0.306 on [-4,4] (at x=2, tanh=0.964 vs approx 0.667), which
    /// makes our LSTM a *softsign-gated* LSTM that cannot match torch.nn.LSTM.
    ///
    /// Since PyTorch/snnTorch is this project's reference for correctness, `true` (exact
    /// sigmoid/tanh, matching torch) is the DEFAULT. Set false only to trade fidelity for
    /// speed, and expect divergence from the reference.
    struct Numerics
    {
        bool exact_activations = true;
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

        /// Learning rate. **Optional in the profile JSON**: when omitted it resolves to the
        /// chosen optimizer's own reference default via
        /// nn::optimizers::reference_learning_rate(optimizer_type) — 1e-3 for adam, 1e-4 for
        /// lion, 2.5e-3 for schedule-free-adamw, 1e-2 for sgd.
        ///
        /// Why optional: these defaults differ by up to an order of magnitude because each
        /// optimizer forms its update differently (Lion steps ±lr on every coordinate). A
        /// profile that switches optimizer_type but keeps another optimizer's lr silently
        /// measures the learning rate instead of the optimizer. Omitting the field makes
        /// "every profile gets its respective lr" true by construction; setting it
        /// explicitly (e.g. to sweep lr) still overrides, and either way the value actually
        /// used is recorded in the run summary.
        ///
        /// Use effective_learning_rate() rather than reading this directly.
        std::optional<float> learning_rate = std::nullopt;

        /// Optimizer used to train the classifier AND the autoencoder feature
        /// extractors. Forwarded to nn::training::TrainerConfig::optimizer_type,
        /// which builds it via OptimizerFactory. One of: "adam" (default),
        /// "sgd", "lion", "schedule-free-adamw".
        /// Default "adam" reproduces every result published before this field existed.
        std::string optimizer_type = "adam";

        /// Momentum for optimizer_type="sgd" (Polyak). Ignored by the others.
        float optimizer_momentum = 0.0f;

        /// Gradient-norm clipping, applied by Trainer to the model's parameter gradients
        /// after backward. **0 = OFF (default)**, which is what PyTorch does unless the
        /// caller explicitly clips, and what keeps our training numerically comparable to
        /// the reference.
        ///
        /// Historically MSELoss/MAELoss ALSO clipped their own gradient, unconditionally, at
        /// norm 1.0 — silently, non-configurably, and overriding this field. That is now
        /// off by default too (see MSELoss::max_gradient_norm); this is the single, honest,
        /// profile-visible clipping knob.
        float gradient_clip_norm = 0.0f;

        /// The lr this run will actually train with: the explicit `learning_rate` when the
        /// profile sets one, otherwise `optimizer_type`'s reference default. Single accessor
        /// so no caller can accidentally read an unresolved nullopt or hard-code 1e-3.
        [[nodiscard]] auto effective_learning_rate() const -> float;
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
    Numerics numerics;
    Dataset dataset;
    FeatureExtraction feature_extraction;
    Paraconsistent paraconsistent;
    Classifier classifier;
    Training training;

    void validate() const;

    static ThesisConfig from_json(const nlohmann::json& j);
    static ThesisConfig from_file(const std::string& path);
};

} // namespace thesis
