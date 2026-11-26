#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "Required.hpp"
#include "nlohmann/json.hpp"

namespace thesis
{

/// The thesis profile (Experiment05). NO FIELD HAS A DEFAULT.
///
/// Every member is a `Required<T>` (Required.hpp): it holds what the profile declared, or
/// nothing, and reading nothing throws. The parser enforces the other half of the contract:
///
///   a key the run READS     must be declared   (absent  -> refused, naming the key)
///   a key the run IGNORES   must be absent     (present -> refused, naming why it is ignored)
///   a key nobody knows      must be absent     (typo    -> refused)
///
/// Which keys a run reads depends on other keys (momentum only for sgd, fusion_mode only for
/// fused, time_steps only for a temporal SNN code, ...). Those rules are written ONCE, as a
/// table, in ThesisConfig.cpp (`for_each_field`); the comments below repeat them for the
/// reader. Why refuse an ignored key: a profile that says `"encoding": "poisson"` for an
/// ANN-AE reads as a poisson run, while the ANN-AE always runs "direct". A declared value
/// that changes nothing is a silent fallback with extra steps.
struct ThesisConfig
{
    struct Experiment
    {
        Required<std::string> run_tag{"experiment.run_tag"};
        Required<std::uint32_t> seed{"experiment.seed"}; ///< non-zero
        Required<int> repeats{"experiment.repeats"};     ///< > 0

        /// Only when repeats > 1: true = every repeat reuses `seed`; false = repeat r uses
        /// seed + r. With one repeat there is nothing to decide, so the key is refused.
        Required<bool> seed_deterministic{"experiment.seed_deterministic"};
    };

    struct Dataset
    {
        /// Which base `root` holds, and so how it is read (ThesisDataset.cpp):
        ///   "coretto2017" -> 10.1117/12.2255697: 15 speakers, imagined + pronounced EEG
        ///                    (6 ch, 1024 Hz), voice 44.1 kHz; the trial-table SQLite.
        ///   "biomex2021"  -> BIOMEX-DB: 51 speakers, pronounced digits only, EEG 14 ch at
        ///                    128 Hz, voice 16 kHz; the paired SQLite written by
        ///                    scripts/data/biomex_to_sqlite.py. It has no imagined speech.
        /// Declared, never inferred from the file: the loader checks that the file is that base.
        Required<std::string> source{"dataset.source"};
        Required<std::string> root{"dataset.root"};
        Required<std::string> results_dir{"dataset.results_dir"};
        Required<std::string> modality{"dataset.modality"}; ///< "voice" | "eeg" | "fused"

        /// Round-robin cap across speakers; 0 = every sample (an explicit choice, not "unset").
        Required<int> max_samples{"dataset.max_samples"};

        /// Only when modality == "fused":
        ///   "early" -> concatenate the raw voice+EEG signals, one extractor run;
        ///   "late"  -> one extractor run per signal, feature vectors concatenated.
        Required<std::string> fusion_mode{"dataset.fusion_mode"};

        /// Only when the run reads EEG (modality "eeg" or "fused"): which EEG trial goes
        /// with each voice trial (EegPairing.hpp).
        ///   "pronounced" -> the EEG of the same trial, recorded while the voice was produced;
        ///   "imagined"   -> an Imagined trial of the same (subject, stimulus), drawn without
        ///                   replacement.
        Required<std::string> eeg_condition{"dataset.eeg_condition"};
        /// Only when eeg_condition == "imagined": the seed of that draw (non-zero). Fixed per
        /// profile, not per repetition: the repetitions vary the training, not the data.
        Required<std::uint32_t> eeg_pairing_seed{"dataset.eeg_pairing_seed"};
    };

    /// Simulated dysphonia (Phase 01, runs that read the voice). The voice of the TEST
    /// speakers is replaced by a dysphonic version of itself; the training speakers stay
    /// clean: the system is trained on healthy voices and meets a dysphonic one. Which
    /// dysphonia is `profile`, and each profile reads its own keys (a key another profile
    /// reads is refused, like any key the run does not read):
    ///
    ///   whisper_mix          level                       VoiceDegradation.hpp: breathy/aphonic
    ///   resynthesis_control  (none)                      VoiceDysphonia.hpp: the F0 profiles
    ///   rough                jitter_pct, shimmer_db
    ///   tremor               f0_depth_pct, rate_hz
    ///   voice_breaks         break_kind, breaks_per_second, break_ms
    ///   diplophonia          period_alt_pct, amp_alt_db
    struct Degradation
    {
        Required<bool> enabled{"degradation.enabled"};
        /// enabled only: "whisper_mix" | "resynthesis_control" | "rough" | "tremor" |
        /// "voice_breaks" | "diplophonia".
        Required<std::string> profile{"degradation.profile"};
        /// whisper_mix only: 0 < level <= 1 (the fraction of the voice's POWER that is breath
        /// instead of voice; 1 = fully whispered, i.e. aphonic). A power fraction, not a
        /// clinical severity unit.
        Required<float> level{"degradation.level"};
        /// enabled only: "probes" -> enrolled with the clean voice, verified with the degraded
        ///                           one (the voice was lost after enrolment);
        ///               "test"   -> enrolment and probes both degraded (dysphonic throughout).
        Required<std::string> scope{"degradation.scope"};
        /// enabled only: seed of the random stream (non-zero): the whisper excitation, or the
        /// per-cycle draws of the F0 profiles.
        Required<std::uint32_t> seed{"degradation.seed"};

        /// rough only: local jitter / shimmer of the output, percent / dB (MEASURED units;
        /// the realised value is calibrated in the tests, see VoiceDysphonia.hpp).
        Required<float> jitter_pct{"degradation.jitter_pct"};
        Required<float> shimmer_db{"degradation.shimmer_db"};
        /// tremor only: peak F0 deviation (percent of F0) and sway rate (Hz).
        Required<float> f0_depth_pct{"degradation.f0_depth_pct"};
        Required<float> rate_hz{"degradation.rate_hz"};
        /// voice_breaks only: "pitch_jump" | "aphonic", breaks per second of voiced speech,
        /// and the length of one break in milliseconds.
        Required<std::string> break_kind{"degradation.break_kind"};
        Required<float> breaks_per_second{"degradation.breaks_per_second"};
        Required<float> break_ms{"degradation.break_ms"};
        /// diplophonia only: alternate cycles longer/shorter by period_alt_pct percent and
        /// louder/softer by amp_alt_db dB.
        Required<float> period_alt_pct{"degradation.period_alt_pct"};
        Required<float> amp_alt_db{"degradation.amp_alt_db"};
    };

    /// Only when feature_extraction.strategy == "handcrafted".
    struct HandcraftedConfig
    {
        Required<std::string> transform{"feature_extraction.handcrafted.transform"}; ///< "dtwpt"
        Required<std::string> scale{
            "feature_extraction.handcrafted.scale"}; ///< "bark" | "mel" | "lfcc"
        Required<std::vector<std::string>> descriptors{
            "feature_extraction.handcrafted.descriptors"};
        Required<int> dtwpt_level{"feature_extraction.handcrafted.dtwpt_level"};

        /// Mother wavelet: "haar" or "daubN", even N in [4, 46] (traits in wavelet/Types.hpp).
        Required<std::string> wavelet{"feature_extraction.handcrafted.wavelet"};

        /// false = Category 1 (per-band energies); true = Category 2 (log + DCT-II across the
        /// band energies: LFCC/MFCC/BFCC for scale = lfcc/mel/bark).
        Required<bool> cepstral{"feature_extraction.handcrafted.cepstral"};
    };

    /// Only when feature_extraction.strategy == "autoencoder".
    struct AutoencoderConfig
    {
        /// "snn-ae" | "ann-ae" | "lstm-ae". lstm-ae is kept for the record (meeting01 paper)
        /// and is used by no thesis profile.
        Required<std::string> model{"feature_extraction.autoencoder.model"};
        Required<std::vector<std::string>> encoder_layer_spec{
            "feature_extraction.autoencoder.encoder_layer_spec"};

        /// snn-ae and ann-ae. The lstm-ae decoder mirrors its encoder, so it has none.
        Required<std::vector<std::string>> decoder_layer_spec{
            "feature_extraction.autoencoder.decoder_layer_spec"};

        /// snn-ae only: "poisson" (rate), "latency" (first-spike time) or "direct" (analog).
        /// The ANN-AE is analog by construction and never reads it.
        Required<std::string> encoding{"feature_extraction.autoencoder.encoding"};

        /// snn-ae with a temporal code (poisson, latency) only: frames per sample. The
        /// direct code presents one analog frame, so T does not exist there.
        Required<int> time_steps{"feature_extraction.autoencoder.time_steps"};

        /// snn-ae only: encoder LIF firing threshold.
        Required<float> voltage_threshold{"feature_extraction.autoencoder.voltage_threshold"};

        /// Reconstruction loss that TRAINS the autoencoder:
        ///   "mse" | "mae"   continuous reconstruction (lstm-ae: mse only — its path is MSE)
        ///   "spikecount"    rate-coded SNN (elementwise MSE + firing-rate band)
        ///   "spiketime"     latency-coded SNN (first-spike-time MSE)
        /// The encoding<->loss pairing is an INVARIANT (.wiki/Concepts/Spike-Encoding.md):
        /// direct -> mse|mae, poisson -> spikecount (or mse/mae as an explicit baseline),
        /// latency -> spiketime (likewise). validate() enforces it. `spiketime` needs a
        /// TIME-MAJOR (T*B, F) batch; the AE trainer builds it (ThesisFeatureExtraction).
        Required<std::string> ae_loss_type{"feature_extraction.autoencoder.ae_loss_type"};

        /// snn-ae only: firing-rate regularization of the encoder (0 = off), against the
        /// dead-latent collapse of .wiki/Guides/Engineering-Fixes-Log.md D1.
        Required<float> firing_rate_reg_lambda{
            "feature_extraction.autoencoder.firing_rate_reg_lambda"};
        /// snn-ae with firing_rate_reg_lambda > 0 only: the target rate band. With the
        /// regularizer off nothing reads them.
        Required<float> firing_rate_min{"feature_extraction.autoencoder.firing_rate_min"};
        Required<float> firing_rate_max{"feature_extraction.autoencoder.firing_rate_max"};
    };

    /// Only when feature_extraction.strategy == "pca".
    ///
    /// Principal component analysis on the same input the autoencoders see (the signal's
    /// 256-bin log power spectrum, min-max normalized per sample), fitted on every sample
    /// like the autoencoders are trained on every sample: the unsupervised, transductive
    /// protocol of Phase 00. The feature vector is the projection on the first
    /// `n_components` principal axes. It is the linear, training-free reference extractor
    /// of the thesis (ThesisPca.hpp).
    struct PcaConfig
    {
        /// 1 <= n_components <= min(256, n_samples - 1).
        Required<int> n_components{"feature_extraction.pca.n_components"};
    };

    struct FeatureExtraction
    {
        Required<std::string> strategy{
            "feature_extraction.strategy"}; ///< "handcrafted" | "autoencoder" | "pca"
        HandcraftedConfig handcrafted;
        AutoencoderConfig autoencoder;
        PcaConfig pca;
    };

    struct Paraconsistent
    {
        Required<bool> enabled{"paraconsistent.enabled"};
        /// enabled only: also score every feature set on speaker subsets — leaving each
        /// speaker out once, and keeping only the TRAINING speakers of each Phase 01 fold —
        /// to show whether the selection depends on the speakers Phase 01 tests on.
        Required<bool> speaker_subsets{"paraconsistent.speaker_subsets"};
        /// speaker_subsets only: the Phase 01 folds to reproduce (k, base seed, repeats;
        /// repetition r uses seed + r, as Phase 01 does).
        Required<int> fold_k{"paraconsistent.fold_k"};
        Required<std::uint32_t> fold_seed{"paraconsistent.fold_seed"};
        Required<int> fold_repeats{"paraconsistent.fold_repeats"};
    };

    struct Classifier
    {
        /// Phase gate. false = Phase 00 (stop after the paraconsistent ranking; the other
        /// classifier keys are then refused). true = Phase 01 (authentication).
        Required<bool> enabled{"classifier.enabled"};

        /// enabled only:
        /// "dsnn"   -> the spiking residual network (the thesis's classifier);
        /// "ann"    -> the DSNN's graph with ReLU for LIF, same initial weights (non-spiking
        /// control); "cosine" -> no network: the standardized features ARE the embedding (the floor
        /// any
        ///             trained classifier must beat).
        Required<std::string> type{"classifier.type"};
        /// dsnn / ann only.
        Required<std::vector<std::string>> layer_spec{"classifier.layer_spec"};
        Required<std::string> text_mode{"classifier.text_mode"}; ///< "dependent" | "independent"
    };

    /// One training block serves two trainings: the autoencoder's (strategy autoencoder)
    /// and the classifier's (classifier.enabled). A key is required when a training that
    /// runs reads it; a run with neither (handcrafted or PCA, Phase 00) has no training.
    struct Training
    {
        /// Both trainings:
        Required<int> epochs{"training.epochs"};
        Required<float> learning_rate{"training.learning_rate"};
        Required<std::string> optimizer_type{
            "training.optimizer_type"}; ///< "adam" | "sgd" | "lion" | "schedule-free-adamw"
        /// sgd only (Polyak momentum); every other optimizer ignores it.
        Required<float> optimizer_momentum{"training.optimizer_momentum"};
        /// Gradient-norm clipping applied by the Trainer; 0 = off (what PyTorch does unless
        /// the caller clips). The only clipping knob: MSELoss/MAELoss no longer clip.
        Required<float> gradient_clip_norm{"training.gradient_clip_norm"};
        Required<int> samples_per_batch{"training.samples_per_batch"};

        /// Classifier only:
        Required<int> k_folds{"training.k_folds"}; ///< >= 2, GroupKFold by speaker
        Required<bool> nested_cv{"training.nested_cv"};
        /// nested_cv only: the inner split is what early stopping watches.
        Required<int> early_stop_patience{"training.early_stop_patience"};
        /// Per-feature z-score fitted on each fold's TRAINING rows only.
        Required<bool> standardize_features{"training.standardize_features"};
        /// Decoupled L2 (AdamW-style); only 2-D weights are decayed. 0 = off.
        Required<float> weight_decay{"training.weight_decay"};

        /// dsnn classifier only: firing-rate regularization (0 = off) ...
        Required<float> firing_rate_reg_lambda{"training.firing_rate_reg_lambda"};
        /// ... and its band, only when that regularization is on.
        Required<float> firing_rate_min{"training.firing_rate_min"};
        Required<float> firing_rate_max{"training.firing_rate_max"};

        /// dsnn classifier only: "none" | "threshold-dependent" (tdBN, Zheng et al., AAAI
        /// 2021: normalizes the pre-spike current over batch+time to N(0, (alpha*V_th)^2)).
        Required<std::string> batch_normalization{"training.batch_normalization"};
        /// tdBN only: alpha (target std = alpha * V_th).
        Required<float> tdbn_alpha{"training.tdbn_alpha"};
    };

    /// One extractor the per-fold protocol may choose: a Phase 00 profile's extraction keys and,
    /// for an autoencoder, the training keys that train it (the run's own training block trains
    /// the classifier). Parsed and validated as a complete Phase 00 profile of the ranked signal
    /// (candidate_config), so every rule of a Phase 00 profile holds for it.
    struct ExtractorCandidate
    {
        FeatureExtraction feature_extraction;
        Training training; ///< declared only when the candidate trains (autoencoder)
    };

    /// The per-fold protocol (.wiki/Experiments/Thesis-Per-Fold-Protocol.md). Read only when the
    /// classifier runs. enabled = false: the extractor and the classifier's hyperparameters are
    /// the profile's own (feature_extraction, classifier.layer_spec, training.learning_rate,
    /// training.weight_decay). enabled = true: they are CHOSEN inside each outer fold, on that
    /// fold's training rows only, and the profile must not declare them (a declared value that
    /// the run replaces would read as applied).
    struct Selection
    {
        Required<bool> enabled{"selection.enabled"};
        /// enabled only: the signal the candidates are ranked on by D_pen ("voice" | "eeg"):
        /// the profile's modality, or "voice" for a fused profile (the fold's voice winner feeds
        /// both parts).
        Required<std::string> rank_modality{"selection.rank_modality"};
        /// enabled only, "selection.extractor_candidates": a JSON array of
        /// {"feature_extraction": {...}, "training": {...}} objects, in the order ties are
        /// broken (first wins). Not a table row: parsed and validated per candidate.
        std::vector<ExtractorCandidate> extractor_candidates;
        /// enabled with a dsnn/ann classifier only: the inner speaker folds of the
        /// hyperparameter search and its grid (every combination is a candidate).
        Required<int> inner_k{"selection.inner_k"};
        Required<std::vector<int>> hidden_widths{"selection.hidden_widths"};
        Required<std::vector<int>> residual_blocks{"selection.residual_blocks"}; ///< after input
        Required<std::vector<float>> learning_rates{"selection.learning_rates"};
        Required<std::vector<float>> weight_decays{"selection.weight_decays"};
    };

    /// A program that embeds this config in its own profile and owns part of it — the
    /// paraconsistent GA, which evolves the autoencoder architecture and has its own
    /// run_tag/results_dir. Its profile must not declare what it overrides or never reads,
    /// for the same reason a standalone profile must not declare an ignored key.
    struct Embedding
    {
        struct Key
        {
            std::string path;   ///< profile path, e.g. "feature_extraction.autoencoder.encoding"
            std::string reason; ///< why the program owns or ignores it (quoted in the refusal)
        };
        std::string program;                   ///< named in error messages
        std::vector<std::string> own_sections; ///< its top-level keys, parsed by the program
        /// Filled by the program before use (e.g. by the genome): refused in the profile;
        /// may be declared or not in the config the program holds.
        std::vector<Key> set_by_program;
        /// Never read by the program: refused in the profile, never declared.
        std::vector<Key> not_read;
    };

    Experiment experiment;
    Dataset dataset;
    FeatureExtraction feature_extraction;
    Paraconsistent paraconsistent;
    Classifier classifier;
    Degradation degradation;
    Training training;
    Selection selection;

    /// nullopt = the thesis binary itself; set = parsed for that embedding program.
    std::optional<Embedding> embedded_in;

    /// Every key the run reads is declared, every key it does not read is not; throws
    /// naming the first violation. Run by from_json and by validate().
    void check_declarations() const;

    /// check_declarations() plus every value check (ranges, enums, cross-field invariants).
    void validate() const;

    /// Parse a thesis profile. Refuses unknown keys, missing keys the run reads, and
    /// declared keys it ignores. Does not run the value checks — call validate().
    static ThesisConfig from_json(const nlohmann::json& j);
    /// The same, for a program that embeds this config (see Embedding).
    static ThesisConfig from_json(const nlohmann::json& j, const Embedding& embedding);
    static ThesisConfig from_file(const std::string& path);

    /// Every declared key, as profile JSON (what the run was configured with).
    [[nodiscard]] auto declared_json() const -> nlohmann::json;

    /// selection.enabled only: the Phase 00-shaped config that extracts with candidate `i` on
    /// `modality` ("voice" | "eeg" | "fused"; fused keeps this profile's fusion_mode): this
    /// run's experiment (seed included) and dataset, the candidate's feature_extraction and
    /// training, no classifier, no ranking output. Validated before it is returned.
    [[nodiscard]] auto candidate_config(std::size_t i, const std::string& modality) const
        -> ThesisConfig;
};

} // namespace thesis
