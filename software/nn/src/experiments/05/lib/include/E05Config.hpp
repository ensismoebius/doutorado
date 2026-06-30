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
        int max_samples = 0;           // 0 = unlimited (for debug: set to small number)
    };

    struct HandcraftedConfig
    {
        std::string transform = "dtwpt"; // "dtwpt" | "lfcc" | "mfcc"
        std::string scale = "lfcc";     // "bark" | "mel" | "lfcc"
        std::vector<std::string> descriptors = {"energy", "zcr", "entropy", "teager"};
        int dtwpt_level = 4;
    };

    struct AutoencoderConfig
    {
        std::string model = "lstm-ae"; // "lstm-ae" | "snn-ae"
        std::vector<std::string> encoder_layer_spec;
        std::vector<std::string> decoder_layer_spec;
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
        std::string type = "rnn";     // "rnn" | "dsnn"
        std::vector<std::string> layer_spec;
        std::string text_mode = "dependent"; // "dependent" | "independent"
    };

    struct Training
    {
        int epochs = 50;
        float learning_rate = 1e-3f;
        int samples_per_batch = 32;
        int early_stop_patience = 10;
        int k_folds = 5;
        bool nested_cv = true;

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
