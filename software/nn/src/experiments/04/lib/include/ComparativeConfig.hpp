#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace comparative_autoencoder_experiment
{

struct ComparativeConfig
{
    struct Experiment
    {
        std::string run_tag;            // REQUIRED
        std::uint32_t seed = 0u;        // REQUIRED (validated non-zero)
        int repeats = 0;                // REQUIRED (validated > 0)
        bool seed_deterministic = false; // optional: false = seeds 42,43,44,...
        bool check_determinism = false;  // optional
    };

    struct Dataset
    {
        std::string dataset_root;            // REQUIRED
        std::string results_dir = "results"; // optional
        int window_size = 0;                 // REQUIRED (validated > 0)
        int max_loaded_train_samples = 0;    // REQUIRED (validated > 0)
        int max_validation_samples = 0;      // REQUIRED (validated > 0)
        std::string latex_data_dir = "";     // optional
        bool save_models = false;            // optional
    };

    struct Training
    {
        int samples_per_batch = 0;                // REQUIRED (validated > 0)
        int batches_per_epoch = 0;                // optional (0 = use all samples)
        int epochs = 0;                           // REQUIRED (validated > 0)
        int early_stop_patience = -1;             // REQUIRED (validated >= 0)
        float learning_rate = 0.0f;               // REQUIRED (validated > 0)
        float learning_rate_biophysical = 0.0f;   // optional (0 = use 0.1 × lr)
        float beta1 = 0.9f;                       // optional (Adam default)
        float beta2 = 0.999f;                     // optional (Adam default)
        float epsilon = 1e-8f;                    // optional (Adam default)
        float max_reconstruct_mean_deviation = 0.25f; // optional
    };

    struct Model
    {
        int latent_dim = 0;        // optional (0 = derive from encoder_layer_spec)
        int lstm_hidden_size = 0;  // optional (0 = derive from encoder_layer_spec)
        int branch_hidden_size = 0;
        int fusion_hidden_size = 0;
        std::string loss_type = "mse"; // optional (from model.loss_function)
        std::vector<std::string> encoder_layer_spec; // REQUIRED
        std::vector<std::string> decoder_layer_spec; // REQUIRED
        std::vector<std::string> branch_encoder_layer_spec;
        std::vector<std::string> branch_decoder_layer_spec;
        std::vector<std::string> fusion_encoder_layer_spec;
        std::vector<std::string> fusion_decoder_layer_spec;
    };

    struct Evaluation
    {
        std::vector<std::string> datasets;          // REQUIRED
        std::vector<std::string> encodings;         // REQUIRED
        std::vector<std::string> snn_architectures; // REQUIRED (use [] for LSTM-only)
        std::vector<float> v_th_values;             // REQUIRED if snn_architectures non-empty
        std::vector<float> alpha_values;            // REQUIRED if snn_architectures non-empty
    };

    Experiment experiment;
    Dataset dataset;
    Training training;
    Model model;
    Evaluation evaluation;

    void validate() const;

    static ComparativeConfig from_flat_json(const nlohmann::json& j)
    {
        ComparativeConfig cfg;

        auto get = [&](const std::string& key, auto& field)
        {
            if (j.contains(key)) field = j[key].get<std::decay_t<decltype(field)>>();
        };

        // Experiment
        get("run_tag", cfg.experiment.run_tag);
        get("seed", cfg.experiment.seed);
        get("repeats", cfg.experiment.repeats);
        get("seed_deterministic", cfg.experiment.seed_deterministic);
        get("check_determinism", cfg.experiment.check_determinism);

        // Dataset
        get("dataset_root", cfg.dataset.dataset_root);
        get("results_dir", cfg.dataset.results_dir);
        get("window_size", cfg.dataset.window_size);
        get("max_loaded_train_samples", cfg.dataset.max_loaded_train_samples);
        get("max_validation_samples", cfg.dataset.max_validation_samples);
        get("latex_data_dir", cfg.dataset.latex_data_dir);
        get("save_models", cfg.dataset.save_models);

        // Training
        get("samples_per_batch", cfg.training.samples_per_batch);
        get("batches_per_epoch", cfg.training.batches_per_epoch);
        get("epochs", cfg.training.epochs);
        get("early_stop_patience", cfg.training.early_stop_patience);
        get("learning_rate", cfg.training.learning_rate);
        get("learning_rate_biophysical", cfg.training.learning_rate_biophysical);
        get("beta1", cfg.training.beta1);
        get("adam_beta1", cfg.training.beta1);
        get("beta2", cfg.training.beta2);
        get("adam_beta2", cfg.training.beta2);
        get("epsilon", cfg.training.epsilon);
        get("adam_epsilon", cfg.training.epsilon);
        get("max_reconstruct_mean_deviation", cfg.training.max_reconstruct_mean_deviation);

        // Model
        get("latent_dim", cfg.model.latent_dim);
        get("lstm_hidden_size", cfg.model.lstm_hidden_size);
        get("loss_function", cfg.model.loss_type);
        get("branch_hidden_size", cfg.model.branch_hidden_size);
        get("fusion_hidden_size", cfg.model.fusion_hidden_size);
        get("encoder_layer_spec", cfg.model.encoder_layer_spec);
        get("decoder_layer_spec", cfg.model.decoder_layer_spec);
        get("branch_encoder_layer_spec", cfg.model.branch_encoder_layer_spec);
        get("branch_decoder_layer_spec", cfg.model.branch_decoder_layer_spec);
        get("fusion_encoder_layer_spec", cfg.model.fusion_encoder_layer_spec);
        get("fusion_decoder_layer_spec", cfg.model.fusion_decoder_layer_spec);

        // Evaluation
        get("datasets", cfg.evaluation.datasets);
        get("encodings", cfg.evaluation.encodings);
        get("snn_architectures", cfg.evaluation.snn_architectures);
        get("v_th_values", cfg.evaluation.v_th_values);
        get("alpha_values", cfg.evaluation.alpha_values);

        return cfg;
    }

    static ComparativeConfig from_nested_json(const nlohmann::json& j)
    {
        ComparativeConfig cfg;

        for (const auto* section : {"experiment", "dataset", "training", "model", "evaluation"})
        {
            if (!j.contains(section))
                throw std::invalid_argument(
                    std::string("ComparativeConfig: required section missing: ") + section);
        }

        const auto& exp = j["experiment"];
        const auto& dat = j["dataset"];
        const auto& trn = j["training"];
        const auto& mdl = j["model"];
        const auto& evl = j["evaluation"];

        // Optional: silently use default if key absent
        auto get = [](const nlohmann::json& sec, const std::string& key, auto& field)
        {
            if (sec.contains(key)) field = sec[key].get<std::decay_t<decltype(field)>>();
        };
        // Required: throw if key absent
        auto require = [](const nlohmann::json& sec,
                           const std::string& section_name,
                           const std::string& key,
                           auto& field)
        {
            if (!sec.contains(key))
                throw std::invalid_argument(
                    "ComparativeConfig: required field missing: " + section_name + "." + key);
            field = sec[key].get<std::decay_t<decltype(field)>>();
        };

        // --- experiment (all required except optional flags) ---
        require(exp, "experiment", "run_tag",  cfg.experiment.run_tag);
        require(exp, "experiment", "seed",     cfg.experiment.seed);
        require(exp, "experiment", "repeats",  cfg.experiment.repeats);
        get(exp, "seed_deterministic", cfg.experiment.seed_deterministic);
        get(exp, "check_determinism",  cfg.experiment.check_determinism);

        // --- dataset ---
        require(dat, "dataset", "dataset_root",             cfg.dataset.dataset_root);
        require(dat, "dataset", "window_size",              cfg.dataset.window_size);
        require(dat, "dataset", "max_loaded_train_samples", cfg.dataset.max_loaded_train_samples);
        require(dat, "dataset", "max_validation_samples",   cfg.dataset.max_validation_samples);
        get(dat, "results_dir",   cfg.dataset.results_dir);
        get(dat, "latex_data_dir", cfg.dataset.latex_data_dir);
        get(dat, "save_models",   cfg.dataset.save_models);

        // --- training ---
        require(trn, "training", "samples_per_batch",  cfg.training.samples_per_batch);
        require(trn, "training", "epochs",             cfg.training.epochs);
        require(trn, "training", "early_stop_patience", cfg.training.early_stop_patience);
        require(trn, "training", "learning_rate",      cfg.training.learning_rate);
        get(trn, "batches_per_epoch",            cfg.training.batches_per_epoch);
        get(trn, "learning_rate_biophysical",    cfg.training.learning_rate_biophysical);
        get(trn, "beta1",     cfg.training.beta1);
        get(trn, "adam_beta1", cfg.training.beta1);
        get(trn, "beta2",     cfg.training.beta2);
        get(trn, "adam_beta2", cfg.training.beta2);
        get(trn, "epsilon",   cfg.training.epsilon);
        get(trn, "adam_epsilon", cfg.training.epsilon);
        get(trn, "max_reconstruct_mean_deviation", cfg.training.max_reconstruct_mean_deviation);

        // --- model ---
        require(mdl, "model", "encoder_layer_spec", cfg.model.encoder_layer_spec);
        require(mdl, "model", "decoder_layer_spec", cfg.model.decoder_layer_spec);
        get(mdl, "latent_dim",      cfg.model.latent_dim);
        get(mdl, "lstm_hidden_size", cfg.model.lstm_hidden_size);
        get(mdl, "loss_function",   cfg.model.loss_type);
        get(mdl, "branch_hidden_size", cfg.model.branch_hidden_size);
        get(mdl, "fusion_hidden_size", cfg.model.fusion_hidden_size);
        get(mdl, "branch_encoder_layer_spec", cfg.model.branch_encoder_layer_spec);
        get(mdl, "branch_decoder_layer_spec", cfg.model.branch_decoder_layer_spec);
        get(mdl, "fusion_encoder_layer_spec", cfg.model.fusion_encoder_layer_spec);
        get(mdl, "fusion_decoder_layer_spec", cfg.model.fusion_decoder_layer_spec);

        // --- evaluation ---
        require(evl, "evaluation", "datasets",          cfg.evaluation.datasets);
        require(evl, "evaluation", "encodings",         cfg.evaluation.encodings);
        require(evl, "evaluation", "snn_architectures", cfg.evaluation.snn_architectures);
        get(evl, "v_th_values",  cfg.evaluation.v_th_values);
        get(evl, "alpha_values", cfg.evaluation.alpha_values);

        return cfg;
    }
};

} // namespace comparative_autoencoder_experiment
