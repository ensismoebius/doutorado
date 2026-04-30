#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace comparative_autoencoder_experiment
{

struct ComparativeConfig
{
    struct Experiment
    {
        std::string run_tag = "snn_lstm_compare";
        std::uint32_t seed = 1337u;
        int repeats = 3;
        bool seed_deterministic = true;
        bool check_determinism = false;
    };

    struct Dataset
    {
        std::string dataset_root = ".";
        std::string results_dir = "results";
        int window_size = 128;
        int max_loaded_train_samples = 512;
        int max_validation_samples = 128;
    };

    struct Training
    {
        int samples_per_batch = 1;
        int batches_per_epoch = 0;
        int epochs = 100;
        int early_stop_patience = 20;
        float learning_rate = 1e-3f;
        float max_reconstruct_mean_deviation = 0.25f;
    };

    struct Model
    {
        int branch_hidden_size = 0;
        int fusion_hidden_size = 0;
        std::vector<std::string> encoder_layer_spec;
        std::vector<std::string> decoder_layer_spec;
        std::vector<std::string> branch_encoder_layer_spec;
        std::vector<std::string> branch_decoder_layer_spec;
        std::vector<std::string> fusion_encoder_layer_spec;
        std::vector<std::string> fusion_decoder_layer_spec;
    };

    struct Evaluation
    {
        std::vector<std::string> datasets = {"fsdd"};
        std::vector<std::string> encodings = {"direct", "poisson", "latency"};
        std::vector<std::string> snn_architectures = {"dense", "conv1d", "recurrent"};
        std::vector<float> v_th_values = {0.5f, 1.0f, 1.5f};
        std::vector<float> alpha_values = {0.8f, 0.9f, 0.99f};
    };

    Experiment experiment;
    Dataset dataset;
    Training training;
    Model model;
    Evaluation evaluation;

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

        // Training
        get("samples_per_batch", cfg.training.samples_per_batch);
        get("batches_per_epoch", cfg.training.batches_per_epoch);
        get("epochs", cfg.training.epochs);
        get("early_stop_patience", cfg.training.early_stop_patience);
        get("learning_rate", cfg.training.learning_rate);
        get("max_reconstruct_mean_deviation", cfg.training.max_reconstruct_mean_deviation);

        // Model
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

        const auto& exp = j["experiment"];
        const auto& dat = j["dataset"];
        const auto& trn = j["training"];
        const auto& mdl = j["model"];
        const auto& evl = j["evaluation"];

        auto get_exp = [&](const std::string& key, auto& field)
        {
            if (exp.contains(key)) field = exp[key].get<std::decay_t<decltype(field)>>();
        };
        auto get_dat = [&](const std::string& key, auto& field)
        {
            if (dat.contains(key)) field = dat[key].get<std::decay_t<decltype(field)>>();
        };
        auto get_trn = [&](const std::string& key, auto& field)
        {
            if (trn.contains(key)) field = trn[key].get<std::decay_t<decltype(field)>>();
        };
        auto get_mdl = [&](const std::string& key, auto& field)
        {
            if (mdl.contains(key)) field = mdl[key].get<std::decay_t<decltype(field)>>();
        };
        auto get_evl = [&](const std::string& key, auto& field)
        {
            if (evl.contains(key)) field = evl[key].get<std::decay_t<decltype(field)>>();
        };

        get_exp("run_tag", cfg.experiment.run_tag);
        get_exp("seed", cfg.experiment.seed);
        get_exp("repeats", cfg.experiment.repeats);
        get_exp("seed_deterministic", cfg.experiment.seed_deterministic);
        get_exp("check_determinism", cfg.experiment.check_determinism);

        get_dat("dataset_root", cfg.dataset.dataset_root);
        get_dat("results_dir", cfg.dataset.results_dir);
        get_dat("window_size", cfg.dataset.window_size);
        get_dat("max_loaded_train_samples", cfg.dataset.max_loaded_train_samples);
        get_dat("max_validation_samples", cfg.dataset.max_validation_samples);

        get_trn("samples_per_batch", cfg.training.samples_per_batch);
        get_trn("batches_per_epoch", cfg.training.batches_per_epoch);
        get_trn("epochs", cfg.training.epochs);
        get_trn("early_stop_patience", cfg.training.early_stop_patience);
        get_trn("learning_rate", cfg.training.learning_rate);
        get_trn("max_reconstruct_mean_deviation", cfg.training.max_reconstruct_mean_deviation);

        get_mdl("branch_hidden_size", cfg.model.branch_hidden_size);
        get_mdl("fusion_hidden_size", cfg.model.fusion_hidden_size);
        get_mdl("encoder_layer_spec", cfg.model.encoder_layer_spec);
        get_mdl("decoder_layer_spec", cfg.model.decoder_layer_spec);
        get_mdl("branch_encoder_layer_spec", cfg.model.branch_encoder_layer_spec);
        get_mdl("branch_decoder_layer_spec", cfg.model.branch_decoder_layer_spec);
        get_mdl("fusion_encoder_layer_spec", cfg.model.fusion_encoder_layer_spec);
        get_mdl("fusion_decoder_layer_spec", cfg.model.fusion_decoder_layer_spec);

        get_evl("datasets", cfg.evaluation.datasets);
        get_evl("encodings", cfg.evaluation.encodings);
        get_evl("snn_architectures", cfg.evaluation.snn_architectures);
        get_evl("v_th_values", cfg.evaluation.v_th_values);
        get_evl("alpha_values", cfg.evaluation.alpha_values);

        return cfg;
    }
};

} // namespace comparative_autoencoder_experiment
