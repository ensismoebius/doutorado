#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace meeting01
{

/// `model.snn_time_steps` was renamed to `model.time_steps` on 2026-09-22, because the
/// `snn_` prefix was a lie: `run_baseline` reads the same field, so it also sets the
/// sequence length the LSTM / GRU / Transformer baselines are unrolled over.
///
/// A profile still carrying the old key must FAIL, not be quietly ignored. Ignoring it
/// would silently fall back to the struct default — which is exactly the class of bug
/// this whole audit existed to remove: a run that completes normally and reports a
/// plausible number produced under a temporal resolution nobody asked for.
inline void reject_renamed_time_steps(const nlohmann::json& section)
{
    if (!section.is_object() || !section.contains("snn_time_steps")) return;

    throw std::invalid_argument(
        "Meeting01Config: 'snn_time_steps' was renamed to 'time_steps' on 2026-09-22 "
        "(the field sets the unroll length for the LSTM/GRU/Transformer baselines too, "
        "not only the SNN's membrane depth). Remedy: rename the key to 'time_steps' in "
        "this profile. Do NOT delete it — dropping the key would silently fall back to "
        "the default of 16 steps.");
}

struct Meeting01Config
{
    struct Experiment
    {
        std::string run_tag;             // REQUIRED
        std::uint32_t seed = 0u;         // REQUIRED (validated non-zero)
        int repeats = 0;                 // REQUIRED (validated > 0)
        bool seed_deterministic = false; // optional: false = seeds 42,43,44,...
        bool check_determinism = false;  // optional
    };

    // Per-dataset override. When evaluation.datasets names something other than
    // the default "fsdd" root, a matching entry here supplies its root, window
    // size, fold count, native sample rate, and per-recording window cap. Any
    // field left 0 / empty inherits the singular Dataset value below.
    struct DatasetSource
    {
        std::string name;                  // REQUIRED ("fsdd" | "audiomnist" | "mitbih")
        std::string root;                  // REQUIRED
        int window_size = 0;               // 0 → inherit Dataset::window_size
        int cv_num_folds = 0;              // 0 → inherit Dataset::cv_num_folds
        int sample_rate = 0;               // native rate to resample from (0 → loader default)
        int max_windows_per_recording = 0; // 0 → unlimited (FSDD); >0 caps long recordings
        // Per-fold stratified window caps (0 → inherit Dataset value; still 0 → unlimited).
        // Applied after the LOSO split, round-robin across recordings, so every recording
        // and speaker stays represented. Bounds per-epoch training cost (batch_size 1).
        int loso_max_train_windows = 0;
        int loso_max_val_windows = 0;
        int loso_max_test_windows = 0;
    };

    struct Dataset
    {
        std::string dataset_root;                      // REQUIRED
        std::string results_dir = "results/meeting01"; // optional (Meeting01 = Meeting01 paper)
        int window_size = 0;                           // REQUIRED (validated > 0)
        int max_loaded_train_samples = 0;              // REQUIRED (validated > 0)
        int max_validation_samples = 0;                // REQUIRED (validated > 0)
        // Nested leave-one-group-out cross-validation (article pipeline).
        // cv_fold < 0  → legacy pooled split (physionet / ad-hoc runs).
        // cv_fold >= 0 → speaker/group-disjoint fold; must be < cv_num_folds,
        //               which must not exceed the distinct speaker count.
        int cv_fold = -1;                   // optional
        int cv_num_folds = 6;               // optional (FSDD speaker count)
        int max_windows_per_recording = 0;  // optional (0 = unlimited; default source)
        int loso_max_train_windows = 0;     // optional (0 = unlimited); stratified per-fold cap
        int loso_max_val_windows = 0;       // optional (0 = unlimited)
        int loso_max_test_windows = 0;      // optional (0 = unlimited)
        std::string latex_data_dir = "";    // optional
        bool save_models = false;           // optional
        std::vector<DatasetSource> sources; // optional per-dataset overrides

        // Resolve the effective source for `name`. "fsdd" (or any name with no
        // explicit entry) falls back to the singular Dataset fields.
        [[nodiscard]] auto resolve(const std::string& name) const -> DatasetSource
        {
            DatasetSource s;
            s.name = name;
            s.root = dataset_root;
            s.window_size = window_size;
            s.cv_num_folds = cv_num_folds;
            s.max_windows_per_recording = max_windows_per_recording;
            s.loso_max_train_windows = loso_max_train_windows;
            s.loso_max_val_windows = loso_max_val_windows;
            s.loso_max_test_windows = loso_max_test_windows;
            for (const auto& e : sources)
            {
                if (e.name != name) continue;
                if (!e.root.empty()) s.root = e.root;
                if (e.window_size > 0) s.window_size = e.window_size;
                if (e.cv_num_folds > 0) s.cv_num_folds = e.cv_num_folds;
                if (e.sample_rate > 0) s.sample_rate = e.sample_rate;
                if (e.max_windows_per_recording > 0)
                    s.max_windows_per_recording = e.max_windows_per_recording;
                if (e.loso_max_train_windows > 0)
                    s.loso_max_train_windows = e.loso_max_train_windows;
                if (e.loso_max_val_windows > 0) s.loso_max_val_windows = e.loso_max_val_windows;
                if (e.loso_max_test_windows > 0) s.loso_max_test_windows = e.loso_max_test_windows;
                break;
            }
            return s;
        }
    };

    struct Training
    {
        int samples_per_batch = 0;                    // REQUIRED (validated > 0)
        int batches_per_epoch = 0;                    // optional (0 = use all samples)
        int epochs = 0;                               // REQUIRED (validated > 0)
        int early_stop_patience = -1;                 // REQUIRED (validated >= 0)
        float learning_rate = 0.0f;                   // REQUIRED (validated > 0)
        float learning_rate_biophysical = 0.0f;       // optional (0 = use 0.1 × lr)
        float beta1 = 0.9f;                           // optional (Adam default)
        float beta2 = 0.999f;                         // optional (Adam default)
        float epsilon = 1e-8f;                        // optional (Adam default)
        float max_reconstruct_mean_deviation = 0.25f; // optional
    };

    struct Model
    {
        int latent_dim = 0;       // optional (0 = derive from encoder_layer_spec)
        int lstm_hidden_size = 0; // optional (0 = derive from encoder_layer_spec)
        // Samples per LSTM timestep. window_size/lstm_frame_size becomes the
        // sequence length. 1 = the old scalar-per-timestep behaviour, which makes
        // the recurrent term cost window_size times more than it needs to.
        // Must divide dataset.window_size.
        int lstm_frame_size = 8;
        // SNN simulation steps per window. The encoder turns one window into a
        // time-major (time_steps * B, window_size) tensor, so this is the number
        // of steps LifBPTT unrolls and over which a spike code can carry information.
        // Must be >= 2: at 1 there is no membrane history, which silently disables
        // both `alpha` and rate/latency coding (see .wiki/Experiments/Meeting01.md).
        int time_steps = 16;
        int branch_hidden_size = 0;
        int fusion_hidden_size = 0;
        // Bottlenecked Transformer-AE baseline dimensions (used only when
        // "transformer-ae" is in evaluation.baselines). Defaults sit near the
        // recurrent baselines' parameter count; the real count is reported.
        int transformer_d_model = 64;
        int transformer_heads = 4;
        int transformer_layers = 2;
        int transformer_d_ff = 128;
        std::string loss_type = "mse";               // optional (from model.loss_function)
        std::vector<std::string> encoder_layer_spec; // REQUIRED
        std::vector<std::string> decoder_layer_spec; // REQUIRED
        std::vector<std::string> branch_encoder_layer_spec;
        std::vector<std::string> branch_decoder_layer_spec;
        std::vector<std::string> fusion_encoder_layer_spec;
        std::vector<std::string> fusion_decoder_layer_spec;
    };

    // NSGA-II architecture search over the SNN-AE's own shape — the ONLY SNN
    // architecture search mechanism (no grid/sweep path exists). encoder depth/width,
    // encoding, architecture, v_th and alpha are ALL genes; evaluation.encodings and
    // evaluation.snn_architectures supply the legal choice pools for the
    // encoding/architecture genes (reusing their existing whitelist validation).
    // Runs whenever evaluation.snn_architectures is non-empty; an empty list means the
    // run is LSTM/GRU/Transformer-only (no SNN arm), the pre-existing escape hatch.
    struct Ga
    {
        int population_size = 10;
        int generations = 8;
        int min_layers = 1;
        int max_layers = 4;
        int min_width = 4;
        int max_width = 128;
        float voltage_threshold_min = 0.1f;
        float voltage_threshold_max = 2.0f;
        float alpha_min = 0.5f;
        float alpha_max = 0.99f;
        double crossover_prob = 0.9;
        double mutation_prob = 0.2;
        int tournament_k = 2;
        // Seeds used to re-score each final Pareto-front member before picking the
        // winner (winner's-curse mitigation). 1 = pick on the single search score.
        int winner_seeds = 3;
        unsigned int seed = 0; // 0 -> derive from experiment.seed + run_id
        int checkpoint_every_generations = 1;
    };

    struct Evaluation
    {
        std::vector<std::string> datasets;  // REQUIRED
        std::vector<std::string> encodings; // REQUIRED
        // Trained non-spiking baseline families to run per fold. Valid entries:
        // "lstm-ae", "gru-ae", "transformer-ae". Default keeps the legacy
        // LSTM-only behaviour. PCA / mean-frame references are added downstream
        // in Python, not here.
        std::vector<std::string> baselines = {"lstm-ae"};
        std::vector<std::string> snn_architectures; // REQUIRED (use [] for SNN-free runs)
        Ga ga; // GA search bounds; only consulted when snn_architectures is non-empty
    };

    Experiment experiment;
    Dataset dataset;
    Training training;
    Model model;
    Evaluation evaluation;

    void validate() const;

    static void parse_sources(const nlohmann::json& arr, std::vector<DatasetSource>& out)
    {
        for (const auto& e : arr)
        {
            DatasetSource s;
            s.name = e.at("name").get<std::string>();
            s.root = e.value("root", std::string{});
            s.window_size = e.value("window_size", 0);
            s.cv_num_folds = e.value("cv_num_folds", 0);
            s.sample_rate = e.value("sample_rate", 0);
            s.max_windows_per_recording = e.value("max_windows_per_recording", 0);
            s.loso_max_train_windows = e.value("loso_max_train_windows", 0);
            s.loso_max_val_windows = e.value("loso_max_val_windows", 0);
            s.loso_max_test_windows = e.value("loso_max_test_windows", 0);
            out.push_back(std::move(s));
        }
    }

    static Meeting01Config from_flat_json(const nlohmann::json& j)
    {
        Meeting01Config cfg;

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
        get("cv_fold", cfg.dataset.cv_fold);
        get("cv_num_folds", cfg.dataset.cv_num_folds);
        get("max_windows_per_recording", cfg.dataset.max_windows_per_recording);
        get("loso_max_train_windows", cfg.dataset.loso_max_train_windows);
        get("loso_max_val_windows", cfg.dataset.loso_max_val_windows);
        get("loso_max_test_windows", cfg.dataset.loso_max_test_windows);
        get("latex_data_dir", cfg.dataset.latex_data_dir);
        get("save_models", cfg.dataset.save_models);
        if (j.contains("dataset_sources")) parse_sources(j["dataset_sources"], cfg.dataset.sources);

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
        get("lstm_frame_size", cfg.model.lstm_frame_size);
        reject_renamed_time_steps(j);
        get("time_steps", cfg.model.time_steps);
        get("loss_function", cfg.model.loss_type);
        get("branch_hidden_size", cfg.model.branch_hidden_size);
        get("fusion_hidden_size", cfg.model.fusion_hidden_size);
        get("transformer_d_model", cfg.model.transformer_d_model);
        get("transformer_heads", cfg.model.transformer_heads);
        get("transformer_layers", cfg.model.transformer_layers);
        get("transformer_d_ff", cfg.model.transformer_d_ff);
        get("encoder_layer_spec", cfg.model.encoder_layer_spec);
        get("decoder_layer_spec", cfg.model.decoder_layer_spec);
        get("branch_encoder_layer_spec", cfg.model.branch_encoder_layer_spec);
        get("branch_decoder_layer_spec", cfg.model.branch_decoder_layer_spec);
        get("fusion_encoder_layer_spec", cfg.model.fusion_encoder_layer_spec);
        get("fusion_decoder_layer_spec", cfg.model.fusion_decoder_layer_spec);

        // Evaluation
        get("datasets", cfg.evaluation.datasets);
        get("encodings", cfg.evaluation.encodings);
        get("baselines", cfg.evaluation.baselines);
        get("snn_architectures", cfg.evaluation.snn_architectures);

        get("ga_population_size", cfg.evaluation.ga.population_size);
        get("ga_generations", cfg.evaluation.ga.generations);
        get("ga_min_layers", cfg.evaluation.ga.min_layers);
        get("ga_max_layers", cfg.evaluation.ga.max_layers);
        get("ga_min_width", cfg.evaluation.ga.min_width);
        get("ga_max_width", cfg.evaluation.ga.max_width);
        get("ga_voltage_threshold_min", cfg.evaluation.ga.voltage_threshold_min);
        get("ga_voltage_threshold_max", cfg.evaluation.ga.voltage_threshold_max);
        get("ga_alpha_min", cfg.evaluation.ga.alpha_min);
        get("ga_alpha_max", cfg.evaluation.ga.alpha_max);
        get("ga_crossover_prob", cfg.evaluation.ga.crossover_prob);
        get("ga_mutation_prob", cfg.evaluation.ga.mutation_prob);
        get("ga_tournament_k", cfg.evaluation.ga.tournament_k);
        get("ga_winner_seeds", cfg.evaluation.ga.winner_seeds);
        get("ga_seed", cfg.evaluation.ga.seed);
        get("ga_checkpoint_every_generations", cfg.evaluation.ga.checkpoint_every_generations);

        return cfg;
    }

    static void parse_ga(const nlohmann::json& sec, Ga& ga)
    {
        auto get = [&](const std::string& key, auto& field)
        {
            if (sec.contains(key)) field = sec[key].get<std::decay_t<decltype(field)>>();
        };
        get("population_size", ga.population_size);
        get("generations", ga.generations);
        get("min_layers", ga.min_layers);
        get("max_layers", ga.max_layers);
        get("min_width", ga.min_width);
        get("max_width", ga.max_width);
        get("voltage_threshold_min", ga.voltage_threshold_min);
        get("voltage_threshold_max", ga.voltage_threshold_max);
        get("alpha_min", ga.alpha_min);
        get("alpha_max", ga.alpha_max);
        get("crossover_prob", ga.crossover_prob);
        get("mutation_prob", ga.mutation_prob);
        get("tournament_k", ga.tournament_k);
        get("winner_seeds", ga.winner_seeds);
        get("seed", ga.seed);
        get("checkpoint_every_generations", ga.checkpoint_every_generations);
    }

    static Meeting01Config from_nested_json(const nlohmann::json& j)
    {
        Meeting01Config cfg;

        for (const auto* section : {"experiment", "dataset", "training", "model", "evaluation"})
        {
            if (!j.contains(section))
                throw std::invalid_argument(
                    std::string("Meeting01Config: required section missing: ") + section);
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
                    "Meeting01Config: required field missing: " + section_name + "." + key);
            field = sec[key].get<std::decay_t<decltype(field)>>();
        };

        // --- experiment (all required except optional flags) ---
        require(exp, "experiment", "run_tag", cfg.experiment.run_tag);
        require(exp, "experiment", "seed", cfg.experiment.seed);
        require(exp, "experiment", "repeats", cfg.experiment.repeats);
        get(exp, "seed_deterministic", cfg.experiment.seed_deterministic);
        get(exp, "check_determinism", cfg.experiment.check_determinism);

        // --- dataset ---
        require(dat, "dataset", "dataset_root", cfg.dataset.dataset_root);
        require(dat, "dataset", "window_size", cfg.dataset.window_size);
        require(dat, "dataset", "max_loaded_train_samples", cfg.dataset.max_loaded_train_samples);
        require(dat, "dataset", "max_validation_samples", cfg.dataset.max_validation_samples);
        get(dat, "cv_fold", cfg.dataset.cv_fold);
        get(dat, "cv_num_folds", cfg.dataset.cv_num_folds);
        get(dat, "max_windows_per_recording", cfg.dataset.max_windows_per_recording);
        get(dat, "loso_max_train_windows", cfg.dataset.loso_max_train_windows);
        get(dat, "loso_max_val_windows", cfg.dataset.loso_max_val_windows);
        get(dat, "loso_max_test_windows", cfg.dataset.loso_max_test_windows);
        get(dat, "results_dir", cfg.dataset.results_dir);
        get(dat, "latex_data_dir", cfg.dataset.latex_data_dir);
        get(dat, "save_models", cfg.dataset.save_models);
        if (dat.contains("sources")) parse_sources(dat["sources"], cfg.dataset.sources);

        // --- training ---
        require(trn, "training", "samples_per_batch", cfg.training.samples_per_batch);
        require(trn, "training", "epochs", cfg.training.epochs);
        require(trn, "training", "early_stop_patience", cfg.training.early_stop_patience);
        require(trn, "training", "learning_rate", cfg.training.learning_rate);
        get(trn, "batches_per_epoch", cfg.training.batches_per_epoch);
        get(trn, "learning_rate_biophysical", cfg.training.learning_rate_biophysical);
        get(trn, "beta1", cfg.training.beta1);
        get(trn, "adam_beta1", cfg.training.beta1);
        get(trn, "beta2", cfg.training.beta2);
        get(trn, "adam_beta2", cfg.training.beta2);
        get(trn, "epsilon", cfg.training.epsilon);
        get(trn, "adam_epsilon", cfg.training.epsilon);
        get(trn, "max_reconstruct_mean_deviation", cfg.training.max_reconstruct_mean_deviation);

        // --- model ---
        require(mdl, "model", "encoder_layer_spec", cfg.model.encoder_layer_spec);
        require(mdl, "model", "decoder_layer_spec", cfg.model.decoder_layer_spec);
        get(mdl, "latent_dim", cfg.model.latent_dim);
        get(mdl, "lstm_hidden_size", cfg.model.lstm_hidden_size);
        get(mdl, "lstm_frame_size", cfg.model.lstm_frame_size);
        reject_renamed_time_steps(mdl);
        get(mdl, "time_steps", cfg.model.time_steps);
        get(mdl, "loss_function", cfg.model.loss_type);
        get(mdl, "branch_hidden_size", cfg.model.branch_hidden_size);
        get(mdl, "fusion_hidden_size", cfg.model.fusion_hidden_size);
        get(mdl, "transformer_d_model", cfg.model.transformer_d_model);
        get(mdl, "transformer_heads", cfg.model.transformer_heads);
        get(mdl, "transformer_layers", cfg.model.transformer_layers);
        get(mdl, "transformer_d_ff", cfg.model.transformer_d_ff);
        get(mdl, "branch_encoder_layer_spec", cfg.model.branch_encoder_layer_spec);
        get(mdl, "branch_decoder_layer_spec", cfg.model.branch_decoder_layer_spec);
        get(mdl, "fusion_encoder_layer_spec", cfg.model.fusion_encoder_layer_spec);
        get(mdl, "fusion_decoder_layer_spec", cfg.model.fusion_decoder_layer_spec);

        // --- evaluation ---
        require(evl, "evaluation", "datasets", cfg.evaluation.datasets);
        require(evl, "evaluation", "encodings", cfg.evaluation.encodings);
        get(evl, "baselines", cfg.evaluation.baselines);
        require(evl, "evaluation", "snn_architectures", cfg.evaluation.snn_architectures);
        if (evl.contains("ga")) parse_ga(evl["ga"], cfg.evaluation.ga);

        return cfg;
    }
};

} // namespace meeting01
