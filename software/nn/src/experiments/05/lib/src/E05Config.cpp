#include "E05Config.hpp"

#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace e05
{

void E05Config::validate() const
{
    if (experiment.run_tag.empty())
        throw std::invalid_argument("E05Config: experiment.run_tag is required");
    if (experiment.seed == 0u)
        throw std::invalid_argument("E05Config: experiment.seed must be non-zero");
    if (experiment.repeats <= 0)
        throw std::invalid_argument("E05Config: experiment.repeats must be > 0");
    if (dataset.root.empty())
        throw std::invalid_argument("E05Config: dataset.root is required");

    auto valid_modality = dataset.modality == "voice" || dataset.modality == "eeg" ||
                          dataset.modality == "fused";
    if (!valid_modality)
        throw std::invalid_argument("E05Config: dataset.modality must be voice/eeg/fused");

    const bool valid_strategy = feature_extraction.strategy == "handcrafted" ||
                                feature_extraction.strategy == "autoencoder";
    if (!valid_strategy)
        throw std::invalid_argument(
            "E05Config: feature_extraction.strategy must be handcrafted or autoencoder");

    if (feature_extraction.strategy == "handcrafted")
    {
        if (feature_extraction.handcrafted.transform != "dtwpt")
            throw std::invalid_argument("E05Config: handcrafted.transform must be dtwpt");

        const auto valid_scale = feature_extraction.handcrafted.scale == "bark" ||
                                 feature_extraction.handcrafted.scale == "mel" ||
                                 feature_extraction.handcrafted.scale == "lfcc";
        if (!valid_scale)
            throw std::invalid_argument("E05Config: handcrafted.scale must be bark/mel/lfcc");

        if (feature_extraction.handcrafted.descriptors.empty())
            throw std::invalid_argument("E05Config: handcrafted.descriptors must not be empty");

        if (feature_extraction.handcrafted.dtwpt_level < 1)
            throw std::invalid_argument("E05Config: handcrafted.dtwpt_level must be >= 1");
    }
    else
    {
        if (feature_extraction.autoencoder.model != "lstm-ae")
            throw std::invalid_argument("E05Config: autoencoder.model must be lstm-ae");
        if (feature_extraction.autoencoder.encoder_layer_spec.empty())
            throw std::invalid_argument("E05Config: autoencoder.encoder_layer_spec is required");
        if (feature_extraction.autoencoder.decoder_layer_spec.empty())
            throw std::invalid_argument("E05Config: autoencoder.decoder_layer_spec is required");
    }

    if (classifier.type != "rnn" && classifier.type != "dsnn")
        throw std::invalid_argument("E05Config: classifier.type must be rnn or dsnn");

    auto valid_text_mode = classifier.text_mode == "dependent" ||
                           classifier.text_mode == "independent";
    if (!valid_text_mode)
        throw std::invalid_argument("E05Config: classifier.text_mode must be dependent/independent");

    if (classifier.layer_spec.size() < 3)
        throw std::invalid_argument(
            "E05Config: classifier.layer_spec must have at least 3 items: "
            "linear:H:act, residual:D, linear:N_speakers:identity");

    const auto starts_with = [](const std::string& s, const std::string& p)
    {
        return s.rfind(p, 0) == 0;
    };

    if (!starts_with(classifier.layer_spec.front(), "linear:"))
        throw std::invalid_argument("E05Config: first classifier.layer_spec item must start with linear:");

    const bool has_residual = std::any_of(classifier.layer_spec.begin(), classifier.layer_spec.end(),
        [&](const std::string& s) { return starts_with(s, "residual:"); });
    if (!has_residual)
        throw std::invalid_argument("E05Config: classifier.layer_spec must include residual:D item");

    if (!starts_with(classifier.layer_spec.back(), "linear:N_speakers:"))
        throw std::invalid_argument(
            "E05Config: last classifier.layer_spec item must start with linear:N_speakers:");

    if (training.epochs <= 0)
        throw std::invalid_argument("E05Config: training.epochs must be > 0");
    if (training.learning_rate <= 0.0f)
        throw std::invalid_argument("E05Config: training.learning_rate must be > 0");
    if (training.samples_per_batch <= 0)
        throw std::invalid_argument("E05Config: training.samples_per_batch must be > 0");
    if (training.early_stop_patience < 0)
        throw std::invalid_argument("E05Config: training.early_stop_patience must be >= 0");
    if (training.k_folds < 2)
        throw std::invalid_argument("E05Config: training.k_folds must be >= 2");
    if (training.weight_decay < 0.0f)
        throw std::invalid_argument("E05Config: training.weight_decay must be >= 0");
    if (training.firing_rate_reg_lambda < 0.0f)
        throw std::invalid_argument("E05Config: training.firing_rate_reg_lambda must be >= 0");
    if (training.firing_rate_min < 0.0f || training.firing_rate_max > 1.0f ||
        training.firing_rate_min > training.firing_rate_max)
        throw std::invalid_argument(
            "E05Config: require 0 <= firing_rate_min <= firing_rate_max <= 1");
}

E05Config E05Config::from_json(const nlohmann::json& j)
{
    E05Config cfg;

    // Experiment
    if (j.contains("experiment"))
    {
        const auto& e = j["experiment"];
        if (e.contains("run_tag")) cfg.experiment.run_tag = e["run_tag"];
        if (e.contains("seed")) cfg.experiment.seed = e["seed"].get<std::uint32_t>();
        if (e.contains("repeats")) cfg.experiment.repeats = e["repeats"];
        if (e.contains("seed_deterministic")) cfg.experiment.seed_deterministic = e["seed_deterministic"];
    }

    // Dataset
    if (j.contains("dataset"))
    {
        const auto& d = j["dataset"];
        if (d.contains("root")) cfg.dataset.root = d["root"];
        if (d.contains("results_dir")) cfg.dataset.results_dir = d["results_dir"];
        if (d.contains("modality")) cfg.dataset.modality = d["modality"];
        if (d.contains("max_samples")) cfg.dataset.max_samples = d["max_samples"];
    }

    // Feature extraction
    if (j.contains("feature_extraction"))
    {
        const auto& fe = j["feature_extraction"];
        if (fe.contains("strategy")) cfg.feature_extraction.strategy = fe["strategy"];

        if (fe.contains("handcrafted"))
        {
            const auto& hc = fe["handcrafted"];
            if (hc.contains("transform")) cfg.feature_extraction.handcrafted.transform = hc["transform"];
            if (hc.contains("scale")) cfg.feature_extraction.handcrafted.scale = hc["scale"];
            if (hc.contains("descriptors"))
                cfg.feature_extraction.handcrafted.descriptors = hc["descriptors"].get<std::vector<std::string>>();
            if (hc.contains("dtwpt_level")) cfg.feature_extraction.handcrafted.dtwpt_level = hc["dtwpt_level"];
        }

        if (fe.contains("autoencoder"))
        {
            const auto& ae = fe["autoencoder"];
            if (ae.contains("model")) cfg.feature_extraction.autoencoder.model = ae["model"];
            if (ae.contains("encoder_layer_spec"))
                cfg.feature_extraction.autoencoder.encoder_layer_spec = ae["encoder_layer_spec"].get<std::vector<std::string>>();
            if (ae.contains("decoder_layer_spec"))
                cfg.feature_extraction.autoencoder.decoder_layer_spec = ae["decoder_layer_spec"].get<std::vector<std::string>>();
        }
    }

    // Paraconsistent
    if (j.contains("paraconsistent"))
    {
        const auto& p = j["paraconsistent"];
        if (p.contains("enabled")) cfg.paraconsistent.enabled = p["enabled"];
    }

    // Classifier
    if (j.contains("classifier"))
    {
        const auto& c = j["classifier"];
        if (c.contains("type")) cfg.classifier.type = c["type"];
        if (c.contains("layer_spec"))
            cfg.classifier.layer_spec = c["layer_spec"].get<std::vector<std::string>>();
        if (c.contains("text_mode")) cfg.classifier.text_mode = c["text_mode"];
    }

    // Training
    if (j.contains("training"))
    {
        const auto& t = j["training"];
        if (t.contains("epochs")) cfg.training.epochs = t["epochs"];
        if (t.contains("learning_rate")) cfg.training.learning_rate = t["learning_rate"].get<float>();
        if (t.contains("samples_per_batch")) cfg.training.samples_per_batch = t["samples_per_batch"];
        if (t.contains("early_stop_patience")) cfg.training.early_stop_patience = t["early_stop_patience"];
        if (t.contains("k_folds")) cfg.training.k_folds = t["k_folds"];
        if (t.contains("nested_cv")) cfg.training.nested_cv = t["nested_cv"];
        if (t.contains("weight_decay")) cfg.training.weight_decay = t["weight_decay"].get<float>();
        if (t.contains("firing_rate_reg_lambda"))
            cfg.training.firing_rate_reg_lambda = t["firing_rate_reg_lambda"].get<float>();
        if (t.contains("firing_rate_min")) cfg.training.firing_rate_min = t["firing_rate_min"].get<float>();
        if (t.contains("firing_rate_max")) cfg.training.firing_rate_max = t["firing_rate_max"].get<float>();
    }

    return cfg;
}

E05Config E05Config::from_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("E05Config: cannot open " + path);
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

} // namespace e05
