#include "ThesisConfig.hpp"

#include <algorithm>
#include <fstream>
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

void ThesisConfig::validate() const
{
    if (experiment.run_tag.empty())
        throw std::invalid_argument("ThesisConfig: experiment.run_tag is required");
    if (experiment.seed == 0u)
        throw std::invalid_argument("ThesisConfig: experiment.seed must be non-zero");
    if (experiment.repeats <= 0)
        throw std::invalid_argument("ThesisConfig: experiment.repeats must be > 0");
    if (dataset.root.empty()) throw std::invalid_argument("ThesisConfig: dataset.root is required");

    auto valid_modality =
        dataset.modality == "voice" || dataset.modality == "eeg" || dataset.modality == "fused";
    if (!valid_modality)
        throw std::invalid_argument("ThesisConfig: dataset.modality must be voice/eeg/fused");

    auto valid_fusion_mode = dataset.fusion_mode == "early" || dataset.fusion_mode == "late";
    if (!valid_fusion_mode)
        throw std::invalid_argument("ThesisConfig: dataset.fusion_mode must be early/late");

    const bool valid_strategy = feature_extraction.strategy == "handcrafted" ||
                                feature_extraction.strategy == "autoencoder";
    if (!valid_strategy)
        throw std::invalid_argument(
            "ThesisConfig: feature_extraction.strategy must be handcrafted or autoencoder");

    if (feature_extraction.strategy == "handcrafted")
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
    else
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

ThesisConfig ThesisConfig::from_json(const nlohmann::json& j)
{
    ThesisConfig cfg;

    // Experiment
    if (j.contains("experiment"))
    {
        const auto& e = j["experiment"];
        if (e.contains("run_tag")) cfg.experiment.run_tag = e["run_tag"];
        if (e.contains("seed")) cfg.experiment.seed = e["seed"].get<std::uint32_t>();
        if (e.contains("repeats")) cfg.experiment.repeats = e["repeats"];
        if (e.contains("seed_deterministic"))
            cfg.experiment.seed_deterministic = e["seed_deterministic"];
    }

    // Dataset
    if (j.contains("numerics"))
    {
        const auto& n = j["numerics"];
        if (n.contains("exact_activations"))
            cfg.numerics.exact_activations = n["exact_activations"].get<bool>();
    }

    if (j.contains("dataset"))
    {
        const auto& d = j["dataset"];
        if (d.contains("root")) cfg.dataset.root = d["root"];
        if (d.contains("results_dir")) cfg.dataset.results_dir = d["results_dir"];
        if (d.contains("modality")) cfg.dataset.modality = d["modality"];
        if (d.contains("fusion_mode")) cfg.dataset.fusion_mode = d["fusion_mode"];
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
            if (hc.contains("transform"))
                cfg.feature_extraction.handcrafted.transform = hc["transform"];
            if (hc.contains("scale")) cfg.feature_extraction.handcrafted.scale = hc["scale"];
            if (hc.contains("descriptors"))
                cfg.feature_extraction.handcrafted.descriptors =
                    hc["descriptors"].get<std::vector<std::string>>();
            if (hc.contains("dtwpt_level"))
                cfg.feature_extraction.handcrafted.dtwpt_level = hc["dtwpt_level"];
            if (hc.contains("wavelet")) cfg.feature_extraction.handcrafted.wavelet = hc["wavelet"];
            if (hc.contains("cepstral"))
                cfg.feature_extraction.handcrafted.cepstral = hc["cepstral"];
        }

        if (fe.contains("autoencoder"))
        {
            const auto& ae = fe["autoencoder"];
            if (ae.contains("model")) cfg.feature_extraction.autoencoder.model = ae["model"];
            if (ae.contains("encoder_layer_spec"))
                cfg.feature_extraction.autoencoder.encoder_layer_spec =
                    ae["encoder_layer_spec"].get<std::vector<std::string>>();
            if (ae.contains("decoder_layer_spec"))
                cfg.feature_extraction.autoencoder.decoder_layer_spec =
                    ae["decoder_layer_spec"].get<std::vector<std::string>>();
            if (ae.contains("encoding"))
                cfg.feature_extraction.autoencoder.encoding = ae["encoding"];
            if (ae.contains("time_steps"))
                cfg.feature_extraction.autoencoder.time_steps = ae["time_steps"];
            if (ae.contains("voltage_threshold"))
                cfg.feature_extraction.autoencoder.voltage_threshold = ae["voltage_threshold"];
            if (ae.contains("firing_rate_reg_lambda"))
                cfg.feature_extraction.autoencoder.firing_rate_reg_lambda =
                    ae["firing_rate_reg_lambda"].get<float>();
            if (ae.contains("firing_rate_min"))
                cfg.feature_extraction.autoencoder.firing_rate_min =
                    ae["firing_rate_min"].get<float>();
            if (ae.contains("firing_rate_max"))
                cfg.feature_extraction.autoencoder.firing_rate_max =
                    ae["firing_rate_max"].get<float>();
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
        if (c.contains("enabled")) cfg.classifier.enabled = c["enabled"];
    }

    // Training
    if (j.contains("training"))
    {
        const auto& t = j["training"];
        if (t.contains("epochs")) cfg.training.epochs = t["epochs"];
        if (t.contains("learning_rate"))
            cfg.training.learning_rate = t["learning_rate"].get<float>();
        if (t.contains("samples_per_batch"))
            cfg.training.samples_per_batch = t["samples_per_batch"];
        if (t.contains("early_stop_patience"))
            cfg.training.early_stop_patience = t["early_stop_patience"];
        if (t.contains("k_folds")) cfg.training.k_folds = t["k_folds"];
        if (t.contains("nested_cv")) cfg.training.nested_cv = t["nested_cv"];
        if (t.contains("standardize_features"))
            cfg.training.standardize_features = t["standardize_features"];
        if (t.contains("optimizer_type"))
            cfg.training.optimizer_type = t["optimizer_type"].get<std::string>();
        if (t.contains("optimizer_momentum"))
            cfg.training.optimizer_momentum = t["optimizer_momentum"].get<float>();
        if (t.contains("gradient_clip_norm"))
            cfg.training.gradient_clip_norm = t["gradient_clip_norm"].get<float>();
        if (t.contains("weight_decay")) cfg.training.weight_decay = t["weight_decay"].get<float>();
        if (t.contains("firing_rate_reg_lambda"))
            cfg.training.firing_rate_reg_lambda = t["firing_rate_reg_lambda"].get<float>();
        if (t.contains("firing_rate_min"))
            cfg.training.firing_rate_min = t["firing_rate_min"].get<float>();
        if (t.contains("firing_rate_max"))
            cfg.training.firing_rate_max = t["firing_rate_max"].get<float>();
        if (t.contains("batch_normalization"))
            cfg.training.batch_normalization = t["batch_normalization"];
        if (t.contains("tdbn_alpha")) cfg.training.tdbn_alpha = t["tdbn_alpha"].get<float>();
    }

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
