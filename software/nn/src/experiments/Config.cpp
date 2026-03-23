/**
 * @file Config.cpp
 * @brief YAML parsing + validation for experiment configuration.
 *
 * Philosophy:
 * - Fail early with clear messages when the experiment spec is inconsistent.
 * - Keep some parameters fixed in early phases to reduce degrees of freedom and
 *   make results comparable.
 */

#include "Config.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

auto Config::load(const std::string& path) -> std::optional<Config>
{
    nlohmann::json node;

    std::ifstream in(path);
    if (!in)
    {
        std::cerr << "Config error (BadFile): Unable to open " << path << '\n';
        return std::nullopt;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    try
    {
        node = nlohmann::json::parse(ss.str());
    }
    catch (const nlohmann::json::parse_error& e)
    {
        std::cerr << "Config error (ParseError): " << e.what() << '\n';
        return std::nullopt;
    }

    try
    {
        Config cfg;

        constexpr double kPhase0WindowSec = 1.5;
        constexpr int kPhase0OverlapPercent = 50;
        constexpr double kNormMin = 0.0;
        constexpr double kNormMax = 1.0;

        // -------------------- window --------------------
        // -------------------- window --------------------
        cfg.duration_sec = node.at("window").at("duration_sec").get<double>();
        cfg.overlap_percent = node.at("window").at("overlap_percent").get<int>();

        if (cfg.overlap_percent < 0 || cfg.overlap_percent > 100)
        {
            throw std::runtime_error("overlap_percent must be in [0, 100]");
        }

        if (std::abs(cfg.duration_sec - kPhase0WindowSec) > 1e-6)
        {
            throw std::runtime_error("window.duration_sec is frozen to 1.5 seconds for PHASE 0");
        }

        if (cfg.overlap_percent != kPhase0OverlapPercent)
        {
            throw std::runtime_error("window.overlap_percent is frozen to 50 for PHASE 0");
        }

        // -------------------- normalization --------------------
        // -------------------- normalization --------------------
        cfg.range = node.at("normalization").at("range").get<std::vector<double>>();
        if (cfg.range.size() != 2)
        {
            throw std::runtime_error("normalization.range must have exactly 2 elements");
        }

        cfg.method = node.at("normalization").at("method").get<std::string>();
        cfg.paraconsistent_prerequisite =
            node.at("normalization").at("paraconsistent_prerequisite").get<bool>();

        if (std::abs(cfg.range[0] - kNormMin) > 1e-6 || std::abs(cfg.range[1] - kNormMax) > 1e-6)
        {
            throw std::runtime_error("normalization.range must be [0, 1] for PHASE 0");
        }

        if (cfg.method != "min-max")
        {
            throw std::runtime_error("normalization.method must be 'min-max' for PHASE 0");
        }

        if (!cfg.paraconsistent_prerequisite)
        {
            throw std::runtime_error(
                "normalization.paraconsistent_prerequisite must stay true for PHASE 0");
        }

        // -------------------- classifier --------------------
        // -------------------- classifier --------------------
        cfg.type = node.at("classifier").at("type").get<std::string>();
        cfg.implementation = node.at("classifier").at("implementation").get<std::string>();
        cfg.resnet_hidden_dim = node.at("classifier").at("hidden_dim").get<int>();
        cfg.resnet_depth = node.at("classifier").at("depth").get<int>();
        cfg.learning_rate = node.at("classifier").at("learning_rate").get<float>();

        const bool type_ok = (cfg.type == "ResNet-SNN" || cfg.type == "ResNet");
        if (!type_ok)
        {
            throw std::runtime_error("classifier.type must be 'ResNet-SNN' for PHASE 0");
        }

        if (cfg.implementation != "SimpleResNet")
        {
            throw std::runtime_error("classifier.implementation must be 'SimpleResNet'");
        }

        // -------------------- dataset --------------------
        // -------------------- dataset --------------------
        cfg.dataset_base_path = node.at("dataset").at("base_path").get<std::string>();
        cfg.sampling_rate = node.at("dataset").at("sampling_rate").get<int>();
        cfg.eeg_sampling_rate = node.at("dataset").at("eeg_sampling_rate").get<int>();

        if (cfg.dataset_base_path.empty())
        {
            throw std::runtime_error("dataset.base_path must not be empty");
        }

        if (cfg.sampling_rate <= 0 || cfg.eeg_sampling_rate <= 0)
        {
            throw std::runtime_error("sampling rates must be positive");
        }

        // -------------------- paraconsistent --------------------
        // -------------------- paraconsistent --------------------
        cfg.enabled = node.at("paraconsistent").at("enabled").get<bool>();
        cfg.optimal_point =
            node.at("paraconsistent").at("optimal_point").get<std::vector<double>>();

        if (cfg.optimal_point.size() != 2)
        {
            throw std::runtime_error("paraconsistent.optimal_point must have exactly 2 elements");
        }

        // -------------------- experiment --------------------
        // -------------------- experiment --------------------
        cfg.seed = node.at("experiment").at("seed").get<int>();
        cfg.cross_validation = node.at("experiment").at("cross_validation").get<bool>();
        cfg.folds = node.at("experiment").at("folds").get<int>();
        cfg.batch_size = node.at("experiment").at("batch_size").get<int>();
        cfg.max_epochs = node.at("experiment").at("max_epochs").get<int>();

        if (cfg.seed < 0)
        {
            throw std::runtime_error("experiment.seed must be non-negative");
        }

        if (cfg.folds < 1)
        {
            throw std::runtime_error("experiment.folds must be >= 1");
        }

        if (cfg.batch_size < 1)
        {
            throw std::runtime_error("experiment.batch_size must be >= 1");
        }

        if (cfg.max_epochs < 1)
        {
            throw std::runtime_error("experiment.max_epochs must be >= 1");
        }

        if (cfg.resnet_hidden_dim < 1 || cfg.resnet_depth < 1)
        {
            throw std::runtime_error("classifier hidden_dim and depth must be positive");
        }

        if (!(cfg.learning_rate > 0.0F))
        {
            throw std::runtime_error("classifier.learning_rate must be positive");
        }

        // -------------------- output --------------------
        // -------------------- output --------------------
        cfg.results_dir = node.at("output").at("results_dir").get<std::string>();
        cfg.metrics_file = node.at("output").at("metrics_file").get<std::string>();
        cfg.torch_state_file = node.at("output").at("torch_state_file").get<std::string>();

        if (cfg.results_dir.empty() || cfg.metrics_file.empty() || cfg.torch_state_file.empty())
        {
            throw std::runtime_error("output paths must not be empty");
        }

        return cfg;
    }
    catch (const nlohmann::json::type_error& e)
    {
        std::cerr << "Config error (TypeError): " << e.what() << '\n';
    }
    catch (const std::out_of_range& e)
    {
        std::cerr << "Config error (MissingKey): " << e.what() << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << "Config error (Validation): " << e.what() << '\n';
    }

    return std::nullopt;
}
