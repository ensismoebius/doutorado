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
#include <iostream>

auto Config::load(const std::string& path) -> std::optional<Config>
{
    YAML::Node node;

    try
    {
        node = YAML::LoadFile(path);
    }
    catch (const YAML::BadFile& e)
    {
        std::cerr << "Config error (BadFile): " << e.what() << '\n';
        return std::nullopt;
    }
    catch (const YAML::ParserException& e)
    {
        std::cerr << "Config error (ParserException): " << e.what() << '\n';
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
        cfg.duration_sec = node["window"]["duration_sec"].as<double>();
        cfg.overlap_percent = node["window"]["overlap_percent"].as<int>();

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
        cfg.range = node["normalization"]["range"].as<std::vector<double>>();
        if (cfg.range.size() != 2)
        {
            throw std::runtime_error("normalization.range must have exactly 2 elements");
        }

        cfg.method = node["normalization"]["method"].as<std::string>();
        cfg.paraconsistent_prerequisite =
            node["normalization"]["paraconsistent_prerequisite"].as<bool>();

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
        cfg.type = node["classifier"]["type"].as<std::string>();
        cfg.implementation = node["classifier"]["implementation"].as<std::string>();
        cfg.resnet_hidden_dim = node["classifier"]["hidden_dim"].as<int>();
        cfg.resnet_depth = node["classifier"]["depth"].as<int>();
        cfg.learning_rate = node["classifier"]["learning_rate"].as<float>();

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
        cfg.dataset_base_path = node["dataset"]["base_path"].as<std::string>();
        cfg.sampling_rate = node["dataset"]["sampling_rate"].as<int>();
        cfg.eeg_sampling_rate = node["dataset"]["eeg_sampling_rate"].as<int>();

        if (cfg.dataset_base_path.empty())
        {
            throw std::runtime_error("dataset.base_path must not be empty");
        }

        if (cfg.sampling_rate <= 0 || cfg.eeg_sampling_rate <= 0)
        {
            throw std::runtime_error("sampling rates must be positive");
        }

        // -------------------- paraconsistent --------------------
        cfg.enabled = node["paraconsistent"]["enabled"].as<bool>();
        cfg.optimal_point = node["paraconsistent"]["optimal_point"].as<std::vector<double>>();

        if (cfg.optimal_point.size() != 2)
        {
            throw std::runtime_error("paraconsistent.optimal_point must have exactly 2 elements");
        }

        // -------------------- experiment --------------------
        cfg.seed = node["experiment"]["seed"].as<int>();
        cfg.cross_validation = node["experiment"]["cross_validation"].as<bool>();
        cfg.folds = node["experiment"]["folds"].as<int>();
        cfg.batch_size = node["experiment"]["batch_size"].as<int>();
        cfg.max_epochs = node["experiment"]["max_epochs"].as<int>();

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
        cfg.results_dir = node["output"]["results_dir"].as<std::string>();
        cfg.metrics_file = node["output"]["metrics_file"].as<std::string>();
        cfg.torch_state_file = node["output"]["torch_state_file"].as<std::string>();

        if (cfg.results_dir.empty() || cfg.metrics_file.empty() || cfg.torch_state_file.empty())
        {
            throw std::runtime_error("output paths must not be empty");
        }

        return cfg;
    }
    catch (const YAML::BadConversion& e)
    {
        std::cerr << "Config error (BadConversion): " << e.what() << '\n';
    }
    catch (const YAML::InvalidNode& e)
    {
        std::cerr << "Config error (InvalidNode): " << e.what() << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << "Config error (Validation): " << e.what() << '\n';
    }

    return std::nullopt;
}
