#include "Config.hpp"

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

        // -------------------- window --------------------
        cfg.duration_sec = node["window"]["duration_sec"].as<double>();
        cfg.overlap_percent = node["window"]["overlap_percent"].as<int>();

        if (cfg.overlap_percent < 0 || cfg.overlap_percent > 100)
        {
            throw std::runtime_error("overlap_percent must be in [0, 100]");
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

        // -------------------- classifier --------------------
        cfg.type = node["classifier"]["type"].as<std::string>();
        cfg.implementation = node["classifier"]["implementation"].as<std::string>();

        // -------------------- dataset --------------------
        cfg.sampling_rate = node["dataset"]["sampling_rate"].as<int>();
        cfg.eeg_sampling_rate = node["dataset"]["eeg_sampling_rate"].as<int>();

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
