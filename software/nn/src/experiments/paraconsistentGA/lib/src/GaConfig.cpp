#include "GaConfig.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace pga
{

double GaConfig::autoencoder_budget_ms() const
{
    return std::max(0.0, constraints.latency_ceiling_ms - constraints.fixed_pipeline_cost_ms);
}

void GaConfig::validate() const
{
    // The embedded thesis config must be a valid autoencoder feature-extraction
    // profile: the GA only ever evolves AE architectures.
    base.validate();

    if (base.feature_extraction.strategy != "autoencoder")
        throw std::invalid_argument(
            "GaConfig: base.feature_extraction.strategy must be 'autoencoder'");

    const auto& model = base.feature_extraction.autoencoder.model;
    if (model != "snn-ae" && model != "ann-ae")
        throw std::invalid_argument(
            "GaConfig: autoencoder.model must be snn-ae or ann-ae (lstm-ae not supported by the "
            "GA)");

    if (ga.population_size < 4)
        throw std::invalid_argument("GaConfig: ga.population_size must be >= 4");
    if (ga.generations < 1) throw std::invalid_argument("GaConfig: ga.generations must be >= 1");
    if (ga.n_seeds < 1) throw std::invalid_argument("GaConfig: ga.n_seeds must be >= 1");
    if (ga.n_losers < 0) throw std::invalid_argument("GaConfig: ga.n_losers must be >= 0");
    if (ga.n_losers >= ga.population_size)
        throw std::invalid_argument(
            "GaConfig: ga.n_losers (" + std::to_string(ga.n_losers) +
            ") must be < population_size (" + std::to_string(ga.population_size) +
            "); at least one elite slot must remain, otherwise selection keeps no winners.");
    if (ga.seed == 0u) throw std::invalid_argument("GaConfig: ga.seed must be non-zero");
    if (ga.crossover_prob < 0.0 || ga.crossover_prob > 1.0)
        throw std::invalid_argument("GaConfig: ga.crossover_prob must be in [0,1]");
    if (ga.mutation_prob < 0.0 || ga.mutation_prob > 1.0)
        throw std::invalid_argument("GaConfig: ga.mutation_prob must be in [0,1]");
    if (ga.tournament_k < 2) throw std::invalid_argument("GaConfig: ga.tournament_k must be >= 2");

    // Free-architecture bounds: depth range and per-layer width range.
    if (ga.bounds.min_width < 1)
        throw std::invalid_argument("GaConfig: ga.bounds.min_width must be >= 1");
    if (ga.bounds.max_width < ga.bounds.min_width)
        throw std::invalid_argument("GaConfig: ga.bounds.max_width must be >= min_width");
    if (ga.bounds.min_layers < 1)
        throw std::invalid_argument("GaConfig: ga.bounds.min_layers must be >= 1");
    if (ga.bounds.max_layers < ga.bounds.min_layers)
        throw std::invalid_argument("GaConfig: ga.bounds.max_layers must be >= min_layers");
    // Strictly decreasing widths need one distinct integer per layer, so the deepest
    // requested genome must fit in the width range. Reject rather than silently
    // truncate every deep genome to a shallower one.
    const int width_room = ga.bounds.max_width - ga.bounds.min_width + 1;
    if (ga.bounds.max_layers > width_room)
        throw std::invalid_argument(
            "GaConfig: ga.bounds.max_layers (" + std::to_string(ga.bounds.max_layers) +
            ") exceeds the number of distinct widths available (" + std::to_string(width_room) +
            " = max_width - min_width + 1). Strictly decreasing layers cannot be formed; "
            "widen the width range or reduce max_layers.");
    if (is_snn_population() && ga.bounds.encoding_choices.empty())
        throw std::invalid_argument("GaConfig: encoding_choices must be non-empty for snn-ae");

    if (constraints.latency_ceiling_ms <= 0.0)
        throw std::invalid_argument("GaConfig: constraints.latency_ceiling_ms must be > 0");
    if (constraints.fixed_pipeline_cost_ms < 0.0)
        throw std::invalid_argument("GaConfig: constraints.fixed_pipeline_cost_ms must be >= 0");
    if (constraints.ns_per_mac <= 0.0)
        throw std::invalid_argument("GaConfig: constraints.ns_per_mac must be > 0");
    if (constraints.tau_rec < 0.0)
        throw std::invalid_argument("GaConfig: constraints.tau_rec must be >= 0");
    if (autoencoder_budget_ms() <= 0.0)
        throw std::invalid_argument(
            "GaConfig: fixed_pipeline_cost_ms >= latency_ceiling_ms leaves the autoencoder no "
            "budget");

    if (run_tag.empty()) throw std::invalid_argument("GaConfig: run_tag is required");
    if (results_dir.empty()) throw std::invalid_argument("GaConfig: results_dir is required");
}

namespace
{
template <typename T>
std::vector<T> get_vec_or(const nlohmann::json& j, const char* key, std::vector<T> fallback)
{
    if (j.contains(key)) return j[key].get<std::vector<T>>();
    return fallback;
}
} // namespace

GaConfig GaConfig::from_json(const nlohmann::json& j)
{
    GaConfig cfg;

    // The whole thesis config is reused verbatim from the same JSON object.
    cfg.base = thesis::ThesisConfig::from_json(j);

    if (j.contains("results_dir")) cfg.results_dir = j["results_dir"];
    if (j.contains("run_tag"))
        cfg.run_tag = j["run_tag"];
    else if (!cfg.base.experiment.run_tag.empty())
        cfg.run_tag = cfg.base.experiment.run_tag;

    if (j.contains("ga"))
    {
        const auto& g = j["ga"];
        if (g.contains("population_size")) cfg.ga.population_size = g["population_size"];
        if (g.contains("generations")) cfg.ga.generations = g["generations"];
        if (g.contains("n_seeds")) cfg.ga.n_seeds = g["n_seeds"];
        if (g.contains("n_losers")) cfg.ga.n_losers = g["n_losers"];
        if (g.contains("crossover_prob")) cfg.ga.crossover_prob = g["crossover_prob"].get<double>();
        if (g.contains("mutation_prob")) cfg.ga.mutation_prob = g["mutation_prob"].get<double>();
        if (g.contains("seed")) cfg.ga.seed = g["seed"].get<std::uint32_t>();
        if (g.contains("tournament_k")) cfg.ga.tournament_k = g["tournament_k"];

        if (g.contains("bounds"))
        {
            const auto& b = g["bounds"];
            if (b.contains("min_layers")) cfg.ga.bounds.min_layers = b["min_layers"];
            if (b.contains("max_layers")) cfg.ga.bounds.max_layers = b["max_layers"];
            if (b.contains("min_width")) cfg.ga.bounds.min_width = b["min_width"];
            if (b.contains("max_width")) cfg.ga.bounds.max_width = b["max_width"];
            cfg.ga.bounds.encoding_choices =
                get_vec_or<std::string>(b, "encoding_choices", cfg.ga.bounds.encoding_choices);
            if (b.contains("evolve_temporal"))
                cfg.ga.bounds.evolve_temporal = b["evolve_temporal"].get<bool>();
            cfg.ga.bounds.time_steps_choices =
                get_vec_or<int>(b, "time_steps_choices", cfg.ga.bounds.time_steps_choices);
            if (b.contains("voltage_threshold_min"))
                cfg.ga.bounds.voltage_threshold_min = b["voltage_threshold_min"].get<float>();
            if (b.contains("voltage_threshold_max"))
                cfg.ga.bounds.voltage_threshold_max = b["voltage_threshold_max"].get<float>();
        }
    }

    if (j.contains("constraints"))
    {
        const auto& c = j["constraints"];
        if (c.contains("latency_ceiling_ms"))
            cfg.constraints.latency_ceiling_ms = c["latency_ceiling_ms"].get<double>();
        if (c.contains("fixed_pipeline_cost_ms"))
            cfg.constraints.fixed_pipeline_cost_ms = c["fixed_pipeline_cost_ms"].get<double>();
        if (c.contains("latency_calibrated"))
            cfg.constraints.latency_calibrated = c["latency_calibrated"].get<bool>();
        if (c.contains("ns_per_mac")) cfg.constraints.ns_per_mac = c["ns_per_mac"].get<double>();
        if (c.contains("tau_rec")) cfg.constraints.tau_rec = c["tau_rec"].get<double>();
    }

    // Feature extraction must run without a classifier (Phase-00-like): the GA
    // scores by paraconsistent d_penalized, never trains the authentication head.
    cfg.base.classifier.enabled = false;
    cfg.base.paraconsistent.enabled = true;

    return cfg;
}

GaConfig GaConfig::from_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("GaConfig: cannot open " + path);
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

} // namespace pga
