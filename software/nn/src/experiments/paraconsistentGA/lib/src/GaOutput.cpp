#include "GaOutput.hpp"

#include <fstream>
#include <stdexcept>

#include "ThesisOutput.hpp" // thesis::ensure_dir (reused)
#include "nlohmann/json.hpp"

namespace pga
{

namespace
{
nlohmann::json genome_to_json(const Genome& g)
{
    return {{"hidden", g.hidden},
        {"latent", g.latent},
        {"encoding", g.encoding},
        {"time_steps", g.time_steps},
        {"voltage_threshold", g.voltage_threshold}};
}

nlohmann::json individual_to_json(const Individual& ind)
{
    return {{"genome", genome_to_json(ind.genome)},
        {"born_generation", ind.born_generation},
        {"alpha", ind.alpha},
        {"beta", ind.beta},
        {"g1", ind.g1},
        {"g2", ind.g2},
        {"d_truth", ind.d_truth},
        {"d_penalized_mean", ind.d_penalized_mean},
        {"d_penalized_std", ind.d_penalized_std},
        {"latent_activity", ind.latent_activity},
        {"param_count", ind.param_count},
        {"inference_cost", ind.inference_cost},
        {"est_latency_ms", ind.est_latency_ms},
        {"feasible", ind.feasible},
        {"constraint_violation", ind.constraint_violation},
        {"rank", ind.rank}};
}
} // namespace

void write_individuals_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<Individual>& history)
{
    thesis::ensure_dir(results_dir);
    const std::string path = results_dir + "/pga_" + run_tag + "_individuals.csv";
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("GaOutput: cannot write " + path);

    f << "born_generation,hidden,latent,encoding,time_steps,voltage_threshold,"
         "alpha,beta,g1,g2,d_truth,d_penalized_mean,d_penalized_std,latent_activity,"
         "param_count,inference_cost,est_latency_ms,feasible,constraint_violation\n";
    for (const auto& ind : history)
    {
        const auto& g = ind.genome;
        f << ind.born_generation << ',' << g.hidden << ',' << g.latent << ',' << g.encoding << ','
          << g.time_steps << ',' << g.voltage_threshold << ',' << ind.alpha << ',' << ind.beta
          << ',' << ind.g1 << ',' << ind.g2 << ',' << ind.d_truth << ',' << ind.d_penalized_mean
          << ',' << ind.d_penalized_std << ',' << ind.latent_activity << ',' << ind.param_count
          << ',' << ind.inference_cost << ',' << ind.est_latency_ms << ',' << (ind.feasible ? 1 : 0)
          << ',' << ind.constraint_violation << '\n';
    }
}

void write_pareto_front_json(const std::string& results_dir,
    const std::string& run_tag,
    const GaConfig& cfg,
    const GaResult& result)
{
    thesis::ensure_dir(results_dir);
    const std::string path = results_dir + "/pga_" + run_tag + "_pareto.json";

    nlohmann::json j;
    j["run_tag"] = run_tag;
    j["population"] = {{"model", cfg.base.feature_extraction.autoencoder.model},
        {"modality", cfg.base.dataset.modality},
        {"fusion_mode", cfg.base.dataset.fusion_mode}};
    j["ga"] = {{"population_size", cfg.ga.population_size},
        {"generations", cfg.ga.generations},
        {"generations_run", result.generations_run},
        {"n_seeds", cfg.ga.n_seeds},
        {"seed", cfg.ga.seed},
        {"crossover_prob", cfg.ga.crossover_prob},
        {"mutation_prob", cfg.ga.mutation_prob}};
    j["constraints"] = {{"latency_ceiling_ms", cfg.constraints.latency_ceiling_ms},
        {"fixed_pipeline_cost_ms", cfg.constraints.fixed_pipeline_cost_ms},
        {"autoencoder_budget_ms", cfg.autoencoder_budget_ms()},
        {"latency_calibrated", cfg.constraints.latency_calibrated},
        {"ns_per_mac", cfg.constraints.ns_per_mac},
        {"tau_rec", cfg.constraints.tau_rec}};
    if (!cfg.constraints.latency_calibrated)
        j["warnings"] = nlohmann::json::array(
            {"latency proxy is UNCALIBRATED (constraints.latency_calibrated=false): "
             "fixed_pipeline_cost_ms and ns_per_mac are estimates, not measured on target "
             "hardware"});

    j["n_evaluated"] = result.history.size();
    j["pareto_front_size"] = result.pareto_front.size();

    nlohmann::json front = nlohmann::json::array();
    for (const auto& ind : result.pareto_front) front.push_back(individual_to_json(ind));
    j["pareto_front"] = std::move(front);

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("GaOutput: cannot write " + path);
    f << j.dump(2) << '\n';
}

} // namespace pga
