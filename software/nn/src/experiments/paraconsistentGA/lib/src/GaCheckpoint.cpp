#include "GaCheckpoint.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "ThesisOutput.hpp" // thesis::ensure_dir (reused)

namespace pga
{

namespace
{
nlohmann::json genome_to_json(const Genome& g)
{
    return {{"encoder_widths", g.encoder_widths},
        {"encoding", g.encoding},
        {"time_steps", g.time_steps},
        {"voltage_threshold", g.voltage_threshold}};
}

Genome genome_from_json(const nlohmann::json& j)
{
    Genome g;
    g.encoder_widths = j.at("encoder_widths").get<std::vector<int>>();
    g.encoding = j.at("encoding").get<std::string>();
    g.time_steps = j.at("time_steps").get<int>();
    g.voltage_threshold = j.at("voltage_threshold").get<float>();
    return g;
}

nlohmann::json diploid_to_json(const DiploidGenome& d)
{
    return {{"hap_a", genome_to_json(d.hap_a)},
        {"hap_b", genome_to_json(d.hap_b)},
        {"dom_a", d.dom_a},
        {"dom_b", d.dom_b}};
}

DiploidGenome diploid_from_json(const nlohmann::json& j)
{
    DiploidGenome d;
    d.hap_a = genome_from_json(j.at("hap_a"));
    d.hap_b = genome_from_json(j.at("hap_b"));
    d.dom_a = j.at("dom_a").get<float>();
    d.dom_b = j.at("dom_b").get<float>();
    return d;
}
} // namespace

std::string checkpoint_state_path(const std::string& results_dir, const std::string& run_tag)
{
    return results_dir + "/pga_" + run_tag + "_checkpoint.json";
}

std::string checkpoint_cache_path(const std::string& results_dir, const std::string& run_tag)
{
    return results_dir + "/pga_" + run_tag + "_cache.jsonl";
}

std::string rng_to_string(const std::mt19937& rng)
{
    std::ostringstream os;
    os << rng;
    return os.str();
}

void rng_from_string(std::mt19937& rng, const std::string& state)
{
    std::istringstream is(state);
    is >> rng;
    if (is.fail())
        throw std::runtime_error(
            "GaCheckpoint: corrupt RNG state in checkpoint — cannot resume deterministically. "
            "Delete the checkpoint file to restart this profile from generation 0.");
}

nlohmann::json individual_to_checkpoint_json(const Individual& ind)
{
    // rank/crowding are omitted on purpose (recomputed from objectives after load; crowding
    // may be +inf, which JSON cannot hold).
    return {{"genotype", diploid_to_json(ind.genotype)},
        {"genome", genome_to_json(ind.genome)},
        {"d_penalized_mean", ind.d_penalized_mean},
        {"d_penalized_std", ind.d_penalized_std},
        {"alpha", ind.alpha},
        {"beta", ind.beta},
        {"g1", ind.g1},
        {"g2", ind.g2},
        {"d_truth", ind.d_truth},
        {"winning_seed_offset", ind.winning_seed_offset},
        {"latent_activity", ind.latent_activity},
        {"param_count", ind.param_count},
        {"inference_cost", ind.inference_cost},
        {"est_latency_ms", ind.est_latency_ms},
        {"evaluated", ind.evaluated},
        {"feasible", ind.feasible},
        {"constraint_violation", ind.constraint_violation},
        {"objectives", ind.objectives},
        {"born_generation", ind.born_generation}};
}

Individual individual_from_checkpoint_json(const nlohmann::json& j)
{
    Individual ind;
    ind.genotype = diploid_from_json(j.at("genotype"));
    ind.genome = genome_from_json(j.at("genome"));
    ind.d_penalized_mean = j.at("d_penalized_mean").get<double>();
    ind.d_penalized_std = j.at("d_penalized_std").get<double>();
    ind.alpha = j.at("alpha").get<double>();
    ind.beta = j.at("beta").get<double>();
    ind.g1 = j.at("g1").get<double>();
    ind.g2 = j.at("g2").get<double>();
    ind.d_truth = j.at("d_truth").get<double>();
    ind.winning_seed_offset = j.at("winning_seed_offset").get<int>();
    ind.latent_activity = j.at("latent_activity").get<double>();
    ind.param_count = j.at("param_count").get<long>();
    ind.inference_cost = j.at("inference_cost").get<long>();
    ind.est_latency_ms = j.at("est_latency_ms").get<double>();
    ind.evaluated = j.at("evaluated").get<bool>();
    ind.feasible = j.at("feasible").get<bool>();
    ind.constraint_violation = j.at("constraint_violation").get<double>();
    ind.objectives = j.at("objectives").get<std::vector<double>>();
    ind.born_generation = j.at("born_generation").get<int>();
    return ind;
}

void append_cache_entry(const std::string& cache_path, const Individual& ind)
{
    // Open-append-close per line so the OS flushes each record; a crash can only ever tear
    // the very last line, which load_cache_entries tolerates.
    std::ofstream f(cache_path, std::ios::app);
    if (!f.is_open())
        throw std::runtime_error("GaCheckpoint: cannot append to cache file " + cache_path);
    f << individual_to_checkpoint_json(ind).dump() << '\n';
}

std::vector<Individual> load_cache_entries(const std::string& cache_path)
{
    std::vector<Individual> out;
    std::ifstream f(cache_path);
    if (!f.is_open()) return out; // no cache yet — fresh run

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) lines.push_back(line);

    for (size_t i = 0; i < lines.size(); ++i)
    {
        try
        {
            out.push_back(individual_from_checkpoint_json(nlohmann::json::parse(lines[i])));
        }
        catch (const std::exception& e)
        {
            if (i + 1 == lines.size())
            {
                // A torn trailing line from a crash mid-append: drop it (that one genome is
                // simply retrained). Announced, not silent.
                std::cerr << "[PGA] checkpoint: dropping incomplete trailing cache line ("
                          << e.what() << ") — its genome will be retrained.\n";
                break;
            }
            throw std::runtime_error("GaCheckpoint: corrupt cache line " + std::to_string(i + 1) +
                                     " in " + cache_path + ": " + e.what());
        }
    }
    return out;
}

bool state_checkpoint_exists(const std::string& results_dir, const std::string& run_tag)
{
    return std::filesystem::exists(checkpoint_state_path(results_dir, run_tag));
}

void save_generation_checkpoint(const std::string& results_dir,
    const std::string& run_tag,
    int generation,
    const std::mt19937& rng,
    const std::vector<Individual>& parents)
{
    thesis::ensure_dir(results_dir);

    nlohmann::json j;
    j["run_tag"] = run_tag;
    j["generation"] = generation;
    j["rng_state"] = rng_to_string(rng);
    nlohmann::json pop = nlohmann::json::array();
    for (const auto& ind : parents) pop.push_back(individual_to_checkpoint_json(ind));
    j["parents"] = std::move(pop);

    // Atomic: write a temp file in the same directory, then rename over the target. rename
    // is atomic on POSIX, so a reader never sees a half-written checkpoint.
    const std::string path = checkpoint_state_path(results_dir, run_tag);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f.is_open())
            throw std::runtime_error("GaCheckpoint: cannot write checkpoint temp " + tmp);
        f << j.dump() << '\n';
    }
    std::filesystem::rename(tmp, path);
}

GenerationCheckpoint load_generation_checkpoint(
    const std::string& results_dir, const std::string& run_tag)
{
    const std::string path = checkpoint_state_path(results_dir, run_tag);
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("GaCheckpoint: cannot open checkpoint " + path);

    nlohmann::json j;
    f >> j;

    GenerationCheckpoint ck;
    ck.generation = j.at("generation").get<int>();
    ck.rng_state = j.at("rng_state").get<std::string>();
    for (const auto& e : j.at("parents")) ck.parents.push_back(individual_from_checkpoint_json(e));
    return ck;
}

void remove_checkpoint_artifacts(const std::string& results_dir, const std::string& run_tag)
{
    std::error_code ec; // best-effort cleanup; missing files are fine
    std::filesystem::remove(checkpoint_state_path(results_dir, run_tag), ec);
    std::filesystem::remove(checkpoint_cache_path(results_dir, run_tag), ec);
}

} // namespace pga
