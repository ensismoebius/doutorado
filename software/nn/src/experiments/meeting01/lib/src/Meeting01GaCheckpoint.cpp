#include "../include/Meeting01GaCheckpoint.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace meeting01::ga
{

namespace
{
nlohmann::json genome_to_json(const Genome& g)
{
    return {{"encoder_widths", g.encoder_widths},
        {"encoding", g.encoding},
        {"architecture", g.architecture},
        {"voltage_threshold", g.voltage_threshold},
        {"alpha", g.alpha}};
}

Genome genome_from_json(const nlohmann::json& j)
{
    Genome g;
    g.encoder_widths = j.at("encoder_widths").get<std::vector<int>>();
    g.encoding = j.at("encoding").get<std::string>();
    g.architecture = j.at("architecture").get<std::string>();
    g.voltage_threshold = j.at("voltage_threshold").get<float>();
    g.alpha = j.at("alpha").get<float>();
    return g;
}
} // namespace

auto checkpoint_state_path(const std::string& results_dir, const std::string& run_tag)
    -> std::string
{
    return results_dir + "/meeting01_ga_" + run_tag + "_checkpoint.json";
}

auto checkpoint_cache_path(const std::string& results_dir, const std::string& run_tag)
    -> std::string
{
    return results_dir + "/meeting01_ga_" + run_tag + "_cache.jsonl";
}

auto rng_to_string(const std::mt19937& rng) -> std::string
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
            "Meeting01GaCheckpoint: corrupt RNG state in checkpoint — cannot resume "
            "deterministically. Delete the checkpoint file to restart this run from generation 0.");
}

auto individual_to_checkpoint_json(const Meeting01GaIndividual& ind) -> nlohmann::json
{
    return {{"genome", genome_to_json(ind.genome)},
        {"val_mse", ind.val_mse},
        {"param_count", ind.param_count},
        {"inference_cost", ind.inference_cost},
        {"feasible", ind.feasible},
        {"constraint_violation", ind.constraint_violation},
        {"objectives", ind.objectives},
        {"born_generation", ind.born_generation}};
}

auto individual_from_checkpoint_json(const nlohmann::json& j) -> Meeting01GaIndividual
{
    Meeting01GaIndividual ind;
    ind.genome = genome_from_json(j.at("genome"));
    ind.val_mse = j.at("val_mse").get<float>();
    ind.param_count = j.at("param_count").get<std::size_t>();
    ind.inference_cost = j.at("inference_cost").get<std::size_t>();
    ind.feasible = j.at("feasible").get<bool>();
    ind.constraint_violation = j.at("constraint_violation").get<double>();
    ind.objectives = j.at("objectives").get<std::vector<double>>();
    ind.born_generation = j.at("born_generation").get<int>();
    return ind;
}

void append_cache_entry(const std::string& cache_path, const Meeting01GaIndividual& ind)
{
    std::ofstream f(cache_path, std::ios::app);
    if (!f.is_open())
        throw std::runtime_error(
            "Meeting01GaCheckpoint: cannot append to cache file " + cache_path);
    f << individual_to_checkpoint_json(ind).dump() << '\n';
}

auto load_cache_entries(const std::string& cache_path) -> std::vector<Meeting01GaIndividual>
{
    std::vector<Meeting01GaIndividual> out;
    std::ifstream f(cache_path);
    if (!f.is_open()) return out; // no cache yet — fresh run

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) lines.push_back(line);

    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        try
        {
            out.push_back(individual_from_checkpoint_json(nlohmann::json::parse(lines[i])));
        }
        catch (const std::exception& e)
        {
            if (i + 1 == lines.size())
            {
                std::cerr << "[Meeting01Ga] checkpoint: dropping incomplete trailing cache line ("
                          << e.what() << ") — its genome will be retrained.\n";
                break;
            }
            throw std::runtime_error("Meeting01GaCheckpoint: corrupt cache line " +
                                     std::to_string(i + 1) + " in " + cache_path + ": " + e.what());
        }
    }
    return out;
}

auto state_checkpoint_exists(const std::string& results_dir, const std::string& run_tag) -> bool
{
    return std::filesystem::exists(checkpoint_state_path(results_dir, run_tag));
}

void save_generation_checkpoint(const std::string& results_dir,
    const std::string& run_tag,
    int generation,
    const std::mt19937& rng,
    const std::vector<Meeting01GaIndividual>& parents)
{
    std::filesystem::create_directories(results_dir);

    nlohmann::json j;
    j["run_tag"] = run_tag;
    j["generation"] = generation;
    j["rng_state"] = rng_to_string(rng);
    nlohmann::json pop = nlohmann::json::array();
    for (const auto& ind : parents) pop.push_back(individual_to_checkpoint_json(ind));
    j["parents"] = std::move(pop);

    const std::string path = checkpoint_state_path(results_dir, run_tag);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f.is_open())
            throw std::runtime_error("Meeting01GaCheckpoint: cannot write checkpoint temp " + tmp);
        f << j.dump() << '\n';
    }
    std::filesystem::rename(tmp, path);
}

auto load_generation_checkpoint(const std::string& results_dir, const std::string& run_tag)
    -> GenerationCheckpoint
{
    const std::string path = checkpoint_state_path(results_dir, run_tag);
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Meeting01GaCheckpoint: cannot open checkpoint " + path);

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

} // namespace meeting01::ga
