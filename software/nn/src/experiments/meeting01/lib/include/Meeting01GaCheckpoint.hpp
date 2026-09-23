#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Meeting01GaFitness.hpp"
#include "nlohmann/json.hpp"

namespace meeting01::ga
{

// Two-layer crash-resilient checkpointing, ported from paraconsistentGA's
// GaCheckpoint.hpp/.cpp (same two-layer shape; the Individual/Genome payload differs,
// so the code is ported rather than shared — see the plan's rationale for why only the
// three pure-NSGA-II functions moved to include/ga/Nsga2Core.hpp).
//
// Templated on `Ind` (generalized 2026-09-22 alongside the multi-family architecture
// search): the two layers below never touch a genome field by name — they call
// `genome_to_json(ind.genome)` / `genome_from_json(j, out)` unqualified and rely on
// ADL to find the right overload for whichever genome type `Ind::genome_type` is
// (Meeting01GaGenome.hpp for the SNN, Meeting01RecurrentGaGenome.hpp for LSTM/GRU,
// Meeting01TransformerGaGenome.hpp for the Transformer). One checkpoint
// implementation serves all four families instead of four copies.
//
//   1. Per-individual cache  — one full Ind per line, appended (and flushed) the
//      instant a genome is scored. Replayed on restart so no genome is ever retrained.
//   2. Per-generation state  — the surviving population + exact RNG state + generation
//      index, written atomically after each generation.
//
// Both are resume ARTIFACTS, removed on successful completion.

auto checkpoint_state_path(const std::string& results_dir, const std::string& run_tag)
    -> std::string;
auto checkpoint_cache_path(const std::string& results_dir, const std::string& run_tag)
    -> std::string;

// std::mt19937 state <-> string (standard stream serialization); round-trips exactly.
auto rng_to_string(const std::mt19937& rng) -> std::string;
void rng_from_string(std::mt19937& rng, const std::string& state);

// rank/crowding intentionally omitted — recomputed post-load (crowding can be +inf,
// not representable in JSON).
template <typename Ind>
auto individual_to_checkpoint_json(const Ind& ind) -> nlohmann::json
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

template <typename Ind>
auto individual_from_checkpoint_json(const nlohmann::json& j) -> Ind
{
    Ind ind;
    genome_from_json(j.at("genome"), ind.genome); // ADL, resolved by ind.genome's type
    ind.val_mse = j.at("val_mse").get<float>();
    ind.param_count = j.at("param_count").get<std::size_t>();
    ind.inference_cost = j.at("inference_cost").get<std::size_t>();
    ind.feasible = j.at("feasible").get<bool>();
    ind.constraint_violation = j.at("constraint_violation").get<double>();
    ind.objectives = j.at("objectives").get<std::vector<double>>();
    ind.born_generation = j.at("born_generation").get<int>();
    return ind;
}

// ── Per-individual cache (layer 1) ───────────────────────────────────────────
template <typename Ind>
void append_cache_entry(const std::string& cache_path, const Ind& ind)
{
    std::ofstream f(cache_path, std::ios::app);
    if (!f.is_open())
        throw std::runtime_error(
            "Meeting01GaCheckpoint: cannot append to cache file " + cache_path);
    f << individual_to_checkpoint_json(ind).dump() << '\n';
}

// A single torn TRAILING line (crash mid-append) is dropped with a warning, not
// treated as corruption; any earlier parse failure throws.
template <typename Ind>
auto load_cache_entries(const std::string& cache_path) -> std::vector<Ind>
{
    std::vector<Ind> out;
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
            out.push_back(individual_from_checkpoint_json<Ind>(nlohmann::json::parse(lines[i])));
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

// ── Per-generation state (layer 2) ───────────────────────────────────────────
template <typename Ind>
struct GenerationCheckpoint
{
    int generation = -1; // last COMPLETED generation
    std::string rng_state;
    std::vector<Ind> parents;
};

auto state_checkpoint_exists(const std::string& results_dir, const std::string& run_tag) -> bool;

template <typename Ind>
void save_generation_checkpoint(const std::string& results_dir,
    const std::string& run_tag,
    int generation,
    const std::mt19937& rng,
    const std::vector<Ind>& parents)
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

template <typename Ind>
auto load_generation_checkpoint(const std::string& results_dir, const std::string& run_tag)
    -> GenerationCheckpoint<Ind>
{
    const std::string path = checkpoint_state_path(results_dir, run_tag);
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Meeting01GaCheckpoint: cannot open checkpoint " + path);

    nlohmann::json j;
    f >> j;

    GenerationCheckpoint<Ind> ck;
    ck.generation = j.at("generation").get<int>();
    ck.rng_state = j.at("rng_state").get<std::string>();
    for (const auto& e : j.at("parents"))
        ck.parents.push_back(individual_from_checkpoint_json<Ind>(e));
    return ck;
}

void remove_checkpoint_artifacts(const std::string& results_dir, const std::string& run_tag);

} // namespace meeting01::ga
