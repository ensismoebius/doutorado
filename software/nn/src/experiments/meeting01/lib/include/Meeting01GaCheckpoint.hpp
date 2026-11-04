#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Meeting01GaFitness.hpp"
#include "Meeting01Output.hpp"
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
//
// Both carry a STAMP (config_hash, results_format) and loading refuses any other stamp:
// a validation MSE scored under one profile, binary or set of windows must not be ranked against
// genomes scored under another. The stamp's hash is checkpoint_hash (Meeting01Dataset.hpp), which
// also covers the split's window values, so a resume across a data-pipeline change is refused
// whether or not that change bumped results_format. Until 2026-11-02 they had none, so RESUME=1
// after a profile edit (epochs, window caps, ...) silently continued the search with the old
// scores, while the fold's manifest recorded the new configuration.
inline auto checkpoint_stamp(std::size_t config_hash) -> nlohmann::json
{
    return {{"config_hash", config_hash}, {"results_format", meeting01::kResultsFormat}};
}

inline void check_checkpoint_stamp(
    const nlohmann::json& j, std::size_t config_hash, const std::string& where)
{
    if (!j.contains("config_hash") || !j.contains("results_format") ||
        j.at("config_hash").get<std::size_t>() != config_hash ||
        j.at("results_format").get<int>() != meeting01::kResultsFormat)
        throw std::runtime_error(
            "Meeting01GaCheckpoint: " + where + " was written under " +
            (j.contains("config_hash")
                    ? "config_hash " + j.at("config_hash").dump() + " / results_format " +
                          j.value("results_format", nlohmann::json()).dump()
                    : std::string("no stamp (a binary before 2026-11-02)")) +
            ", this run is config_hash " + std::to_string(config_hash) + " / results_format " +
            std::to_string(meeting01::kResultsFormat) +
            ": the profile, the binary or the windows changed since this search was interrupted, "
            "and "
            "resuming "
            "would rank genomes scored under both. Restore what wrote it, "
            "or delete this search's meeting01_ga_<tag>_checkpoint.json and _cache.jsonl to "
            "restart it.");
}

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
void append_cache_entry(const std::string& cache_path, const Ind& ind, std::size_t config_hash)
{
    std::ofstream f(cache_path, std::ios::app);
    if (!f.is_open())
        throw std::runtime_error(
            "Meeting01GaCheckpoint: cannot append to cache file " + cache_path);
    nlohmann::json j = individual_to_checkpoint_json(ind);
    j.update(checkpoint_stamp(config_hash));
    f << j.dump() << '\n';
}

// A single torn TRAILING line (crash mid-append) is dropped with a warning, not
// treated as corruption, and removed from the file: it has no '\n', so the resumed
// search's next append would be glued onto it, and that glued line, no longer the last
// once another entry follows, would make the NEXT resume refuse the whole cache.
// Only an unparseable line can be torn (a cut JSON object lacks its closing brace); a
// line that parses is checked like any other: another stamp or a missing field throws.
template <typename Ind>
auto load_cache_entries(const std::string& cache_path, std::size_t config_hash) -> std::vector<Ind>
{
    std::vector<Ind> out;
    std::ifstream f(cache_path);
    if (!f.is_open()) return out; // no cache yet — fresh run

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) lines.push_back(line);
    f.close();

    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        const std::string where = cache_path + " line " + std::to_string(i + 1);
        nlohmann::json j;
        try
        {
            j = nlohmann::json::parse(lines[i]);
        }
        catch (const nlohmann::json::parse_error& e)
        {
            if (i + 1 != lines.size())
                throw std::runtime_error(
                    "Meeting01GaCheckpoint: corrupt cache " + where + ": " + e.what());
            std::cerr << "[Meeting01Ga] checkpoint: dropping incomplete trailing cache line ("
                      << e.what() << ") — its genome will be retrained.\n";
            const std::string tmp = cache_path + ".tmp";
            {
                std::ofstream g(tmp, std::ios::trunc);
                for (std::size_t k = 0; k < i; ++k) g << lines[k] << '\n';
                if (!g)
                    throw std::runtime_error("Meeting01GaCheckpoint: cannot rewrite " + cache_path +
                                             " without its torn last line (writing " + tmp +
                                             " failed).");
            }
            std::filesystem::rename(tmp, cache_path);
            break;
        }
        check_checkpoint_stamp(j, config_hash, where);
        try
        {
            out.push_back(individual_from_checkpoint_json<Ind>(j));
        }
        catch (const nlohmann::json::exception& e)
        {
            throw std::runtime_error("Meeting01GaCheckpoint: cache " + where +
                                     " parses but is not an individual of this binary (" +
                                     e.what() +
                                     "). Delete this search's _checkpoint.json and "
                                     "_cache.jsonl to restart it.");
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
    const std::vector<Ind>& parents,
    std::size_t config_hash)
{
    std::filesystem::create_directories(results_dir);

    nlohmann::json j = checkpoint_stamp(config_hash);
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
auto load_generation_checkpoint(
    const std::string& results_dir, const std::string& run_tag, std::size_t config_hash)
    -> GenerationCheckpoint<Ind>
{
    const std::string path = checkpoint_state_path(results_dir, run_tag);
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Meeting01GaCheckpoint: cannot open checkpoint " + path);

    nlohmann::json j;
    f >> j;
    check_checkpoint_stamp(j, config_hash, path);

    GenerationCheckpoint<Ind> ck;
    ck.generation = j.at("generation").get<int>();
    ck.rng_state = j.at("rng_state").get<std::string>();
    for (const auto& e : j.at("parents"))
        ck.parents.push_back(individual_from_checkpoint_json<Ind>(e));
    return ck;
}

void remove_checkpoint_artifacts(const std::string& results_dir, const std::string& run_tag);

} // namespace meeting01::ga
