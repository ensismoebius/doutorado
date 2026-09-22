#pragma once

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
//   1. Per-individual cache  — one full Meeting01GaIndividual per line, appended (and
//      flushed) the instant a genome is scored. Replayed on restart so no genome is
//      ever retrained.
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
auto individual_to_checkpoint_json(const Meeting01GaIndividual& ind) -> nlohmann::json;
auto individual_from_checkpoint_json(const nlohmann::json& j) -> Meeting01GaIndividual;

// ── Per-individual cache (layer 1) ───────────────────────────────────────────
void append_cache_entry(const std::string& cache_path, const Meeting01GaIndividual& ind);

// A single torn TRAILING line (crash mid-append) is dropped with a warning, not
// treated as corruption; any earlier parse failure throws.
auto load_cache_entries(const std::string& cache_path) -> std::vector<Meeting01GaIndividual>;

// ── Per-generation state (layer 2) ───────────────────────────────────────────
struct GenerationCheckpoint
{
    int generation = -1; // last COMPLETED generation
    std::string rng_state;
    std::vector<Meeting01GaIndividual> parents;
};

auto state_checkpoint_exists(const std::string& results_dir, const std::string& run_tag) -> bool;

void save_generation_checkpoint(const std::string& results_dir,
    const std::string& run_tag,
    int generation,
    const std::mt19937& rng,
    const std::vector<Meeting01GaIndividual>& parents);

auto load_generation_checkpoint(const std::string& results_dir, const std::string& run_tag)
    -> GenerationCheckpoint;

void remove_checkpoint_artifacts(const std::string& results_dir, const std::string& run_tag);

} // namespace meeting01::ga
