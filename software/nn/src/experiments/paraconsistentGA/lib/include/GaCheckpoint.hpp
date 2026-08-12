#pragma once

#include <random>
#include <string>
#include <vector>

#include "GaFitness.hpp"
#include "nlohmann/json.hpp"

namespace pga
{

// ── Crash-resilient checkpointing ────────────────────────────────────────────
//
// Two layers persist a run so a power loss costs at most one in-flight training
// (.wiki/Experiments/ParaconsistentGA-Design.md §6.1):
//
//   1. Per-individual cache  — `pga_<tag>_cache.jsonl`, one full Individual per line,
//      appended (and flushed) the instant a genome is scored. On restart it is replayed
//      into the eval cache, so no genome is ever retrained.
//   2. Per-generation state  — `pga_<tag>_checkpoint.json`, the surviving population +
//      the exact RNG state + the generation index, written atomically after each
//      generation. On restart the loop resumes from the next generation.
//
// Both files are resume ARTIFACTS, not results: they are removed on successful
// completion, leaving only the CSV + Pareto JSON.

std::string checkpoint_state_path(const std::string& results_dir, const std::string& run_tag);
std::string checkpoint_cache_path(const std::string& results_dir, const std::string& run_tag);

// std::mt19937 state <-> string (its standard stream serialization). Round-trips exactly,
// so a restored engine produces the identical subsequent draw sequence.
std::string rng_to_string(const std::mt19937& rng);
void rng_from_string(std::mt19937& rng, const std::string& state);

// Full Individual <-> JSON: EVERY field needed to resume, including the diploid genotype
// (parents need it to reproduce). rank/crowding are intentionally omitted — they are
// recomputed from objectives after a restore, and crowding can be +inf (not representable
// in JSON).
nlohmann::json individual_to_checkpoint_json(const Individual& ind);
Individual individual_from_checkpoint_json(const nlohmann::json& j);

// ── Per-individual cache (layer 1) ───────────────────────────────────────────

// Append one scored individual as a JSONL line and flush. Cheap; called once per
// distinct genome the moment it is trained.
void append_cache_entry(const std::string& cache_path, const Individual& ind);

// Replay a cache file into a vector of individuals (discovery order preserved). A single
// torn TRAILING line (a crash mid-append) is dropped with a warning — that genome is
// simply retrained. Any earlier parse failure is real corruption and throws.
std::vector<Individual> load_cache_entries(const std::string& cache_path);

// ── Per-generation state (layer 2) ───────────────────────────────────────────

struct GenerationCheckpoint
{
    int generation = -1;             // last COMPLETED generation
    std::string rng_state;           // engine state at the end of that generation
    std::vector<Individual> parents; // the population that survived it
};

bool state_checkpoint_exists(const std::string& results_dir, const std::string& run_tag);

// Atomic write (temp file + rename) so a crash mid-write never corrupts the checkpoint.
void save_generation_checkpoint(const std::string& results_dir,
    const std::string& run_tag,
    int generation,
    const std::mt19937& rng,
    const std::vector<Individual>& parents);

GenerationCheckpoint load_generation_checkpoint(
    const std::string& results_dir, const std::string& run_tag);

// Remove both resume artifacts (state + cache). Called on successful completion.
void remove_checkpoint_artifacts(const std::string& results_dir, const std::string& run_tag);

} // namespace pga
