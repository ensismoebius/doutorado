#include "../include/Meeting01GaCheckpoint.hpp"

#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace meeting01::ga
{

// The genome-specific serialization helpers (genome_to_json/genome_from_json) that
// used to live here as private functions moved to each genome's own header/.cpp
// (Meeting01GaGenome.*, Meeting01RecurrentGaGenome.*, Meeting01TransformerGaGenome.*)
// when the checkpoint layer above was generalized to serve all four families — see
// Meeting01GaCheckpoint.hpp's header comment. Everything below is genome-agnostic.

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

auto state_checkpoint_exists(const std::string& results_dir, const std::string& run_tag) -> bool
{
    return std::filesystem::exists(checkpoint_state_path(results_dir, run_tag));
}

void remove_checkpoint_artifacts(const std::string& results_dir, const std::string& run_tag)
{
    std::error_code ec; // best-effort cleanup; missing files are fine
    std::filesystem::remove(checkpoint_state_path(results_dir, run_tag), ec);
    std::filesystem::remove(checkpoint_cache_path(results_dir, run_tag), ec);
}

} // namespace meeting01::ga
