#include "EegPairing.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

namespace thesis
{

auto parse_eeg_condition(const std::string& name) -> EegCondition
{
    if (name == "pronounced") return EegCondition::Pronounced;
    if (name == "imagined") return EegCondition::Imagined;
    throw std::invalid_argument(
        "EegPairing: EEG condition \"" + name + "\" is neither \"pronounced\" nor \"imagined\"");
}

namespace
{

// One stream per (subject, stimulus): the draw for one stimulus does not depend on how
// many trials another stimulus consumed, nor on the order subjects are loaded in.
auto stream_seed(std::uint32_t seed, int subject_id, int stimulus_id) -> std::uint32_t
{
    std::uint64_t h = 1469598103934665603ULL; // FNV-1a
    for (std::uint64_t x : {static_cast<std::uint64_t>(seed),
             static_cast<std::uint64_t>(static_cast<std::uint32_t>(subject_id)),
             static_cast<std::uint64_t>(static_cast<std::uint32_t>(stimulus_id))})
    {
        h ^= x;
        h *= 1099511628211ULL;
    }
    return static_cast<std::uint32_t>(h ^ (h >> 32));
}

auto where(int subject_id) -> std::string
{
    return "EegPairing: subject " + std::to_string(subject_id);
}

auto pair_pronounced(const std::vector<VoiceTrialMeta>& voice,
    const std::vector<EegTrialMeta>& eeg_trials,
    const ModalityIds& modality,
    int subject_id) -> std::vector<long long>
{
    std::map<long long, const EegTrialMeta*> by_row;
    for (const auto& t : eeg_trials)
        if (!by_row.emplace(t.original_row, &t).second)
            throw std::runtime_error(where(subject_id) + " has two EEG trials at original_row " +
                                     std::to_string(t.original_row));

    std::vector<long long> paired;
    paired.reserve(voice.size());
    for (const auto& v : voice)
    {
        const auto it = by_row.find(v.original_row);
        if (it == by_row.end())
            throw std::runtime_error(where(subject_id) + ": the voice trial at original_row " +
                                     std::to_string(v.original_row) + " has no EEG");
        const EegTrialMeta& e = *it->second;
        if (e.modality_id != modality.pronounced || e.stimulus_id != v.stimulus_id)
            throw std::runtime_error(where(subject_id) + ": the EEG at original_row " +
                                     std::to_string(v.original_row) + " is modality " +
                                     std::to_string(e.modality_id) + ", stimulus " +
                                     std::to_string(e.stimulus_id) + "; its voice trial is " +
                                     "Pronounced, stimulus " + std::to_string(v.stimulus_id));
        paired.push_back(e.original_row);
    }
    return paired;
}

auto pair_imagined(const std::vector<VoiceTrialMeta>& voice,
    const std::vector<EegTrialMeta>& eeg_trials,
    const ModalityIds& modality,
    std::uint32_t seed,
    int subject_id) -> std::vector<long long>
{
    std::map<int, std::vector<long long>> pool; // stimulus -> imagined original_rows
    for (const auto& t : eeg_trials)
        if (t.modality_id == modality.imagined) pool[t.stimulus_id].push_back(t.original_row);

    std::map<int, std::size_t> needed;
    for (const auto& v : voice) ++needed[v.stimulus_id];
    for (const auto& [stimulus, n] : needed)
    {
        auto& rows = pool[stimulus];
        if (rows.size() < n)
            throw std::runtime_error(where(subject_id) + ", stimulus " + std::to_string(stimulus) +
                                     ": " + std::to_string(n) + " voice trials but only " +
                                     std::to_string(rows.size()) + " imagined EEG trials");
        std::sort(rows.begin(), rows.end());
        std::mt19937 rng(stream_seed(seed, subject_id, stimulus));
        std::shuffle(rows.begin(), rows.end(), rng);
    }

    std::map<int, std::size_t> next;
    std::vector<long long> paired;
    paired.reserve(voice.size());
    for (const auto& v : voice) paired.push_back(pool[v.stimulus_id][next[v.stimulus_id]++]);
    return paired;
}

} // namespace

auto pair_eeg_trials(const std::vector<VoiceTrialMeta>& voice,
    const std::vector<EegTrialMeta>& eeg_trials,
    const ModalityIds& modality,
    EegCondition condition,
    std::uint32_t seed,
    int subject_id) -> std::vector<long long>
{
    if (modality.pronounced == modality.imagined)
        throw std::invalid_argument(
            "EegPairing: the Pronounced and Imagined modality ids coincide");
    return condition == EegCondition::Pronounced
               ? pair_pronounced(voice, eeg_trials, modality, subject_id)
               : pair_imagined(voice, eeg_trials, modality, seed, subject_id);
}

} // namespace thesis
