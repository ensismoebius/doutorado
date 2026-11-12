/**
 * @file src/experiments/thesis/lib/include/EegPairing.hpp
 * @brief Which EEG trial goes with each voice trial: the same trial, or an imagined one.
 *
 * Why this exists. The dataset (10.1117/12.2255697) records the voice only in the
 * "Pronounced" trials: 1974 of them. EEG exists for those AND for 7341 "Imagined" trials,
 * in which the speaker articulates the stimulus silently. A thesis sample is one voice
 * trial plus one EEG trial, and the EEG condition decides which:
 *
 *   condition    EEG trial paired with voice trial v (subject s, stimulus x)
 *   ----------   ------------------------------------------------------------------
 *   pronounced   v itself: the EEG recorded while the voice was being produced
 *   imagined     an Imagined trial of the same (s, x), drawn without replacement
 *
 * The voice and the imagined EEG are two separate acts by design: an authentication
 * attempt is "say the phrase, then imagine it". Drawing without replacement keeps every
 * EEG trial in at most one sample, so no two samples share a recording.
 *
 * The defect this replaces (found 2026-11-12). The loader read the EEG at
 * `original_row - 1`, a 1-based convention neither source uses: every voice sample got the
 * PREVIOUS trial of the same speaker (1833 pronounced, 141 imagined, the stimulus matching
 * in 193 of 1974, i.e. chance). Speaker identity survived, so the results looked plausible;
 * the phrase and the modality did not.
 *
 * Why identity is right (not merely another guess). The SQLite stores the audio on a trial
 * row, and the trials that carry audio are exactly the 1974 Pronounced ones and none of the
 * 7341 Imagined: a shift by one row would put ~141 voices on imagined trials, as the old
 * reading did. The stimulus check below is NOT that evidence -- in the SQLite one trial row
 * holds one stimulus, so the voice's and its own EEG's agree by construction; it guards the
 * code (a reader that fetched another row would fail it), and the modality check is the one
 * that tells the hypotheses apart.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace thesis
{

enum class EegCondition
{
    Pronounced,
    Imagined,
};

/// "pronounced" | "imagined"; anything else throws std::invalid_argument.
auto parse_eeg_condition(const std::string& name) -> EegCondition;

/// One trial that has EEG, as the trial table describes it.
struct EegTrialMeta
{
    long long original_row = 0; ///< the trial's row in the subject's original EEG matrix
    int modality_id = 0;
    int stimulus_id = 0;
};

/// One voice trial: its own original_row (the EEG of the same trial) and its stimulus.
struct VoiceTrialMeta
{
    long long original_row = 0;
    int stimulus_id = 0;
};

/// The modality ids of the dataset's `modalities` table.
struct ModalityIds
{
    int pronounced = 0;
    int imagined = 0;
};

/**
 * For each voice trial (in order), the original_row of the EEG trial it is paired with.
 *
 * Pronounced: the EEG trial with the voice trial's own original_row, which must exist, be
 * Pronounced and carry the same stimulus. Imagined: per stimulus, the subject's Imagined
 * trials sorted by original_row, shuffled with mt19937(seed mixed with subject and
 * stimulus), and handed out in voice-trial order. Fewer imagined than voice trials for a
 * stimulus throws: a reused recording is not a new sample.
 */
auto pair_eeg_trials(const std::vector<VoiceTrialMeta>& voice,
    const std::vector<EegTrialMeta>& eeg_trials,
    const ModalityIds& modality,
    EegCondition condition,
    std::uint32_t seed,
    int subject_id) -> std::vector<long long>;

} // namespace thesis
