/**
 * @file src/experiments/thesis/lib/include/FeatureDiskCache.hpp
 * @brief Handcrafted candidate features on disk, shared by every process of the per-fold
 *        protocol (.wiki/Experiments/Thesis-Per-Fold-Protocol.md, "Feature cache").
 *
 * The problem. Each per-fold unit (one profile, one repetition) ranks every Phase 00 candidate
 * of its signal; for the voice that is 138 handcrafted extractions over 1974 utterances, about
 * 3.4 h of one CPU (GridUnesp, 2026-11-18, job 5330057), before any fold. The result is the
 * same in every unit that reads that signal of that base: a handcrafted vector depends on its
 * own sample only, not on the seed, the repetition, the classifier or the fold. Three hundred
 * units would compute the same 138 matrices three hundred times.
 *
 * The rule. A cached matrix is reused only when everything it was computed from is the same:
 *
 *   key = revision of the deployed code        (THESIS_FEATURE_CACHE_REVISION)
 *       + fingerprint of the loaded samples    (every audio and EEG value, labels, rates)
 *       + the signal (voice | eeg | fused)
 *       + the candidate's whole feature_extraction block
 *
 * The file is named by a hash of the key and holds the key itself; a file whose stored key
 * differs from the expected one is refused, never read. Missing file: in "write" mode the
 * matrix is computed and written (atomically: temp + rename, so concurrent writers of the
 * same identical matrix cannot tear it); in "read" mode it is an error, because the arrays
 * of a phase are meant to find every matrix the warm step wrote, and a miss there would
 * silently cost hours (thesis --warm-feature-cache, thesis_warm_feature_cache.sbatch).
 *
 * Only the ranked signal's handcrafted candidates go through it. PCA and the autoencoders are
 * fitted on each fold's training rows (nothing to share), and the winner's fused or degraded
 * features are a few extractions per fold, kept in memory.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "ThesisDataset.hpp"
#include "ThesisFeatureExtraction.hpp"

namespace thesis
{

/// FNV-1a 64 over everything a feature can depend on in `view`: sample count, rates, and per
/// sample its subject, stimulus, phrase, and every audio and EEG value (as stored).
auto view_fingerprint(const ThesisDatasetView& view) -> std::uint64_t;

class FeatureDiskCache
{
   public:
    enum class Mode
    {
        Write, ///< a miss computes and writes (the warm step)
        Read   ///< a miss throws (the arrays, after the warm step)
    };

    /// `dir` is created if missing; `revision` must be non-empty.
    FeatureDiskCache(std::string dir, std::string revision, Mode mode);

    /// THESIS_FEATURE_CACHE (directory), THESIS_FEATURE_CACHE_REVISION (the deploy's label)
    /// and THESIS_FEATURE_CACHE_MODE ("read" | "write"). nullopt when THESIS_FEATURE_CACHE is
    /// unset or empty; a directory without a revision or a valid mode throws.
    static auto from_env() -> std::optional<FeatureDiskCache>;

    /// The matrix of `key` (the parts above, minus the revision, which is added here): read,
    /// or computed by `compute` and written (Write mode). `n_rows` is the view's sample count;
    /// a cached matrix of another height throws.
    auto load_or_compute(const std::string& key,
        std::size_t n_rows,
        const std::function<FeatureSet()>& compute) const -> FeatureSet;

    [[nodiscard]] auto mode() const -> Mode
    {
        return mode_;
    }
    [[nodiscard]] auto path_for(const std::string& key) const -> std::string;

   private:
    std::string dir_;
    std::string revision_;
    Mode mode_;
};

} // namespace thesis
