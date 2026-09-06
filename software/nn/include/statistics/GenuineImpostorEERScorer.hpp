/**
 * @file GenuineImpostorEERScorer.hpp
 * @brief GenuineImpostorEERScorer strategy (extracted from eer_scorer.hpp).
 */

#ifndef NN_STATISTICS_GENUINE_IMPOSTOR_EER_SCORER_HPP
#define NN_STATISTICS_GENUINE_IMPOSTOR_EER_SCORER_HPP

#include <cstddef>
#include <vector>

#include "statistics/eer_scorer.hpp"

namespace statistics
{

/**
 * @brief SOTA EER via genuine/impostor cosine-similarity trial sweeping.
 *
 * Protocol (matches NIST SRE / ASVspoof convention):
 * 1. For each speaker in the test fold, take the first n_enroll_per_speaker
 *    samples as the enrollment set; the remainder become probes.
 *    Speakers with fewer than n_enroll_per_speaker + 1 samples are skipped.
 * 2. Enrollment template = L2-normalised mean of enrollment embedding vectors.
 * 3. For each probe, compute cosine similarity vs. every speaker template:
 *    - Probe vs. own speaker → genuine trial.
 *    - Probe vs. other speaker → impostor trial.
 * 4. Sort all trials by score; sweep threshold (descending) to build FAR/FRR
 *    curve; linearly interpolate the crossing point → EER.
 *
 * Returns NaN when no valid trials exist (e.g. every speaker has too few samples).
 *
 * @param n_enroll_per_speaker Utterances used per speaker as the enrollment
 *        template (default 1).  Increase for more stable templates.
 */
class GenuineImpostorEERScorer : public IEERScorer
{
   public:
    explicit GenuineImpostorEERScorer(std::size_t n_enroll_per_speaker = 1U);

    [[nodiscard]] auto compute_eer(const std::vector<std::vector<float>>& embeddings,
        const std::vector<int>& labels,
        int n_classes) const -> double override;

    /**
     * @brief AUC via Wilcoxon-Mann-Whitney: P(genuine_score > impostor_score).
     *
     * Ties contribute 0.5.  Returns NaN when fewer than 2 speakers have valid
     * templates (same degeneracy condition as compute_eer).
     */
    [[nodiscard]] auto compute_auc(const std::vector<std::vector<float>>& embeddings,
        const std::vector<int>& labels,
        int n_classes) const -> double override;

   private:
    std::size_t n_enroll_;
};

} // namespace statistics

#endif // NN_STATISTICS_GENUINE_IMPOSTOR_EER_SCORER_HPP
