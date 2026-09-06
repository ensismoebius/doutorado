/**
 * @file ClassificationEERScorer.hpp
 * @brief ClassificationEERScorer strategy (extracted from eer_scorer.hpp).
 */

#ifndef NN_STATISTICS_CLASSIFICATION_EER_SCORER_HPP
#define NN_STATISTICS_CLASSIFICATION_EER_SCORER_HPP

#include <vector>

#include "statistics/eer_scorer.hpp"

namespace statistics
{

/**
 * @brief Speaker-verification EER scorer (audit m-4: upgraded).
 *
 * Historically this derived EER from closed-set confusion matrices (a
 * non-standard, inferior protocol). It now delegates to the genuine/impostor
 * cosine-similarity method (same as GenuineImpostorEERScorer with one
 * enrollment utterance), so EER and AUC are computed from the score
 * distribution. The class name is retained for backward compatibility.
 */
class ClassificationEERScorer : public IEERScorer
{
   public:
    [[nodiscard]] auto compute_eer(const std::vector<std::vector<float>>& embeddings,
        const std::vector<int>& labels,
        int n_classes) const -> double override;

    [[nodiscard]] auto compute_auc(const std::vector<std::vector<float>>& embeddings,
        const std::vector<int>& labels,
        int n_classes) const -> double override;
};

} // namespace statistics

#endif // NN_STATISTICS_CLASSIFICATION_EER_SCORER_HPP
