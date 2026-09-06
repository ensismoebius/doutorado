/**
 * @file eer_scorer.hpp
 * @brief Pluggable EER computation strategies for speaker verification evaluation.
 *
 * Two concrete strategies are provided (each in its own file):
 *  - ClassificationEERScorer  : legacy — derives EER from closed-set confusion matrices.
 *    See ClassificationEERScorer.hpp.
 *  - GenuineImpostorEERScorer : SOTA — enrollment/probe scoring with genuine/impostor
 *    cosine-similarity trials and threshold sweep. See GenuineImpostorEERScorer.hpp.
 *
 * Pass any IEERScorer to run_classifier() to select the evaluation protocol.
 */

#ifndef NN_STATISTICS_EER_SCORER_HPP
#define NN_STATISTICS_EER_SCORER_HPP

#include <limits>
#include <vector>

namespace statistics
{

// ── Interface ─────────────────────────────────────────────────────────────────

/**
 * @brief Pluggable strategy for computing EER from model output embeddings.
 *
 * Implementations receive the raw model logits (one row per test sample) and
 * the corresponding class labels, and return EER in [0, 1] or NaN when the
 * computation is degenerate (e.g. too few samples per speaker).
 *
 * embeddings[i] — model output vector for test sample i (size == n_classes or any D).
 * labels[i]     — integer class label for test sample i.
 * n_classes     — total number of classes in the trained model.
 */
struct IEERScorer
{
    virtual ~IEERScorer() = default;

    /**
     * @brief Compute Equal Error Rate.
     * @return EER in [0, 1] or NaN when computation is degenerate.
     */
    [[nodiscard]] virtual auto compute_eer(const std::vector<std::vector<float>>& embeddings,
        const std::vector<int>& labels,
        int n_classes) const -> double = 0;

    /**
     * @brief Compute AUC (Area Under the ROC curve) for speaker verification.
     *
     * Default implementation returns NaN.  Override in strategies that produce
     * genuine/impostor score distributions (e.g. GenuineImpostorEERScorer).
     *
     * AUC is estimated via the Wilcoxon-Mann-Whitney statistic:
     *   AUC = P(genuine_score > impostor_score)
     * which equals the area under the ROC curve for the two score distributions.
     */
    [[nodiscard]] virtual auto compute_auc(const std::vector<std::vector<float>>& embeddings,
        const std::vector<int>& labels,
        int n_classes) const -> double
    {
        (void) embeddings;
        (void) labels;
        (void) n_classes;
        return std::numeric_limits<double>::quiet_NaN();
    }
};

} // namespace statistics

#endif // NN_STATISTICS_EER_SCORER_HPP
