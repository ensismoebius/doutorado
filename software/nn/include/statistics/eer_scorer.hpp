/**
 * @file eer_scorer.hpp
 * @brief Pluggable EER computation strategies for speaker verification evaluation.
 *
 * Two concrete strategies are provided:
 *  - ClassificationEERScorer  : legacy — derives EER from closed-set confusion matrices.
 *  - GenuineImpostorEERScorer : SOTA — enrollment/probe scoring with genuine/impostor
 *                               cosine-similarity trials and threshold sweep.
 *
 * Pass any IEERScorer to run_classifier() to select the evaluation protocol.
 */

#ifndef NN_STATISTICS_EER_SCORER_HPP
#define NN_STATISTICS_EER_SCORER_HPP

#include <cstddef>
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
        (void)embeddings; (void)labels; (void)n_classes;
        return std::numeric_limits<double>::quiet_NaN();
    }
};

// ── ClassificationEERScorer ───────────────────────────────────────────────────

/**
 * @brief Legacy EER from closed-set classification confusion matrices.
 *
 * Performs argmax on each embedding row, builds one-vs-rest ConfusionMatrix
 * per class, then calls calculateEER().  Non-standard for speaker verification
 * but preserved for backward compatibility and ablation comparisons.
 *
 * Known limitation: returns NaN when grouped CV places test speakers entirely
 * outside the training set and the model's argmax never predicts those classes.
 */
class ClassificationEERScorer : public IEERScorer
{
public:
    [[nodiscard]] auto compute_eer(const std::vector<std::vector<float>>& embeddings,
        const std::vector<int>& labels,
        int n_classes) const -> double override;
};

// ── GenuineImpostorEERScorer ──────────────────────────────────────────────────

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

#endif // NN_STATISTICS_EER_SCORER_HPP
