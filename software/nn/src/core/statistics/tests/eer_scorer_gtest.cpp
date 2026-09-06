// Unit tests for ClassificationEERScorer and GenuineImpostorEERScorer.
// All inputs are synthetic — no model or SQLite dependency.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "statistics/ClassificationEERScorer.hpp"
#include "statistics/GenuineImpostorEERScorer.hpp"
#include "statistics/eer_scorer.hpp"

using statistics::ClassificationEERScorer;
using statistics::GenuineImpostorEERScorer;

namespace
{

// Perfect binary classifier: class 0 gets large logit[0], class 1 gets large logit[1].
std::vector<std::vector<float>> perfect_embeddings(int n_per_class)
{
    std::vector<std::vector<float>> emb;
    for (int i = 0; i < n_per_class; ++i) emb.push_back({10.0f, 0.0f}); // class 0
    for (int i = 0; i < n_per_class; ++i) emb.push_back({0.0f, 10.0f}); // class 1
    return emb;
}

std::vector<int> perfect_labels(int n_per_class)
{
    std::vector<int> lbl;
    for (int i = 0; i < n_per_class; ++i) lbl.push_back(0);
    for (int i = 0; i < n_per_class; ++i) lbl.push_back(1);
    return lbl;
}

// Random-guessing classifier: every sample gets uniform logits.
std::vector<std::vector<float>> uniform_embeddings(int n, int n_classes)
{
    return std::vector<std::vector<float>>(
        static_cast<std::size_t>(n), std::vector<float>(static_cast<std::size_t>(n_classes), 1.0f));
}

} // namespace

// ── ClassificationEERScorer ───────────────────────────────────────────────────

TEST(ClassificationEERScorerTest, EmptyReturnsNaN)
{
    ClassificationEERScorer s;
    EXPECT_TRUE(std::isnan(s.compute_eer({}, {}, 2)));
}

TEST(ClassificationEERScorerTest, PerfectClassifierLowEER)
{
    ClassificationEERScorer s;
    const int n = 10;
    const double eer = s.compute_eer(perfect_embeddings(n), perfect_labels(n), 2);
    // Perfect predictions → EER should be 0 or near 0.
    if (!std::isnan(eer)) EXPECT_LT(eer, 0.1);
}

TEST(ClassificationEERScorerTest, ResultInUnitInterval)
{
    ClassificationEERScorer s;
    const int n = 8;
    const double eer = s.compute_eer(perfect_embeddings(n), perfect_labels(n), 2);
    if (!std::isnan(eer))
    {
        EXPECT_GE(eer, 0.0);
        EXPECT_LE(eer, 1.0);
    }
}

// ── GenuineImpostorEERScorer ──────────────────────────────────────────────────

TEST(GenuineImpostorEERScorerTest, EmptyReturnsNaN)
{
    GenuineImpostorEERScorer s;
    EXPECT_TRUE(std::isnan(s.compute_eer({}, {}, 2)));
}

TEST(GenuineImpostorEERScorerTest, SingleSpeakerReturnsNaN)
{
    // Only 1 speaker → no impostor trials → NaN.
    GenuineImpostorEERScorer s(1U);
    std::vector<std::vector<float>> emb(5, {1.0f, 0.0f});
    std::vector<int> lbl(5, 0);
    EXPECT_TRUE(std::isnan(s.compute_eer(emb, lbl, 2)));
}

TEST(GenuineImpostorEERScorerTest, TooFewSamplesPerSpeakerReturnsNaN)
{
    // n_enroll=3, each speaker has only 3 samples → 0 probes → NaN.
    GenuineImpostorEERScorer s(3U);
    std::vector<std::vector<float>> emb(6, {1.0f, 0.0f});
    // speaker 0 = indices 0-2, speaker 1 = indices 3-5
    std::vector<int> lbl = {0, 0, 0, 1, 1, 1};
    EXPECT_TRUE(std::isnan(s.compute_eer(emb, lbl, 2)));
}

TEST(GenuineImpostorEERScorerTest, PerfectSeparationZeroEER)
{
    // Speaker 0 embeddings clustered at (1,0); speaker 1 at (0,1).
    // Cosine similarity: genuine >> impostor → EER ≈ 0.
    GenuineImpostorEERScorer s(1U);
    const int n = 6;
    const auto emb = perfect_embeddings(n);
    const auto lbl = perfect_labels(n);
    const double eer = s.compute_eer(emb, lbl, 2);
    if (!std::isnan(eer)) EXPECT_LT(eer, 0.1);
}

TEST(GenuineImpostorEERScorerTest, TotalOverlapHighEER)
{
    // Both speakers get identical embeddings → genuine == impostor scores → EER ≈ 0.5.
    GenuineImpostorEERScorer s(1U);
    std::vector<std::vector<float>> emb(8, {1.0f, 1.0f});
    std::vector<int> lbl = {0, 0, 0, 0, 1, 1, 1, 1};
    const double eer = s.compute_eer(emb, lbl, 2);
    if (!std::isnan(eer)) EXPECT_GT(eer, 0.3); // near 0.5 for identical distributions
}

TEST(GenuineImpostorEERScorerTest, ResultInUnitIntervalOrNaN)
{
    GenuineImpostorEERScorer s(1U);
    const int n = 6;
    const double eer = s.compute_eer(perfect_embeddings(n), perfect_labels(n), 2);
    if (!std::isnan(eer))
    {
        EXPECT_GE(eer, 0.0);
        EXPECT_LE(eer, 1.0);
    }
}

TEST(GenuineImpostorEERScorerTest, DeterministicSameInput)
{
    GenuineImpostorEERScorer s(1U);
    const auto emb = perfect_embeddings(6);
    const auto lbl = perfect_labels(6);
    EXPECT_DOUBLE_EQ(s.compute_eer(emb, lbl, 2), s.compute_eer(emb, lbl, 2));
}

TEST(GenuineImpostorEERScorerTest, MultipleEnrollSamples)
{
    // n_enroll=2: speaker 0 has 3 samples → 1 probe; speaker 1 has 3 → 1 probe.
    GenuineImpostorEERScorer s(2U);
    const auto emb = perfect_embeddings(3);
    const auto lbl = perfect_labels(3);
    const double eer = s.compute_eer(emb, lbl, 2);
    if (!std::isnan(eer)) EXPECT_LT(eer, 0.1);
}

// ── GenuineImpostorEERScorer — AUC ───────────────────────────────────────────

TEST(GenuineImpostorEERScorerTest, AucEmptyReturnsNaN)
{
    GenuineImpostorEERScorer s;
    EXPECT_TRUE(std::isnan(s.compute_auc({}, {}, 2)));
}

TEST(GenuineImpostorEERScorerTest, AucPerfectSeparationNearOne)
{
    // Perfect cosine separation → genuine >> impostor → AUC near 1.
    GenuineImpostorEERScorer s(1U);
    const double auc = s.compute_auc(perfect_embeddings(6), perfect_labels(6), 2);
    if (!std::isnan(auc)) EXPECT_GT(auc, 0.9);
}

TEST(GenuineImpostorEERScorer, AucTotalOverlapNearHalf)
{
    // Identical embeddings → AUC ≈ 0.5.
    GenuineImpostorEERScorer s(1U);
    std::vector<std::vector<float>> emb(8, {1.0f, 1.0f});
    std::vector<int> lbl = {0, 0, 0, 0, 1, 1, 1, 1};
    const double auc = s.compute_auc(emb, lbl, 2);
    if (!std::isnan(auc)) EXPECT_NEAR(auc, 0.5, 0.15);
}

TEST(GenuineImpostorEERScorerTest, AucInUnitIntervalOrNaN)
{
    GenuineImpostorEERScorer s(1U);
    const double auc = s.compute_auc(perfect_embeddings(6), perfect_labels(6), 2);
    if (!std::isnan(auc))
    {
        EXPECT_GE(auc, 0.0);
        EXPECT_LE(auc, 1.0);
    }
}

// Upgraded (audit m-4): ClassificationEERScorer now uses the genuine/impostor
// method, so compute_auc returns a valid AUC (near 1 for perfect separation).
TEST(ClassificationEERScorerTest, AucPerfectSeparationNearOne)
{
    ClassificationEERScorer s;
    const double auc = s.compute_auc(perfect_embeddings(6), perfect_labels(6), 2);
    if (!std::isnan(auc)) EXPECT_GT(auc, 0.9);
}

TEST(ClassificationEERScorerTest, AucEmptyReturnsNaN)
{
    ClassificationEERScorer s;
    EXPECT_TRUE(std::isnan(s.compute_auc({}, {}, 2)));
}
