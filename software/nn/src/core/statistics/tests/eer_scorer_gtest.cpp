// Unit tests for ClassificationEERScorer and GenuineImpostorEERScorer.
// All inputs are synthetic — no model or SQLite dependency.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
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

// Unit vectors at the given angles (degrees). With labels {0,0,0,0,1,1,1,1} and one enrolment
// sample per speaker (its first), speaker 0 enrols at deg[0] and speaker 1 at deg[4]; the
// other six are probes. Their cosine scores are distinct within the genuine trials and within
// the impostor trials. perfect_embeddings() above gives every genuine trial the same score,
// the one input on which a sweep in the wrong direction still returns the right EER.
std::vector<std::vector<float>> at_degrees(const std::vector<double>& deg)
{
    constexpr double kPi = 3.14159265358979323846;
    std::vector<std::vector<float>> emb;
    for (double d : deg)
        emb.push_back({static_cast<float>(std::cos(d * kPi / 180.0)),
            static_cast<float>(std::sin(d * kPi / 180.0))});
    return emb;
}

const std::vector<int> kTwoSpeakers{0, 0, 0, 0, 1, 1, 1, 1};

// Probes 10/20/30 deg from their own template, 60/70/80 deg from the other one.
const std::vector<double> kSeparated{0, 10, 20, 30, 90, 80, 70, 60};
// Probes 60/70/80 deg from their own template, 10/20/30 deg from the other one.
const std::vector<double> kInverted{0, 80, 70, 60, 90, 10, 20, 30};

} // namespace

// ── EER orientation (regression, 2026-11-12) ─────────────────────────────────
// The EER sweep accepted LOW similarity scores until 2026-11-12: a perfect system scored
// EER 0.667 on kSeparated and a fully inverted one scored 0, while AUC was right. Thesis
// Phase 01 reported EER above 0.5 next to AUC above 0.5 in the same fold (0.724 / 0.518),
// which no single ROC allows (AUC <= 1 - EER^2).

TEST(GenuineImpostorEERScorerTest, SeparatedUntiedScoresHaveZeroEer)
{
    GenuineImpostorEERScorer s(1U);
    const auto emb = at_degrees(kSeparated);
    EXPECT_DOUBLE_EQ(s.compute_auc(emb, kTwoSpeakers, 2), 1.0);
    EXPECT_DOUBLE_EQ(s.compute_eer(emb, kTwoSpeakers, 2), 0.0);
}

TEST(GenuineImpostorEERScorerTest, InvertedUntiedScoresHaveEerOne)
{
    GenuineImpostorEERScorer s(1U);
    const auto emb = at_degrees(kInverted);
    EXPECT_DOUBLE_EQ(s.compute_auc(emb, kTwoSpeakers, 2), 0.0);
    EXPECT_DOUBLE_EQ(s.compute_eer(emb, kTwoSpeakers, 2), 1.0);
}

TEST(ClassificationEERScorerTest, SeparatedUntiedScoresHaveZeroEer)
{
    ClassificationEERScorer s;
    EXPECT_DOUBLE_EQ(s.compute_eer(at_degrees(kSeparated), kTwoSpeakers, 2), 0.0);
}

// ── Unscorable sets throw (they returned NaN, or dropped a speaker, until 2026-11-12) ──

TEST(ClassificationEERScorerTest, EmptyThrows)
{
    ClassificationEERScorer s;
    EXPECT_THROW((void) s.compute_eer({}, {}, 2), std::invalid_argument);
    EXPECT_THROW((void) s.compute_auc({}, {}, 2), std::invalid_argument);
}

TEST(GenuineImpostorEERScorerTest, EmptyThrows)
{
    GenuineImpostorEERScorer s(1U);
    EXPECT_THROW((void) s.compute_eer({}, {}, 2), std::invalid_argument);
    EXPECT_THROW((void) s.compute_auc({}, {}, 2), std::invalid_argument);
}

TEST(GenuineImpostorEERScorerTest, SingleSpeakerThrows)
{
    // Only 1 speaker → no impostor trials.
    GenuineImpostorEERScorer s(1U);
    std::vector<std::vector<float>> emb(5, {1.0f, 0.0f});
    std::vector<int> lbl(5, 0);
    EXPECT_THROW((void) s.compute_eer(emb, lbl, 2), std::invalid_argument);
}

TEST(GenuineImpostorEERScorerTest, SpeakerWithoutProbeThrows)
{
    // n_enroll=3: speaker 1 has 3 samples → 0 probes. It used to be dropped from the
    // trials while speaker 0 was still scored.
    GenuineImpostorEERScorer s(3U);
    std::vector<std::vector<float>> emb(7, {1.0f, 0.0f});
    std::vector<int> lbl = {0, 0, 0, 0, 1, 1, 1};
    EXPECT_THROW((void) s.compute_eer(emb, lbl, 2), std::invalid_argument);
}

TEST(GenuineImpostorEERScorerTest, ZeroEmbeddingThrows)
{
    // A zero vector has no cosine; it used to score 0 against every template.
    GenuineImpostorEERScorer s(1U);
    auto emb = perfect_embeddings(4);
    emb[2] = {0.0f, 0.0f};
    EXPECT_THROW((void) s.compute_eer(emb, perfect_labels(4), 2), std::invalid_argument);
}

TEST(GenuineImpostorEERScorerTest, NonFiniteEmbeddingThrows)
{
    GenuineImpostorEERScorer s(1U);
    auto emb = perfect_embeddings(4);
    emb[5] = {std::numeric_limits<float>::quiet_NaN(), 1.0f};
    EXPECT_THROW((void) s.compute_auc(emb, perfect_labels(4), 2), std::invalid_argument);
}

TEST(GenuineImpostorEERScorerTest, LabelCountMismatchThrows)
{
    GenuineImpostorEERScorer s(1U);
    EXPECT_THROW(
        (void) s.compute_eer(perfect_embeddings(4), {0, 0, 1, 1}, 2), std::invalid_argument);
}

TEST(GenuineImpostorEERScorerTest, ZeroEnrolmentThrows)
{
    GenuineImpostorEERScorer s(0U);
    EXPECT_THROW(
        (void) s.compute_eer(perfect_embeddings(4), perfect_labels(4), 2), std::invalid_argument);
}

// ── Scorable sets ────────────────────────────────────────────────────────────

TEST(ClassificationEERScorerTest, PerfectClassifierZeroEer)
{
    ClassificationEERScorer s;
    EXPECT_DOUBLE_EQ(s.compute_eer(perfect_embeddings(10), perfect_labels(10), 2), 0.0);
}

TEST(GenuineImpostorEERScorerTest, PerfectSeparationZeroEER)
{
    // Speaker 0 embeddings clustered at (1,0); speaker 1 at (0,1).
    GenuineImpostorEERScorer s(1U);
    EXPECT_DOUBLE_EQ(s.compute_eer(perfect_embeddings(6), perfect_labels(6), 2), 0.0);
}

TEST(GenuineImpostorEERScorerTest, TotalOverlapHalfEer)
{
    // Both speakers get identical embeddings → every trial ties → EER = AUC = 0.5.
    GenuineImpostorEERScorer s(1U);
    std::vector<std::vector<float>> emb(8, {1.0f, 1.0f});
    std::vector<int> lbl = {0, 0, 0, 0, 1, 1, 1, 1};
    EXPECT_DOUBLE_EQ(s.compute_eer(emb, lbl, 2), 0.5);
    EXPECT_DOUBLE_EQ(s.compute_auc(emb, lbl, 2), 0.5);
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
    // n_enroll=2: each speaker has 3 samples → 1 probe each.
    GenuineImpostorEERScorer s(2U);
    EXPECT_DOUBLE_EQ(s.compute_eer(perfect_embeddings(3), perfect_labels(3), 2), 0.0);
}

TEST(GenuineImpostorEERScorerTest, AucPerfectSeparationIsOne)
{
    GenuineImpostorEERScorer s(1U);
    EXPECT_DOUBLE_EQ(s.compute_auc(perfect_embeddings(6), perfect_labels(6), 2), 1.0);
}

// Upgraded (audit m-4): ClassificationEERScorer uses the genuine/impostor method.
TEST(ClassificationEERScorerTest, AucPerfectSeparationIsOne)
{
    ClassificationEERScorer s;
    EXPECT_DOUBLE_EQ(s.compute_auc(perfect_embeddings(6), perfect_labels(6), 2), 1.0);
}
