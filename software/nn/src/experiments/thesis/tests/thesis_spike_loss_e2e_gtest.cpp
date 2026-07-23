// End-to-end smoke: does each spike loss actually train the SNN-AE without a
// shape/layout error, and produce a non-degenerate latent?
#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisDataset.hpp"
#include "../lib/include/ThesisFeatureExtraction.hpp"
#include "layers/losses/SpikeTimeLoss.hpp"

using namespace thesis;

namespace
{
ThesisDatasetView synthetic_view(int n_subjects, int per_subject, int siglen)
{
    ThesisDatasetView v;
    for (int s = 0; s < n_subjects; ++s)
        for (int k = 0; k < per_subject; ++k)
        {
            ThesisSample smp;
            smp.subject_id = s;
            smp.stimulus = k;
            smp.eeg = nn::Tensor(1, static_cast<nn::Index>(siglen));
            for (int i = 0; i < siglen; ++i)
                smp.eeg.at(0, i) =
                    static_cast<float>(0.5 + 0.4 * std::sin(0.05 * i + s) + 0.02 * k);
            v.samples.push_back(std::move(smp));
        }
    v.n_subjects = n_subjects;
    v.n_stimuli = per_subject;
    return v;
}

ThesisConfig::FeatureExtraction snn_fe(const char* encoding, const char* loss, int T)
{
    ThesisConfig::FeatureExtraction fe;
    fe.strategy = "autoencoder";
    fe.autoencoder.model = "snn-ae";
    fe.autoencoder.encoder_layer_spec = {"linear:16:leaky", "linear:8:identity"};
    fe.autoencoder.decoder_layer_spec = {"linear:16:leaky", "linear:output:identity"};
    fe.autoencoder.encoding = encoding;
    fe.autoencoder.ae_loss_type = loss;
    fe.autoencoder.time_steps = T;
    fe.autoencoder.voltage_threshold = 0.2f;
    fe.autoencoder.firing_rate_reg_lambda = 0.5f;
    fe.autoencoder.firing_rate_min = 0.1f;
    fe.autoencoder.firing_rate_max = 0.8f;
    return fe;
}

ThesisConfig::Training fast_training()
{
    ThesisConfig::Training t;
    t.epochs = 2;
    t.learning_rate = 0.001f;
    t.samples_per_batch = 8;
    return t;
}
} // namespace

TEST(ThesisSpikeLossE2E, SpikeCountTrainsOnPoisson)
{
    auto view = synthetic_view(3, 4, 128);
    auto sets = extract_features(
        view, snn_fe("poisson", "spikecount", 4), fast_training(), "eeg", "late", 42u);
    ASSERT_FALSE(sets.empty());
    ASSERT_EQ(sets[0].vectors.size(), view.samples.size());
    EXPECT_EQ(sets[0].vectors[0].size(), 8u) << "latent dim should be 8";
}

TEST(ThesisSpikeLossE2E, SpikeTimeTrainsOnLatencyTimeMajor)
{
    auto view = synthetic_view(3, 4, 128);
    auto sets = extract_features(
        view, snn_fe("latency", "spiketime", 4), fast_training(), "eeg", "late", 42u);
    ASSERT_FALSE(sets.empty());
    ASSERT_EQ(sets[0].vectors.size(), view.samples.size());
    EXPECT_EQ(sets[0].vectors[0].size(), 8u) << "latent dim should be 8";
}

// batch_size > 1 for spiketime: inputs are pre-interleaved into (T*g, D) groups with
// row = t*g + b, so real batching works despite SpikeTimeLoss's fixed layout.
TEST(ThesisSpikeLossE2E, SpikeTimeHonoursBatchSizeGreaterThanOne)
{
    auto view = synthetic_view(3, 4, 128); // 12 samples
    auto tr = fast_training();
    tr.samples_per_batch = 5; // 12 = 5 + 5 + 2 -> exercises a trailing partial group
    auto sets = extract_features(view, snn_fe("latency", "spiketime", 4), tr, "eeg", "late", 42u);
    ASSERT_FALSE(sets.empty());
    EXPECT_EQ(sets[0].vectors.size(), view.samples.size());
    EXPECT_EQ(sets[0].vectors[0].size(), 8u);
}

// THE core guarantee, swept over batch sizes: for ANY configuration the extractor
// either returns well-formed features, or it RAISES. It must never return features
// produced by an untrained model. (Whether a given (lr, batch, T, threshold, seed)
// combination lands in the no-spike deadlock is genuinely config-dependent — the point
// is that landing there is loud, not silent.)
TEST(ThesisSpikeLossE2E, SpikeTimeIsNeverSilentlyWrongAcrossBatchSizes)
{
    auto view = synthetic_view(2, 3, 128); // 6 samples
    for (int bs : {1, 2, 4, 6, 8})
    {
        auto tr = fast_training();
        tr.samples_per_batch = bs;
        try
        {
            auto sets =
                extract_features(view, snn_fe("latency", "spiketime", 3), tr, "eeg", "late", 7u);
            // Returned => the guard verified the gradient was live. Features must be sane.
            ASSERT_FALSE(sets.empty()) << "bs=" << bs;
            EXPECT_EQ(sets[0].vectors.size(), view.samples.size()) << "bs=" << bs;
            EXPECT_EQ(sets[0].vectors[0].size(), 8u) << "bs=" << bs;
        }
        catch (const std::runtime_error& e)
        {
            // Threw => must be the dead-gradient guard, naming cause and remedy.
            const std::string m = e.what();
            EXPECT_NE(m.find("ALL-ZERO gradient"), std::string::npos)
                << "bs=" << bs << " threw something other than the guard: " << m;
        }
    }
}

// ─── SpikeTimeLoss gradient liveness ────────────────────────────────────────
//
// SpikeTimeLossImpl::backward writes a gradient ONLY at the predicted first-spike
// row: `if (t < T) grad.at(t*B + b, f) = g;`. When a unit never spikes, pt == T,
// the guard fails, and NO gradient is written — silent no-learning, not an error.
// The decoder ends in `linear:output:identity` (continuous) and first_spike_times
// thresholds at > 0.5, so this is reachable if the decoder saturates below 0.5.
// These pin both sides of that behaviour so a regression is visible.

namespace
{
float abs_sum(const nn::Tensor& g)
{
    float s = 0.0f;
    for (size_t i = 0; i < g.rows(); ++i)
        for (size_t j = 0; j < g.cols(); ++j) s += std::abs(g.at(i, j));
    return s;
}
} // namespace

TEST(ThesisSpikeTimeGradient, ZeroWhenPredictionNeverCrossesThreshold)
{
    const int T = 4, B = 2, F = 3;
    SpikeTimeLossImpl<nn::Backend> loss(T);
    nn::Tensor tgt(T * B, F);
    tgt.setZero();
    for (int b = 0; b < B; ++b) tgt.at(1 * B + b, 0) = 1.0f; // target spikes at t=1

    nn::Tensor pred(T * B, F);
    pred.setZero();
    for (size_t i = 0; i < pred.rows(); ++i)
        for (size_t j = 0; j < pred.cols(); ++j) pred.at(i, j) = 0.49f; // just below threshold

    loss.set_target(tgt);
    loss.forward(pred, true);
    EXPECT_FLOAT_EQ(abs_sum(loss.backward(pred)), 0.0f)
        << "documented no-spike deadlock: silent zero gradient";
}

TEST(ThesisSpikeTimeGradient, NonZeroWhenPredictionSpikes)
{
    const int T = 4, B = 2, F = 3;
    SpikeTimeLossImpl<nn::Backend> loss(T);
    nn::Tensor tgt(T * B, F);
    tgt.setZero();
    for (int b = 0; b < B; ++b) tgt.at(1 * B + b, 0) = 1.0f;

    nn::Tensor pred(T * B, F);
    pred.setZero();
    for (int b = 0; b < B; ++b) pred.at(3 * B + b, 0) = 1.0f; // spikes late -> real error

    loss.set_target(tgt);
    loss.forward(pred, true);
    EXPECT_GT(abs_sum(loss.backward(pred)), 0.0f);
}

// ─── "meaningless gradients never happen silently" ──────────────────────────
//
// Two complementary guarantees:
//   (A) LIVENESS  — every wired loss actually moves the weights on real data, proven
//                   with the encoder rate-regularizer DISABLED so the only possible
//                   source of change is the reconstruction loss gradient itself.
//   (B) DETECTION — if a gradient ever does go all-zero, the run throws instead of
//                   emitting features from an untrained model.

namespace
{
double max_abs_diff(
    const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        for (size_t j = 0; j < a[i].size() && j < b[i].size(); ++j)
            m = std::max(m, std::abs(a[i][j] - b[i][j]));
    return m;
}

// Train the same config for 1 vs many epochs. Identical features => nothing was
// learned => the gradient was dead.
double learning_delta(const char* model, const char* enc, const char* loss, int T, float lam)
{
    auto view = synthetic_view(3, 4, 128);
    auto fe = snn_fe(enc, loss, T);
    fe.autoencoder.model = model;
    fe.autoencoder.firing_rate_reg_lambda = lam;

    ThesisConfig::Training few = fast_training();
    few.epochs = 1;
    few.learning_rate = 0.01f;
    ThesisConfig::Training many = few;
    many.epochs = 40;

    auto a = extract_features(view, fe, few, "eeg", "late", 42u);
    auto b = extract_features(view, fe, many, "eeg", "late", 42u);
    EXPECT_FALSE(a.empty());
    EXPECT_FALSE(b.empty());
    return max_abs_diff(a[0].vectors, b[0].vectors);
}
} // namespace

// (A) POLICY — the guarantee itself: an all-zero-gradient run can never return
// normally. Tested directly and deterministically, rather than by trying to coax a
// real network into the deadlock (which depends on init and is not reproducible).
TEST(ThesisGradientPolicy, AllZeroGradientAlwaysThrows)
{
    EXPECT_THROW(
        assert_gradients_were_live(10, 10, "spiketime", "latency", 0.0f), std::runtime_error);
    EXPECT_THROW(
        assert_gradients_were_live(1, 1, "spikecount", "poisson", 0.0f), std::runtime_error);
}

TEST(ThesisGradientPolicy, PartialZeroGradientIsAllowed)
{
    // Some zero-gradient batches are legitimate near convergence; only ALL is fatal.
    EXPECT_NO_THROW(assert_gradients_were_live(10, 9, "spiketime", "latency", 0.5f));
    EXPECT_NO_THROW(assert_gradients_were_live(10, 0, "spikecount", "poisson", 0.5f));
    EXPECT_NO_THROW(assert_gradients_were_live(0, 0, "spiketime", "latency", 0.5f));
}

TEST(ThesisGradientPolicy, ErrorNamesCauseAndRemedy)
{
    try
    {
        assert_gradients_were_live(7, 7, "spiketime", "latency", 0.0f);
        FAIL() << "expected throw";
    }
    catch (const std::runtime_error& e)
    {
        const std::string m = e.what();
        EXPECT_NE(m.find("ALL-ZERO gradient"), std::string::npos);
        EXPECT_NE(m.find("spiketime"), std::string::npos);
        EXPECT_NE(m.find("firing_rate_reg_lambda"), std::string::npos);
        EXPECT_NE(m.find("voltage_threshold"), std::string::npos);
        EXPECT_NE(m.find("Refusing to emit features"), std::string::npos);
    }
}

// (B) LIVENESS on real data. For the CONTINUOUS models a latent change is a reliable
// probe. For the SNN it is NOT: the latent is a spike rate over T steps, so it is
// quantised to {0, 1/T, ..., 1} and can legitimately stay put while weights move.
// There, the meaningful assertion is that extract_features RETURNS — which means the
// runtime guard checked every batch and found the gradient live.
TEST(ThesisGradientLiveness, ContinuousLossesMoveTheLatent_RateRegDisabled)
{
    EXPECT_GT(learning_delta("ann-ae", "direct", "mse", 1, 0.0f), 0.0)
        << "ann-ae/direct/mse learned nothing";
    EXPECT_GT(learning_delta("ann-ae", "direct", "mae", 1, 0.0f), 0.0)
        << "ann-ae/direct/mae learned nothing";
}

TEST(ThesisGradientLiveness, SpikeLossesPassTheRuntimeGuard)
{
    // Returning normally == the guard saw a non-zero gradient on at least one batch.
    // Covers both the shipped lambda=0.5 profiles and the isolated lambda=0 case.
    for (float lam : {0.0f, 0.5f})
        for (auto [enc, loss, T] :
            {std::tuple{"poisson", "spikecount", 4}, std::tuple{"latency", "spiketime", 4}})
        {
            auto view = synthetic_view(3, 4, 128);
            auto fe = snn_fe(enc, loss, T);
            fe.autoencoder.firing_rate_reg_lambda = lam;
            EXPECT_NO_THROW({
                auto sets = extract_features(view, fe, fast_training(), "eeg", "late", 42u);
                EXPECT_FALSE(sets.empty());
            }) << "guard tripped for "
               << loss << " lam=" << lam;
        }
}
