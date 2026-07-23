// End-to-end smoke: does each spike loss actually train the SNN-AE without a
// shape/layout error, and produce a non-degenerate latent?
#include <gtest/gtest.h>

#include <cstdint>
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

// Batch size must not change WHAT is learned about each sample's identity: the latent
// dimensionality and per-sample alignment hold regardless of grouping.
TEST(ThesisSpikeLossE2E, SpikeTimeBatchSizeDoesNotBreakAlignment)
{
    auto view = synthetic_view(2, 3, 128); // 6 samples
    for (int bs : {1, 2, 4, 6, 8})
    {
        auto tr = fast_training();
        tr.samples_per_batch = bs;
        auto sets =
            extract_features(view, snn_fe("latency", "spiketime", 3), tr, "eeg", "late", 7u);
        ASSERT_FALSE(sets.empty()) << "bs=" << bs;
        EXPECT_EQ(sets[0].vectors.size(), view.samples.size()) << "bs=" << bs;
        EXPECT_EQ(sets[0].vectors[0].size(), 8u) << "bs=" << bs;
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
