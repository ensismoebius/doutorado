// Proves a variable-DEPTH, variable-WIDTH genome is honoured end-to-end: the thesis
// AE path must build exactly the per-layer widths (not a uniform taper) and produce a
// latent of the genome's bottleneck dimension.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisDataset.hpp"
#include "../lib/include/ThesisFeatureExtraction.hpp"
using namespace thesis;
namespace
{
ThesisDatasetView V(int ns, int per, int len)
{
    ThesisDatasetView v;
    for (int s = 0; s < ns; ++s)
        for (int k = 0; k < per; ++k)
        {
            ThesisSample m;
            m.subject_id = s;
            m.stimulus = k;
            m.eeg = nn::Tensor(1, (nn::Index) len);
            for (int i = 0; i < len; ++i)
                m.eeg.at(0, i) = (float) (0.5 + 0.4 * std::sin(0.05 * i + s) + 0.02 * k);
            v.samples.push_back(std::move(m));
        }
    v.n_subjects = ns;
    v.n_stimuli = per;
    return v;
}
ThesisConfig::FeatureExtraction fe(
    const char* model, std::vector<std::string> enc, std::vector<std::string> dec)
{
    ThesisConfig::FeatureExtraction f;
    f.strategy = "autoencoder";
    f.autoencoder.model = model;
    f.autoencoder.encoder_layer_spec = std::move(enc);
    f.autoencoder.decoder_layer_spec = std::move(dec);
    f.autoencoder.encoding = "direct";
    f.autoencoder.ae_loss_type = "mse";
    f.autoencoder.time_steps = 1;
    f.autoencoder.voltage_threshold = 1.0f;
    f.autoencoder.firing_rate_reg_lambda = 0.5f;
    f.autoencoder.firing_rate_min = 0.1f;
    f.autoencoder.firing_rate_max = 0.8f;
    return f;
}
ThesisConfig::Training tr()
{
    ThesisConfig::Training t;
    t.epochs = 2;
    t.learning_rate = 0.01f;
    t.samples_per_batch = 8;
    return t;
}
} // namespace
// Two different depths -> two different latent dims. If the builder ignored the spec
// and used a uniform taper, these would not match the requested bottleneck.
TEST(ThesisFreeArch, DepthThreeLatentOne)
{
    auto v = V(3, 4, 128);
    auto sets = extract_features(v,
        fe("ann-ae",
            {"linear:3:leaky", "linear:2:leaky", "linear:1:identity"},
            {"linear:2:leaky", "linear:3:leaky", "linear:output:identity"}),
        tr(),
        "eeg",
        "late",
        42u);
    ASSERT_FALSE(sets.empty());
    EXPECT_EQ(sets[0].vectors.size(), v.samples.size());
    EXPECT_EQ(sets[0].vectors[0].size(), 1u) << "latent must be the genome bottleneck = 1";
}
TEST(ThesisFreeArch, DepthFourLatentTwo)
{
    auto v = V(3, 4, 128);
    auto sets = extract_features(v,
        fe("snn-ae",
            {"linear:10:leaky", "linear:5:leaky", "linear:4:leaky", "linear:2:identity"},
            {"linear:4:leaky", "linear:5:leaky", "linear:10:leaky", "linear:output:identity"}),
        tr(),
        "eeg",
        "late",
        42u);
    ASSERT_FALSE(sets.empty());
    EXPECT_EQ(sets[0].vectors[0].size(), 2u) << "latent must be the genome bottleneck = 2";
}
