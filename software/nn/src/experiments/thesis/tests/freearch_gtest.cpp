// Proves a variable-DEPTH, variable-WIDTH genome is honoured end-to-end: the thesis
// AE path must build exactly the per-layer widths (not a uniform taper) and produce a
// latent of the genome's bottleneck dimension.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisDataset.hpp"
#include "../lib/include/ThesisFeatureExtraction.hpp"
#include "ThesisTestConfig.hpp"
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
                // A frequency per (subject, stimulus): see test::synthetic_value.
                m.eeg.at(0, i) = test::synthetic_value(
                    i, 0.02 + 0.03 * s + 0.004 * k, static_cast<std::uint32_t>(97 * s + k));
            v.samples.push_back(std::move(m));
        }
    v.n_subjects = ns;
    v.n_stimuli = per;
    return v;
}
// An EEG autoencoder run with the genome-shaped spec; SNN keys only for snn-ae (direct code:
// no time_steps).
ThesisConfig fe(const char* model, std::vector<std::string> enc, std::vector<std::string> dec)
{
    ThesisConfig c = test::phase00("eeg");
    test::set_training(c, 2, 0.01f, 8);
    test::set_autoencoder(c, model, std::move(enc), std::move(dec), "mse");
    if (std::string(model) == "snn-ae") test::set_snn(c, "direct", 1, 1.0f, 0.5f, 0.1f, 0.8f);
    c.validate();
    return c;
}
} // namespace
// Two different depths -> two different latent dims. If the builder ignored the spec
// and used a uniform taper, these would not match the requested bottleneck.
TEST(ThesisFreeArch, DepthThreeLatentOne)
{
    auto v = V(3, 4, 1024);
    auto sets = extract_features(v,
        fe("ann-ae",
            {"linear:3:leaky", "linear:2:leaky", "linear:1:identity"},
            {"linear:2:leaky", "linear:3:leaky", "linear:output:identity"}),
        nullptr);
    ASSERT_FALSE(sets.empty());
    EXPECT_EQ(sets[0].vectors.size(), v.samples.size());
    EXPECT_EQ(sets[0].vectors[0].size(), 1u) << "latent must be the genome bottleneck = 1";
}
TEST(ThesisFreeArch, DepthFourLatentTwo)
{
    auto v = V(3, 4, 1024);
    auto sets = extract_features(v,
        fe("snn-ae",
            {"linear:10:leaky", "linear:5:leaky", "linear:4:leaky", "linear:2:identity"},
            {"linear:4:leaky", "linear:5:leaky", "linear:10:leaky", "linear:output:identity"}),
        nullptr);
    ASSERT_FALSE(sets.empty());
    EXPECT_EQ(sets[0].vectors[0].size(), 2u) << "latent must be the genome bottleneck = 2";
}
