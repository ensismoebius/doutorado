// PCA, the linear reference extractor (ThesisPca.hpp). Known answer first: four points
// built on two orthogonal axes, so the principal axes and every projection are exact.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../lib/include/ThesisFeatureExtraction.hpp"
#include "../lib/include/ThesisPca.hpp"
#include "ThesisTestConfig.hpp"

using namespace thesis;

namespace
{
// u1 = (1, 2)/sqrt(5), u2 = (-2, 1)/sqrt(5). Row i = t_i u1 + s_i u2 with
//   t = (-3, -1, 1, 3)        sum 0, variance along u1 = (9+1+1+9)/3 = 20/3
//   s = (0.5, -0.5, -0.5, 0.5) sum 0, variance along u2 = (4 * 0.25)/3 = 1/3
// and sum t_i s_i = 0, so u1, u2 ARE the principal axes, with eigenvalues 20/3 and 1/3.
const double kR5 = std::sqrt(5.0);
const std::vector<double> kT = {-3.0, -1.0, 1.0, 3.0};
const std::vector<double> kS = {0.5, -0.5, -0.5, 0.5};

std::vector<std::vector<float>> known_rows()
{
    std::vector<std::vector<float>> rows;
    for (size_t i = 0; i < kT.size(); ++i)
        rows.push_back({static_cast<float>((kT[i] * 1.0 + kS[i] * -2.0) / kR5),
            static_cast<float>((kT[i] * 2.0 + kS[i] * 1.0) / kR5)});
    return rows;
}
} // namespace

TEST(ThesisPca, RecoversKnownAxesVariancesAndProjections)
{
    const auto rows = known_rows();
    const PcaProjection pca = fit_pca(rows, 2);

    ASSERT_EQ(pca.components.size(), 2u);
    EXPECT_NEAR(pca.explained_variance[0], 20.0 / 3.0, 1e-5);
    EXPECT_NEAR(pca.explained_variance[1], 1.0 / 3.0, 1e-5);

    // Axis 1 = u1 = (0.447, 0.894): its largest coordinate is already positive.
    EXPECT_NEAR(pca.components[0][0], 1.0 / kR5, 1e-5);
    EXPECT_NEAR(pca.components[0][1], 2.0 / kR5, 1e-5);
    // Axis 2 = u2 = (-0.894, 0.447) has a NEGATIVE largest coordinate, so it is flipped
    // to (0.894, -0.447): the sign convention, not the solver, decides the orientation.
    EXPECT_NEAR(pca.components[1][0], 2.0 / kR5, 1e-5);
    EXPECT_NEAR(pca.components[1][1], -1.0 / kR5, 1e-5);

    // Projection of row i: (t_i, -s_i) — the second axis was flipped.
    for (size_t i = 0; i < rows.size(); ++i)
    {
        const auto p = project_pca(pca, rows[i]);
        ASSERT_EQ(p.size(), 2u);
        EXPECT_NEAR(p[0], kT[i], 1e-5) << "row " << i;
        EXPECT_NEAR(p[1], -kS[i], 1e-5) << "row " << i;
    }
}

TEST(ThesisPca, KeepsTheLeadingComponentsOnly)
{
    const auto pca = fit_pca(known_rows(), 1);
    ASSERT_EQ(pca.components.size(), 1u);
    EXPECT_NEAR(project_pca(pca, known_rows()[3])[0], 3.0, 1e-5);
}

TEST(ThesisPca, RefusesMoreComponentsThanTheDataHas)
{
    // 4 samples of 2 values: at most min(2, 4 - 1) = 2 components.
    EXPECT_THROW(fit_pca(known_rows(), 3), std::invalid_argument);
    EXPECT_THROW(fit_pca(known_rows(), 0), std::invalid_argument);

    // Rank 1 (every point on u1): a second axis would be arbitrary.
    std::vector<std::vector<float>> line;
    for (double t : kT)
        line.push_back({static_cast<float>(t / kR5), static_cast<float>(2.0 * t / kR5)});
    EXPECT_NO_THROW(fit_pca(line, 1));
    EXPECT_THROW(fit_pca(line, 2), std::invalid_argument);
}

TEST(ThesisPca, RefusesRaggedRows)
{
    auto rows = known_rows();
    rows[2].push_back(1.0f);
    EXPECT_THROW(fit_pca(rows, 1), std::invalid_argument);
    const auto pca = fit_pca(known_rows(), 1);
    EXPECT_THROW(project_pca(pca, {1.0f}), std::invalid_argument);
}

namespace
{
ThesisDatasetView sine_view(int n)
{
    ThesisDatasetView v;
    for (int s = 0; s < n; ++s)
    {
        ThesisSample m;
        m.subject_id = s % 3;
        m.stimulus = s;
        m.audio = nn::Tensor(1, 2048);
        m.eeg = nn::Tensor(1, 1024);
        for (int i = 0; i < 2048; ++i)
            m.audio.at(0, i) = static_cast<float>(std::sin(0.01 * (s + 1) * i));
        for (int i = 0; i < 1024; ++i)
            m.eeg.at(0, i) =
                static_cast<float>(std::sin(0.02 * i + s) + 0.1 * std::cos(0.3 * s * i));
        v.samples.push_back(std::move(m));
    }
    v.n_subjects = 3;
    v.n_stimuli = n;
    return v;
}

ThesisConfig pca_cfg(const std::string& modality, int k)
{
    ThesisConfig c = test::phase00(modality);
    c.feature_extraction.strategy = "pca";
    c.feature_extraction.pca.n_components = k;
    c.validate();
    return c;
}
} // namespace

TEST(ThesisPcaStrategy, ProducesKCenteredFeaturesPerSample)
{
    const auto view = sine_view(12);
    const auto sets = extract_features(view, pca_cfg("eeg", 4), nullptr);
    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets[0].label, "pca");
    ASSERT_EQ(sets[0].vectors.size(), view.samples.size());
    for (size_t d = 0; d < 4; ++d)
    {
        double mean = 0.0;
        for (const auto& v : sets[0].vectors)
        {
            ASSERT_EQ(v.size(), 4u);
            mean += v[d];
        }
        EXPECT_NEAR(mean / static_cast<double>(view.samples.size()), 0.0, 1e-6)
            << "projections of the fit set are centered by construction";
    }
}

TEST(ThesisPcaStrategy, LateFusionFitsOnePcaPerSignal)
{
    const auto view = sine_view(12);
    const auto sets = extract_features(view, pca_cfg("fused", 3), nullptr);
    ASSERT_EQ(sets.size(), 1u);
    EXPECT_EQ(sets[0].label, "pca-fused-late");
    EXPECT_EQ(sets[0].vectors[0].size(), 6u) << "3 voice + 3 EEG components";
}

TEST(ThesisPcaStrategy, IsDeterministic)
{
    const auto view = sine_view(12);
    const auto a = extract_features(view, pca_cfg("eeg", 4), nullptr);
    const auto b = extract_features(view, pca_cfg("eeg", 4), nullptr);
    EXPECT_EQ(a[0].vectors, b[0].vectors);
}
