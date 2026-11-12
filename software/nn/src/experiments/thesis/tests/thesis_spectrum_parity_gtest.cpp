// The autoencoders' and PCA's input held against external libraries.
//
// WHY. thesis::spectral_input decides what every autoencoder and PCA of the thesis sees. Its
// predecessor (the waveform averaged in 256 blocks) was wrong for months without one test
// failing, because the tests only asked "is the output 256 finite values". These tests ask
// "is it the number scipy computes":
//
//   quantity          reference (scripts/testing/gen_thesis_spectrum_refs.py)   C++
//   power spectrum    scipy.signal.welch, symmetric Hann, 512/256, no detrend   spectral_input
//   PCA projection    torch.linalg.svd of the centered rows                     fit_pca,
//   project_pca
//
// The fixture (fixtures/thesis_spectrum_refs.npz) is committed, so neither scipy nor torch
// is needed to build or to run in CI.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "../lib/include/ThesisFeatureExtraction.hpp"
#include "../lib/include/ThesisPca.hpp"
#include "cnpy.h"

using namespace thesis;

namespace
{
const cnpy::npz_t& refs()
{
    static const cnpy::npz_t data = []
    {
        const std::filesystem::path p =
            std::filesystem::path(__FILE__).parent_path() / "fixtures" / "thesis_spectrum_refs.npz";
        return cnpy::npz_load(p.string());
    }();
    return data;
}

// A fixture array as float32 values in numpy C order, with its shape checked by the caller.
const cnpy::NpyArray& arr(const std::string& key)
{
    const auto it = refs().find(key);
    if (it == refs().end()) throw std::runtime_error("missing fixture key: " + key);
    if (it->second.word_size != sizeof(float))
        throw std::runtime_error("fixture key " + key + " is not float32");
    return it->second;
}

std::vector<double> signal(const std::string& key)
{
    const cnpy::NpyArray& a = arr(key);
    const float* p = a.data<float>();
    return {p, p + a.num_vals};
}

std::vector<float> row(const cnpy::NpyArray& a, size_t r, size_t width)
{
    const float* p = a.data<float>() + r * width;
    return {p, p + width};
}
} // namespace

// Signal lengths in the fixture: 512 (exactly one frame), 4096 (15 frames) and 5000
// (18 frames; its last 136 samples are not read, by the rule spectral_input documents).
//
// Tolerance 1e-4 on values in [0, 1]. The reference runs in float64; the C++ windows and
// transforms in float32 (FFTW's single-precision plan). Measured worst difference over the
// three signals: 3e-6 (recorded per case as worst_abs_diff_case_<i>).
TEST(ThesisSpectrumParity, SpectralInputMatchesScipyWelch)
{
    const auto n_cases = static_cast<int>(arr("n_cases").data<float>()[0]);
    ASSERT_EQ(n_cases, 3);
    for (int c = 0; c < n_cases; ++c)
    {
        const std::vector<float> got = spectral_input(signal("signal_" + std::to_string(c)));
        const cnpy::NpyArray& want = arr("input_" + std::to_string(c));
        ASSERT_EQ(want.num_vals, 256u);
        ASSERT_EQ(got.size(), 256u);
        double worst = 0.0;
        for (size_t b = 0; b < 256; ++b)
        {
            const double diff = std::abs(static_cast<double>(got[b]) - want.data<float>()[b]);
            worst = std::max(worst, diff);
            EXPECT_NEAR(got[b], want.data<float>()[b], 1e-4) << "case " << c << ", bin " << b + 1;
        }
        RecordProperty("worst_abs_diff_case_" + std::to_string(c), std::to_string(worst));
    }
}

// The whole chain on 12 signals (3 "speakers" x 4 takes): spectral_input of each, PCA with
// k = 4 fitted on the 12 rows, every row projected. torch's axes are unoriented (v and -v
// are both singular vectors) while fit_pca orients each axis, so a column is compared up to
// one sign, chosen on its largest reference value.
TEST(ThesisSpectrumParity, PcaOfSpectralInputMatchesTorchSvd)
{
    constexpr size_t n = 12, d = 256, k = 4;
    const cnpy::NpyArray& ref_rows = arr("pca_rows");
    const cnpy::NpyArray& ref_proj = arr("pca_projection");
    const cnpy::NpyArray& ref_var = arr("pca_explained_variance");
    ASSERT_EQ(ref_rows.num_vals, n * d);
    ASSERT_EQ(ref_proj.num_vals, n * k);
    ASSERT_EQ(ref_var.num_vals, k);

    std::vector<std::vector<float>> rows;
    for (size_t r = 0; r < n; ++r)
    {
        rows.push_back(spectral_input(signal("pca_signal_" + std::to_string(r))));
        const std::vector<float> want = row(ref_rows, r, d);
        for (size_t b = 0; b < d; ++b)
            ASSERT_NEAR(rows[r][b], want[b], 1e-4) << "row " << r << ", bin " << b + 1;
    }

    const PcaProjection pca = fit_pca(rows, static_cast<int>(k));
    std::vector<std::vector<double>> got;
    for (const auto& r : rows) got.push_back(project_pca(pca, r));

    for (size_t c = 0; c < k; ++c)
    {
        EXPECT_NEAR(
            pca.explained_variance[c], ref_var.data<float>()[c], 0.02 * ref_var.data<float>()[c])
            << "explained variance of axis " << c;
        size_t arg = 0;
        for (size_t r = 1; r < n; ++r)
            if (std::abs(ref_proj.data<float>()[r * k + c]) >
                std::abs(ref_proj.data<float>()[arg * k + c]))
                arg = r;
        const double sign =
            (got[arg][c] >= 0.0) == (ref_proj.data<float>()[arg * k + c] >= 0.0) ? 1.0 : -1.0;
        for (size_t r = 0; r < n; ++r)
            EXPECT_NEAR(sign * got[r][c], ref_proj.data<float>()[r * k + c], 2e-2)
                << "row " << r << ", axis " << c;
    }
}
