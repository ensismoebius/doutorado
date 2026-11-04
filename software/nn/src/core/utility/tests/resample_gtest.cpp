#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "utility/Resample.hpp"

namespace
{

auto tone(double hz, double fs, std::size_t n) -> std::vector<float>
{
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] =
            static_cast<float>(std::sin(2.0 * std::numbers::pi * hz * static_cast<double>(i) / fs));
    return x;
}

auto rms(const std::vector<float>& x, std::size_t skip) -> double
{
    double sum = 0.0;
    for (std::size_t i = skip; i + skip < x.size(); ++i)
        sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    return std::sqrt(sum / static_cast<double>(x.size() - 2U * skip));
}

} // namespace

// The case the loader needs: an EEG band tone recorded at 128 Hz must come out as the tone a
// 160 Hz recording would hold. 40 Hz is the band's upper edge; away from the ends the error is
// that of the kernel, not of the rate.
TEST(ResampleBandlimited, A128HzToneBecomesThe160HzTone)
{
    constexpr std::size_t kInput = 128U * 30U;
    constexpr std::size_t kSkip = 160U * 2U;
    for (const double hz : {1.0, 10.0, 25.0, 40.0})
    {
        const auto y = nn::utility::resample_bandlimited(tone(hz, 128.0, kInput), 128.0, 160.0);
        ASSERT_EQ(y.size(), 160U * 30U - 1U) << "last output inside the input";
        const auto truth = tone(hz, 160.0, y.size());
        double worst = 0.0;
        for (std::size_t i = kSkip; i + kSkip < y.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(y[i] - truth[i])));
        EXPECT_LT(worst, 1e-3) << hz << " Hz";
    }
}

TEST(ResampleBandlimited, ADownsampleDoesNotAlias)
{
    constexpr std::size_t kInput = 512U * 30U;
    constexpr std::size_t kSkip = 160U * 2U;
    // 100 Hz is above the 80 Hz Nyquist of 160 Hz: it must vanish, not fold down to 60 Hz.
    const auto gone = nn::utility::resample_bandlimited(tone(100.0, 512.0, kInput), 512.0, 160.0);
    EXPECT_LT(rms(gone, kSkip), 1e-2);
    const auto kept = nn::utility::resample_bandlimited(tone(30.0, 512.0, kInput), 512.0, 160.0);
    EXPECT_NEAR(rms(kept, kSkip), 1.0 / std::numbers::sqrt2, 1e-2);
}

TEST(ResampleBandlimited, AConstantStaysConstantUpToTheEnds)
{
    const std::vector<float> x(500, 3.5F);
    for (const float v : nn::utility::resample_bandlimited(x, 128.0, 160.0))
        EXPECT_NEAR(v, 3.5F, 1e-4F);
}

TEST(ResampleBandlimited, SameRateReturnsTheInputAndBadArgumentsThrow)
{
    const std::vector<float> x{1.0F, 2.0F, 3.0F};
    EXPECT_EQ(nn::utility::resample_bandlimited(x, 160.0, 160.0), x);
    EXPECT_THROW(nn::utility::resample_bandlimited(x, 0.0, 160.0), std::invalid_argument);
    EXPECT_THROW(nn::utility::resample_bandlimited(x, 160.0, -1.0), std::invalid_argument);
    EXPECT_THROW(nn::utility::resample_bandlimited({}, 128.0, 160.0), std::invalid_argument);
    EXPECT_THROW(nn::utility::resample_bandlimited(x, 128.0, 160.0, 0), std::invalid_argument);
}
