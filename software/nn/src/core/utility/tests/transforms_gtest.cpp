/**
 * @file transforms_gtest.cpp
 * @brief Unit tests for nn::transforms: Compose, AudioMeanStdNormalize,
 *        EEGWindowZScore, FusedModalityTransform, GaussianNoise, RandomCrop,
 *        RandomIndexCrop, and WindowZScore.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor/Tensor.hpp"
#include "utility/Transforms.hpp"

namespace
{

// ─── helpers ─────────────────────────────────────────────────────────────────

// Build a (rows × cols) tensor filled with a constant value.
auto constant_tensor(nn::Index rows, nn::Index cols, float value) -> nn::Tensor
{
    nn::Tensor t(rows, cols);
    for (nn::Index i = 0; i < rows; ++i)
        for (nn::Index j = 0; j < cols; ++j) t.at(i, j) = value;
    return t;
}

// Build a (rows × cols) tensor where t[i,j] = base + i * cols + j.
auto sequential_tensor(nn::Index rows, nn::Index cols, float base = 0.0F) -> nn::Tensor
{
    nn::Tensor t(rows, cols);
    for (nn::Index i = 0; i < rows; ++i)
        for (nn::Index j = 0; j < cols; ++j) t.at(i, j) = base + static_cast<float>(i * cols + j);
    return t;
}

// Compute the mean of all elements in row i.
auto row_mean(const nn::Tensor& t, nn::Index i) -> float
{
    float sum = 0.0F;
    for (nn::Index j = 0; j < t.cols(); ++j) sum += t.at(i, j);
    return sum / static_cast<float>(t.cols());
}

// Compute the population std of all elements in row i.
auto row_std(const nn::Tensor& t, nn::Index i) -> float
{
    const float mu = row_mean(t, i);
    float sq_sum = 0.0F;
    for (nn::Index j = 0; j < t.cols(); ++j)
    {
        const float d = t.at(i, j) - mu;
        sq_sum += d * d;
    }
    return std::sqrt(sq_sum / static_cast<float>(t.cols()));
}

// Compute the mean of column j across all rows.
auto col_mean(const nn::Tensor& t, nn::Index j) -> float
{
    float sum = 0.0F;
    for (nn::Index i = 0; i < t.rows(); ++i) sum += t.at(i, j);
    return sum / static_cast<float>(t.rows());
}

// Compute the population std of column j.
auto col_std(const nn::Tensor& t, nn::Index j) -> float
{
    const float mu = col_mean(t, j);
    float sq_sum = 0.0F;
    for (nn::Index i = 0; i < t.rows(); ++i)
    {
        const float d = t.at(i, j) - mu;
        sq_sum += d * d;
    }
    return std::sqrt(sq_sum / static_cast<float>(t.rows()));
}

} // namespace

// ─── EEGWindowZScore ─────────────────────────────────────────────────────────

TEST(EEGWindowZScore, OutputShapePreserved)
{
    nn::transforms::EEGWindowZScore zscore;
    const auto input = sequential_tensor(4, 8);
    const auto out = zscore(input);
    EXPECT_EQ(out.rows(), input.rows());
    EXPECT_EQ(out.cols(), input.cols());
}

TEST(EEGWindowZScore, PerRowZeroMean)
{
    nn::transforms::EEGWindowZScore zscore;
    const auto input = sequential_tensor(5, 10);
    const auto out = zscore(input);
    for (nn::Index i = 0; i < out.rows(); ++i)
        EXPECT_NEAR(row_mean(out, i), 0.0F, 1e-5F) << "row " << i;
}

TEST(EEGWindowZScore, PerRowUnitStd)
{
    nn::transforms::EEGWindowZScore zscore;
    const auto input = sequential_tensor(5, 10);
    const auto out = zscore(input);
    for (nn::Index i = 0; i < out.rows(); ++i)
        EXPECT_NEAR(row_std(out, i), 1.0F, 1e-5F) << "row " << i;
}

TEST(EEGWindowZScore, ConstantRowProducesZeros)
{
    // A constant row has zero variance; result should be near zero (eps stabilized).
    nn::transforms::EEGWindowZScore zscore;
    const auto input = constant_tensor(3, 6, 7.0F);
    const auto out = zscore(input);
    for (nn::Index i = 0; i < out.rows(); ++i)
        for (nn::Index j = 0; j < out.cols(); ++j) EXPECT_NEAR(out.at(i, j), 0.0F, 1e-4F);
}

TEST(EEGWindowZScore, EmptyTensorPassthrough)
{
    nn::transforms::EEGWindowZScore zscore;
    const nn::Tensor empty(0, 8);
    const auto out = zscore(empty);
    EXPECT_EQ(out.rows(), 0u);
    EXPECT_EQ(out.cols(), 8u);
}

// ─── AudioMeanStdNormalize ───────────────────────────────────────────────────

TEST(AudioMeanStdNormalize, InfersDimensionsFromFirstBatch)
{
    nn::transforms::AudioMeanStdNormalize norm;
    norm.accumulate(sequential_tensor(10, 4));
    norm.finalize();
    EXPECT_TRUE(norm.is_fitted());
}

TEST(AudioMeanStdNormalize, OutputShapePreserved)
{
    nn::transforms::AudioMeanStdNormalize norm;
    const auto train = sequential_tensor(20, 4);
    norm.accumulate(train);
    norm.finalize();
    const auto out = norm(sequential_tensor(5, 4));
    EXPECT_EQ(out.rows(), 5u);
    EXPECT_EQ(out.cols(), 4u);
}

TEST(AudioMeanStdNormalize, FittingProducesZeroMeanPerColumn)
{
    // After fitting and applying to the same training data the column means
    // should be approximately zero (within floating-point tolerance).
    nn::transforms::AudioMeanStdNormalize norm;
    const auto train = sequential_tensor(20, 6);
    norm.accumulate(train);
    norm.finalize();
    const auto out = norm(train);
    for (nn::Index j = 0; j < out.cols(); ++j)
        EXPECT_NEAR(col_mean(out, j), 0.0F, 1e-4F) << "column " << j;
}

TEST(AudioMeanStdNormalize, FittingProducesUnitStdPerColumn)
{
    nn::transforms::AudioMeanStdNormalize norm;
    const auto train = sequential_tensor(20, 6);
    norm.accumulate(train);
    norm.finalize();
    const auto out = norm(train);
    for (nn::Index j = 0; j < out.cols(); ++j)
        EXPECT_NEAR(col_std(out, j), 1.0F, 1e-4F) << "column " << j;
}

TEST(AudioMeanStdNormalize, MultiBatchAccumulationMatchesSingleBatch)
{
    // Fitting on two halves must produce the same statistics as fitting on
    // the full tensor at once.
    const auto full = sequential_tensor(20, 4);
    const auto half1 = full.block(0, 0, 10, 4);
    const auto half2 = full.block(10, 0, 10, 4);

    nn::transforms::AudioMeanStdNormalize norm_single;
    norm_single.accumulate(full);
    norm_single.finalize();

    nn::transforms::AudioMeanStdNormalize norm_multi;
    norm_multi.accumulate(half1);
    norm_multi.accumulate(half2);
    norm_multi.finalize();

    const auto out_single = norm_single(full);
    const auto out_multi = norm_multi(full);

    for (nn::Index i = 0; i < out_single.rows(); ++i)
        for (nn::Index j = 0; j < out_single.cols(); ++j)
            EXPECT_NEAR(out_single.at(i, j), out_multi.at(i, j), 1e-5F);
}

TEST(AudioMeanStdNormalize, ThrowsIfAppliedBeforeFinalize)
{
    nn::transforms::AudioMeanStdNormalize norm;
    norm.accumulate(sequential_tensor(5, 3));
    EXPECT_THROW(norm(sequential_tensor(2, 3)), std::runtime_error);
}

TEST(AudioMeanStdNormalize, ThrowsOnDimensionMismatch)
{
    nn::transforms::AudioMeanStdNormalize norm;
    norm.accumulate(sequential_tensor(10, 4));
    norm.finalize();
    EXPECT_THROW(norm(sequential_tensor(3, 6)), std::invalid_argument);
}

TEST(AudioMeanStdNormalize, ThrowsOnZeroSamples)
{
    nn::transforms::AudioMeanStdNormalize norm;
    EXPECT_THROW(norm.finalize(), std::runtime_error);
}

TEST(AudioMeanStdNormalize, ConstantColumnProducesZeros)
{
    // A column with zero variance should produce near-zero output values.
    nn::transforms::AudioMeanStdNormalize norm;
    const auto train = constant_tensor(10, 3, 5.0F);
    norm.accumulate(train);
    norm.finalize();
    const auto out = norm(train);
    for (nn::Index i = 0; i < out.rows(); ++i)
        for (nn::Index j = 0; j < out.cols(); ++j) EXPECT_NEAR(out.at(i, j), 0.0F, 1e-4F);
}

// ─── Compose ─────────────────────────────────────────────────────────────────

TEST(Compose, AppliesTransformsInOrder)
{
    // Chain two EEGWindowZScore calls: the second one acts on already-normalized
    // data which has zero mean and unit std, so its output is also near unchanged.
    using namespace nn::transforms;
    const auto zscore1 = std::make_shared<EEGWindowZScore>();
    const auto zscore2 = std::make_shared<EEGWindowZScore>();
    Compose compose({zscore1, zscore2});

    const auto input = sequential_tensor(4, 8);
    const auto out = compose(input);

    EXPECT_EQ(out.rows(), input.rows());
    EXPECT_EQ(out.cols(), input.cols());
    // After two z-score passes each row is still unit-normalized.
    for (nn::Index i = 0; i < out.rows(); ++i)
        EXPECT_NEAR(row_mean(out, i), 0.0F, 1e-4F) << "row " << i;
}

TEST(Compose, EmptyStepsReturnsSameData)
{
    nn::transforms::Compose compose({});
    const auto input = sequential_tensor(3, 5);
    const auto out = compose(input);
    for (nn::Index i = 0; i < input.rows(); ++i)
        for (nn::Index j = 0; j < input.cols(); ++j) EXPECT_EQ(out.at(i, j), input.at(i, j));
}

// ─── FusedModalityTransform ───────────────────────────────────────────────────

TEST(FusedModalityTransform, EegColumnsAreZeroMeanPerRow)
{
    using namespace nn::transforms;

    // 3 EEG cols, 4 audio cols; fused tensor has 7 cols.
    constexpr nn::Index kEeg = 3, kAudio = 4;

    auto audio_norm = std::make_shared<AudioMeanStdNormalize>();
    const auto train_audio = sequential_tensor(10, kAudio, 1.0F);
    audio_norm->accumulate(train_audio);
    audio_norm->finalize();

    auto eeg_z = std::make_shared<EEGWindowZScore>();
    FusedModalityTransform fused(kEeg, kAudio, eeg_z, audio_norm);

    // Build fused input: EEG block followed by audio block.
    nn::Tensor input(5, kEeg + kAudio);
    for (nn::Index i = 0; i < 5; ++i)
    {
        for (nn::Index j = 0; j < kEeg; ++j) input.at(i, j) = static_cast<float>(i * kEeg + j);
        for (nn::Index j = 0; j < kAudio; ++j) input.at(i, kEeg + j) = static_cast<float>(j + 1);
    }

    const auto out = fused(input);
    EXPECT_EQ(out.rows(), 5u);
    EXPECT_EQ(out.cols(), kEeg + kAudio);

    // EEG block rows (cols 0..kEeg-1) should have zero mean.
    for (nn::Index i = 0; i < out.rows(); ++i)
    {
        float sum = 0.0F;
        for (nn::Index j = 0; j < kEeg; ++j) sum += out.at(i, j);
        EXPECT_NEAR(sum / static_cast<float>(kEeg), 0.0F, 1e-4F) << "EEG row " << i;
    }
}

TEST(FusedModalityTransform, NullTransformLeavesColumnRangeUnmodified)
{
    using namespace nn::transforms;

    constexpr nn::Index kEeg = 4, kAudio = 3;
    const auto input = sequential_tensor(6, kEeg + kAudio);

    // Pass nullptr for the EEG transform → EEG columns unchanged.
    auto audio_norm = std::make_shared<AudioMeanStdNormalize>();
    audio_norm->accumulate(input.block(0, kEeg, input.rows(), kAudio));
    audio_norm->finalize();

    FusedModalityTransform fused(kEeg, kAudio, nullptr, audio_norm);
    const auto out = fused(input);

    for (nn::Index i = 0; i < out.rows(); ++i)
        for (nn::Index j = 0; j < kEeg; ++j)
            EXPECT_EQ(out.at(i, j), input.at(i, j)) << "EEG col " << j << " row " << i;
}

// AudioMeanStdNormalize: empty batch early return (AudioMeanStdNormalize.hpp line 31)
TEST(AudioMeanStdNormalize, EmptyBatchIsIgnored)
{
    nn::transforms::AudioMeanStdNormalize norm;
    // accumulate with an empty tensor - should return early without side effects
    nn::Tensor empty_batch(0, 4);
    norm.accumulate(empty_batch);
    // Also accumulate a zero-col tensor
    nn::Tensor zero_col(4, 0);
    norm.accumulate(zero_col);
    // No crash = pass (early return path is covered)
}

// ─── RandomCrop ────────────────────────────────────────────────────────────────

TEST(RandomCrop, ThrowsOnNonColumnInput)
{
    nn::transforms::RandomCrop crop(4, /*seed=*/1);
    EXPECT_THROW(crop(sequential_tensor(10, 2)), std::invalid_argument);
}

TEST(RandomCrop, ThrowsWhenSignalShorterThanCropSize)
{
    nn::transforms::RandomCrop crop(10, /*seed=*/1);
    EXPECT_THROW(crop(sequential_tensor(4, 1)), std::invalid_argument);
}

TEST(RandomCrop, ReturnsUnchangedWhenSignalEqualsCropSize)
{
    nn::transforms::RandomCrop crop(6, /*seed=*/1);
    const auto input = sequential_tensor(6, 1);
    const auto out = crop(input);
    ASSERT_EQ(out.rows(), input.rows());
    for (nn::Index i = 0; i < input.rows(); ++i) EXPECT_EQ(out.at(i, 0), input.at(i, 0));
}

TEST(RandomCrop, CropIsContiguousSubrangeOfInput)
{
    // sequential_tensor(N, 1, base) is strictly increasing by 1 per row, so any
    // contiguous crop is identifiable by its first value and step.
    constexpr nn::Index kN = 40, kCrop = 5;
    nn::transforms::RandomCrop crop(kCrop, /*seed=*/7);
    const auto input = sequential_tensor(kN, 1, 100.0F);
    const auto out = crop(input);

    ASSERT_EQ(out.rows(), kCrop);
    ASSERT_EQ(out.cols(), 1u);
    const float first = out.at(0, 0);
    EXPECT_GE(first, 100.0F);
    EXPECT_LE(first, 100.0F + static_cast<float>(kN - kCrop));
    for (nn::Index i = 1; i < kCrop; ++i)
        EXPECT_FLOAT_EQ(out.at(i, 0), first + static_cast<float>(i));
}

TEST(RandomCrop, DeterministicForFixedSeed)
{
    const auto input = sequential_tensor(50, 1);
    nn::transforms::RandomCrop crop_a(8, /*seed=*/42);
    nn::transforms::RandomCrop crop_b(8, /*seed=*/42);
    const auto out_a = crop_a(input);
    const auto out_b = crop_b(input);
    for (nn::Index i = 0; i < out_a.rows(); ++i) EXPECT_EQ(out_a.at(i, 0), out_b.at(i, 0));
}

TEST(RandomCrop, DiversifiesOffsetAcrossCalls)
{
    // Same instance, many calls on the same (large) signal: the offset must not be
    // pinned to the same value every time -- this is the exact property that fixes
    // the "always window 0" AudioMNIST degeneracy when a dataset uses RandomCrop
    // instead of a fixed, deterministic slice.
    nn::transforms::RandomCrop crop(4, /*seed=*/123);
    const auto input = sequential_tensor(200, 1);

    std::vector<float> first_values;
    for (int i = 0; i < 30; ++i) first_values.push_back(crop(input).at(0, 0));

    const bool all_same = std::all_of(first_values.begin(),
        first_values.end(),
        [&](float v) { return v == first_values.front(); });
    EXPECT_FALSE(all_same) << "RandomCrop offset never changed across 30 calls";
}

// ─── RandomIndexCrop ─────────────────────────────────────────────────────────

TEST(RandomIndexCrop, PreservesMultisetOfIndices)
{
    nn::transforms::RandomIndexCrop crop(/*seed=*/1);
    const std::vector<std::size_t> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto out = crop(input);
    std::sort(out.begin(), out.end());
    EXPECT_EQ(out, input);
}

TEST(RandomIndexCrop, EmptyInputReturnsEmpty)
{
    nn::transforms::RandomIndexCrop crop(/*seed=*/1);
    EXPECT_TRUE(crop({}).empty());
}

TEST(RandomIndexCrop, DeterministicForFixedSeed)
{
    const std::vector<std::size_t> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    nn::transforms::RandomIndexCrop crop_a(/*seed=*/42);
    nn::transforms::RandomIndexCrop crop_b(/*seed=*/42);
    EXPECT_EQ(crop_a(input), crop_b(input));
}

TEST(RandomIndexCrop, DiversifiesOrderAcrossCalls)
{
    // Same instance, called once per "recording" (as stratified_window_cap does):
    // the draw sequence must not degenerate into the same order every time.
    nn::transforms::RandomIndexCrop crop(/*seed=*/7);
    const std::vector<std::size_t> input = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<std::size_t> first_elements;
    for (int i = 0; i < 30; ++i) first_elements.push_back(crop(input).front());

    const bool all_same = std::all_of(first_elements.begin(),
        first_elements.end(),
        [&](std::size_t v) { return v == first_elements.front(); });
    EXPECT_FALSE(all_same) << "RandomIndexCrop order never changed across 30 calls";
}

// ─── WindowZScore ────────────────────────────────────────────────────────────

TEST(WindowZScore, OutputShapePreserved)
{
    nn::transforms::WindowZScore zscore;
    const auto input = sequential_tensor(32, 1);
    const auto out = zscore(input);
    EXPECT_EQ(out.rows(), input.rows());
    EXPECT_EQ(out.cols(), input.cols());
}

TEST(WindowZScore, WholeTensorZeroMeanUnitStd)
{
    // Whole-tensor convention (not per-row like EEGWindowZScore): one mean/std over
    // every element, matching nn::utility::zscore_inplace and this framework's
    // (window_size, 1) single-channel window shape.
    nn::transforms::WindowZScore zscore;
    const auto input = sequential_tensor(64, 1, -10.0F);
    const auto out = zscore(input);

    float sum = 0.0F, sq_sum = 0.0F;
    for (nn::Index i = 0; i < out.rows(); ++i)
    {
        sum += out.at(i, 0);
        sq_sum += out.at(i, 0) * out.at(i, 0);
    }
    const float mean = sum / static_cast<float>(out.rows());
    const float var = sq_sum / static_cast<float>(out.rows()) - mean * mean;
    EXPECT_NEAR(mean, 0.0F, 1e-4F);
    EXPECT_NEAR(std::sqrt(var), 1.0F, 1e-3F);
}

TEST(WindowZScore, ConstantInputProducesZeros)
{
    nn::transforms::WindowZScore zscore;
    const auto input = constant_tensor(16, 1, 3.0F);
    const auto out = zscore(input);
    for (nn::Index i = 0; i < out.rows(); ++i) EXPECT_NEAR(out.at(i, 0), 0.0F, 1e-3F);
}

TEST(WindowZScore, DoesNotMutateInput)
{
    // Unlike the zscore_inplace() it wraps, the transform is value-in/value-out --
    // callers that only hold `const Tensor&` must see it unchanged after the call.
    nn::transforms::WindowZScore zscore;
    const auto input = sequential_tensor(10, 1, 5.0F);
    const auto before = input;
    (void) zscore(input);
    for (nn::Index i = 0; i < input.rows(); ++i) EXPECT_EQ(input.at(i, 0), before.at(i, 0));
}

// ─── GaussianNoise ───────────────────────────────────────────────────────────

TEST(GaussianNoise, NegativeStdThrows)
{
    EXPECT_THROW(nn::transforms::GaussianNoise(-1.0F, /*seed=*/1), std::invalid_argument);
}

TEST(GaussianNoise, ZeroStdIsExactPassThrough)
{
    // std::normal_distribution at stddev=0 is a standard-library precondition
    // violation (UB) -- operator() must short-circuit to identity instead of
    // drawing from a zero-stddev distribution. This is also the disabled/default
    // state (denoising_noise_std=0.0f in Meeting01Config).
    nn::transforms::GaussianNoise noise(0.0F, /*seed=*/1);
    const auto input = sequential_tensor(32, 1, -5.0F);
    const auto out = noise(input);
    for (nn::Index i = 0; i < input.rows(); ++i) EXPECT_EQ(out.at(i, 0), input.at(i, 0));
}

TEST(GaussianNoise, OutputShapePreserved)
{
    nn::transforms::GaussianNoise noise(0.5F, /*seed=*/1);
    const auto input = sequential_tensor(32, 1);
    const auto out = noise(input);
    EXPECT_EQ(out.rows(), input.rows());
    EXPECT_EQ(out.cols(), input.cols());
}

TEST(GaussianNoise, DoesNotMutateInput)
{
    nn::transforms::GaussianNoise noise(0.5F, /*seed=*/1);
    const auto input = sequential_tensor(10, 1, 5.0F);
    const auto before = input;
    (void) noise(input);
    for (nn::Index i = 0; i < input.rows(); ++i) EXPECT_EQ(input.at(i, 0), before.at(i, 0));
}

TEST(GaussianNoise, DeterministicForFixedSeed)
{
    const auto input = sequential_tensor(64, 1);
    nn::transforms::GaussianNoise noise_a(0.5F, /*seed=*/42);
    nn::transforms::GaussianNoise noise_b(0.5F, /*seed=*/42);
    const auto out_a = noise_a(input);
    const auto out_b = noise_b(input);
    for (nn::Index i = 0; i < input.rows(); ++i) EXPECT_EQ(out_a.at(i, 0), out_b.at(i, 0));
}

TEST(GaussianNoise, DiversifiesAcrossCalls)
{
    // Stateful RNG (matches RandomCrop/RandomIndexCrop): a second call on the same
    // instance must not repeat the first call's draw -- required for per-run noise
    // fixed across epochs but still i.i.d. per sample within `make_triples`.
    nn::transforms::GaussianNoise noise(0.5F, /*seed=*/7);
    const auto input = constant_tensor(8, 1, 0.0F);
    const auto out_1 = noise(input);
    const auto out_2 = noise(input);

    bool any_differs = false;
    for (nn::Index i = 0; i < input.rows(); ++i)
        if (out_1.at(i, 0) != out_2.at(i, 0)) any_differs = true;
    EXPECT_TRUE(any_differs) << "GaussianNoise draw identical across consecutive calls";
}

TEST(GaussianNoise, PerturbsWithApproximatelyCorrectStd)
{
    // Large constant input -> output deviations ARE the noise draws; check their
    // empirical std lands near the configured std (loose tolerance, statistical).
    constexpr float kStd = 2.0F;
    nn::transforms::GaussianNoise noise(kStd, /*seed=*/99);
    const auto input = constant_tensor(4000, 1, 10.0F);
    const auto out = noise(input);

    float sq_sum = 0.0F;
    for (nn::Index i = 0; i < out.rows(); ++i)
    {
        const float d = out.at(i, 0) - input.at(i, 0);
        sq_sum += d * d;
    }
    const float empirical_std = std::sqrt(sq_sum / static_cast<float>(out.rows()));
    EXPECT_NEAR(empirical_std, kStd, 0.15F);
}
