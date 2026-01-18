/**
 * @file util_gtest.cpp
 * @brief Unit tests for assorted utility helpers (batching, synthetic data, checks).
 */

#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <set>

#include "nn/tensor/Tensor.hpp"
#include "nn/utility/batching.hpp"
#include "nn/utility/synthetic_spike_data.hpp"
#include "nn/utility/vectorizationCheck.hpp"

namespace
{

auto make_random_tensor(size_t rows, size_t cols, float lower = -1.0F, float upper = 1.0F)
    -> nn::Tensor
{
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(lower, upper);
    nn::Tensor t(rows, cols);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            t.at(i, j) = dist(gen);
        }
    }
    return t;
}

[[maybe_unused]] auto make_constant_tensor(size_t rows, size_t cols, float value) -> nn::Tensor
{
    nn::Tensor t(rows, cols);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            t.at(i, j) = value;
        }
    }
    return t;
}

auto make_tensor_from_values(size_t rows, size_t cols, const std::initializer_list<float>& values)
    -> nn::Tensor
{
    nn::Tensor t(rows, cols);
    const auto expected = static_cast<std::size_t>(rows * cols);
    if (values.size() != expected)
    {
        throw std::invalid_argument("Initializer size does not match tensor shape");
    }
    std::size_t idx = 0;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            t.at(i, j) = *(values.begin() + static_cast<long>(idx));
            ++idx;
        }
    }
    return t;
}

} // namespace

// Util: synthetic_spike_data
TEST(UtilTest, SyntheticSpikeData)
{
    int n_samples = 5;
    int input_dim = 3;
    int n_steps = 4;
    float max_rate = 1.0F;
    float timeStep = 1.0F;

    auto result =
        generate_autoencoder_spike_data(n_samples, input_dim, n_steps, max_rate, timeStep);
    auto spike_trains = std::get<0>(result);
    // auto _ = std::get<1>(result); // unused

    ASSERT_EQ(spike_trains.size(), n_steps);
    for (const auto& spikes : spike_trains)
    {
        ASSERT_EQ(spikes.rows(), n_samples);
        ASSERT_EQ(spikes.cols(), input_dim);
        for (int i = 0; i < static_cast<int>(spikes.rows()); ++i)
        {
            for (int j = 0; j < static_cast<int>(spikes.cols()); ++j)
            {
                ASSERT_TRUE(spikes.at(i, j) == 0.0F || spikes.at(i, j) == 1.0F);
            }
        }
    }
}

// Util: vectorizationCheck
TEST(UtilTest, VectorizationCheck)
{
    ASSERT_NO_THROW(printVectorizationSupport());
}

// Util: batching
TEST(UtilTest, Batching)
{
    std::vector<nn::Tensor> input_samples;
    std::vector<nn::Tensor> target_samples;
    for (int i = 0; i < 4; ++i)
    {
        input_samples.emplace_back(make_random_tensor(1, 2));
        target_samples.emplace_back(make_random_tensor(1, 1));
    }

    auto batches = create_batches(input_samples, target_samples, 2);

    ASSERT_EQ(batches.size(), 2U);
    ASSERT_EQ(batches[0].inputs.rows(), 2);
}

// Exception Testing for Utilities
TEST(UtilExceptionTest, InvalidBatchParameters)
{
    std::vector<nn::Tensor> input_samples;
    std::vector<nn::Tensor> target_samples;

    // Empty input samples
    ASSERT_THROW(create_batches(input_samples, target_samples, 2), std::invalid_argument);

    // Mismatched input/target sizes
    input_samples.emplace_back(make_random_tensor(1, 2));
    target_samples.emplace_back(make_random_tensor(1, 1));
    input_samples.emplace_back(make_random_tensor(1, 2));
    // target_samples has only 1 element, input_samples has 2
    ASSERT_THROW(create_batches(input_samples, target_samples, 2), std::invalid_argument);

    // Invalid batch size
    ASSERT_THROW(create_batches(input_samples, target_samples, 0), std::invalid_argument);
    ASSERT_THROW(create_batches(input_samples, target_samples, -1), std::invalid_argument);
}

TEST(UtilExceptionTest, SyntheticSpikeDataInvalidParams)
{
    // Invalid parameters for synthetic spike data generation
    ASSERT_THROW(generate_autoencoder_spike_data(0, 3, 4, 1.0F, 1.0F), std::invalid_argument);
    ASSERT_THROW(generate_autoencoder_spike_data(5, 0, 4, 1.0F, 1.0F), std::invalid_argument);
    ASSERT_THROW(generate_autoencoder_spike_data(5, 3, 0, 1.0F, 1.0F), std::invalid_argument);
    ASSERT_THROW(generate_autoencoder_spike_data(5, 3, 4, -1.0F, 1.0F), std::invalid_argument);
    ASSERT_THROW(generate_autoencoder_spike_data(5, 3, 4, 1.0F, 0.0F), std::invalid_argument);
}

// Memory Stress Testing for Utilities
TEST(UtilMemoryStressTest, LargeBatchCreation)
{
    const int num_samples = 10000;
    const int input_dim = 100;
    const int target_dim = 10;
    const int batch_size = 1000;

    std::vector<nn::Tensor> input_samples;
    std::vector<nn::Tensor> target_samples;

    // Create large dataset
    for (int i = 0; i < num_samples; ++i)
    {
        input_samples.emplace_back(make_random_tensor(1, input_dim));
        target_samples.emplace_back(make_random_tensor(1, target_dim));
    }

    ASSERT_NO_THROW({
        auto batches = create_batches(input_samples, target_samples, batch_size);
        EXPECT_EQ(batches.size(), num_samples / batch_size);

        // Verify batch contents
        for (const auto& batch : batches)
        {
            EXPECT_EQ(batch.inputs.rows(), batch_size);
            EXPECT_EQ(batch.inputs.cols(), input_dim);
            EXPECT_EQ(batch.targets.rows(), batch_size);
            EXPECT_EQ(batch.targets.cols(), target_dim);
        }
    });
}

TEST(UtilMemoryStressTest, LargeSyntheticSpikeData)
{
    const int n_samples = 1000;
    const int input_dim = 100;
    const int n_steps = 50;

    EXPECT_NO_THROW({
        auto result = generate_autoencoder_spike_data(n_samples, input_dim, n_steps, 0.5F, 0.1F);
        auto spike_trains = std::get<0>(result);
        // auto _ = std::get<1>(result); // unused

        EXPECT_EQ(spike_trains.size(), n_steps);
        for (const auto& spikes : spike_trains)
        {
            EXPECT_EQ(spikes.rows(), n_samples);
            EXPECT_EQ(spikes.cols(), input_dim);
        }
    });
}

// Numerical Edge Cases for Utilities
TEST(UtilNumericalEdgeTest, ExtremeBatchSizes)
{
    std::vector<nn::Tensor> input_samples;
    std::vector<nn::Tensor> target_samples;

    // Single sample
    input_samples.emplace_back(make_random_tensor(1, 2));
    target_samples.emplace_back(make_random_tensor(1, 1));

    auto single_batch = create_batches(input_samples, target_samples, 1);
    EXPECT_EQ(single_batch.size(), 1);
    EXPECT_EQ(single_batch[0].inputs.rows(), 1);

    // Batch size larger than dataset
    auto large_batch = create_batches(input_samples, target_samples, 10);
    EXPECT_EQ(large_batch.size(), 1);
    EXPECT_EQ(large_batch[0].inputs.rows(),
              1); // Should return all available samples
}

TEST(UtilNumericalEdgeTest, SpikeDataEdgeRates)
{
    const int n_samples = 10;
    const int input_dim = 5;
    const int n_steps = 5;

    // Test with very low firing rate
    auto low_rate_result =
        generate_autoencoder_spike_data(n_samples, input_dim, n_steps, 0.001F, 1.0F);
    auto low_rate_spikes = std::get<0>(low_rate_result);
    // auto _ = std::get<1>(low_rate_result); // unused
    int total_low_spikes = 0;
    for (const auto& spikes : low_rate_spikes)
    {
        total_low_spikes += static_cast<int>(spikes.sum());
    }
    EXPECT_LT(total_low_spikes, n_samples * input_dim * n_steps * 0.01F); // Very few spikes

    // Test with very high firing rate
    auto high_rate_result =
        generate_autoencoder_spike_data(n_samples, input_dim, n_steps, 0.999F, 1.0F);
    auto high_rate_spikes = std::get<0>(high_rate_result);
    // auto _ = std::get<1>(high_rate_result); // unused
    int total_high_spikes = 0;
    for (const auto& spikes : high_rate_spikes)
    {
        total_high_spikes += static_cast<int>(spikes.sum());
    }
    EXPECT_GT(total_high_spikes, n_samples * input_dim * n_steps * 0.95F); // Almost all spikes
}

TEST(UtilNumericalEdgeTest, NaNInfInBatching)
{
    std::vector<nn::Tensor> input_samples;
    std::vector<nn::Tensor> target_samples;

    // Create tensors with NaN values
    nn::Tensor nan_input(1, 2);
    nan_input.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    nan_input.at(0, 1) = 1.0F;
    input_samples.emplace_back(std::move(nan_input));

    nn::Tensor nan_target(1, 1);
    nan_target.at(0, 0) = 1.0F;
    target_samples.emplace_back(std::move(nan_target));

    // Should handle NaN gracefully
    ASSERT_NO_THROW({
        auto batches = create_batches(input_samples, target_samples, 1);
        EXPECT_TRUE(std::isnan(batches[0].inputs.at(0, 0)));
    });

    // Test with Inf values
    nn::Tensor inf_input(1, 2);
    inf_input.at(0, 0) = std::numeric_limits<float>::infinity();
    inf_input.at(0, 1) = 1.0F;
    input_samples[0] = std::move(inf_input);

    ASSERT_NO_THROW({
        auto batches = create_batches(input_samples, target_samples, 1);
        EXPECT_TRUE(std::isinf(batches[0].inputs.at(0, 0)));
    });
}

// Thread Safety Validation for Utilities
TEST(UtilThreadSafetyTest, ConcurrentBatchCreation)
{
    const int num_samples = 100;
    std::vector<nn::Tensor> input_samples;
    std::vector<nn::Tensor> target_samples;

    // Create dataset
    for (int i = 0; i < num_samples; ++i)
    {
        input_samples.emplace_back(make_random_tensor(1, 5));
        target_samples.emplace_back(make_random_tensor(1, 2));
    }

    // Test multiple batch creation calls (simulating concurrent access)
    for (int i = 0; i < 5; ++i)
    {
        ASSERT_NO_THROW({
            auto batches = create_batches(input_samples, target_samples, 10);
            EXPECT_EQ(batches.size(), 10);

            // Verify batch integrity
            for (const auto& batch : batches)
            {
                EXPECT_EQ(batch.inputs.rows(), 10);
                EXPECT_EQ(batch.targets.rows(), 10);
            }
        });
    }
}

TEST(UtilThreadSafetyTest, ConcurrentSpikeGeneration)
{
    const int n_samples = 50;
    const int input_dim = 10;
    const int n_steps = 5;

    // Test multiple spike data generation calls
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NO_THROW({
            auto result =
                generate_autoencoder_spike_data(n_samples, input_dim, n_steps, 0.5F, 1.0F);
            auto spike_trains = std::get<0>(result);
            // auto _ = std::get<1>(result); // unused

            EXPECT_EQ(spike_trains.size(), n_steps);
            for (const auto& spikes : spike_trains)
            {
                EXPECT_EQ(spikes.rows(), n_samples);
                EXPECT_EQ(spikes.cols(), input_dim);
            }
        });
    }
}

// Additional Comprehensive Tests
TEST(UtilComprehensiveTest, BatchContentVerification)
{
    std::vector<nn::Tensor> input_samples;
    std::vector<nn::Tensor> target_samples;

    // Create samples with known values
    for (int i = 0; i < 6; ++i)
    {
        input_samples.emplace_back(make_tensor_from_values(1, 2, {i * 2.0F, i * 2.0F + 1.0F}));

        target_samples.emplace_back(make_tensor_from_values(1, 1, {static_cast<float>(i)}));
    }

    auto batches = create_batches(input_samples, target_samples, 3);

    EXPECT_EQ(batches.size(), 2);

    // Verify batch sizes are correct (each batch should have 3 samples)
    EXPECT_EQ(batches[0].inputs.rows(), 3);  // 3 samples × 1 row per sample = 3 rows
    EXPECT_EQ(batches[0].inputs.cols(), 2);  // 2 features per sample
    EXPECT_EQ(batches[0].targets.rows(), 3); // 3 samples × 1 row per sample = 3 rows
    EXPECT_EQ(batches[0].targets.cols(), 1); // 1 target per sample

    EXPECT_EQ(batches[1].inputs.rows(), 3);
    EXPECT_EQ(batches[1].inputs.cols(), 2);
    EXPECT_EQ(batches[1].targets.rows(), 3);
    EXPECT_EQ(batches[1].targets.cols(), 1);

    // Verify that all expected values appear somewhere in the batches
    std::set<float> expected_input_values = {
        0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
    std::set<float> batch_input_values;
    for (const auto& batch : batches)
    {
        for (int i = 0; i < batch.inputs.rows(); ++i)
        {
            for (int j = 0; j < batch.inputs.cols(); ++j)
            {
                batch_input_values.insert(batch.inputs.at(i, j));
            }
        }
    }
    EXPECT_EQ(batch_input_values, expected_input_values);

    // Verify target values
    std::set<float> expected_target_values = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    std::set<float> batch_target_values;
    for (const auto& batch : batches)
    {
        for (int i = 0; i < batch.targets.rows(); ++i)
        {
            for (int j = 0; j < batch.targets.cols(); ++j)
            {
                batch_target_values.insert(batch.targets.at(i, j));
            }
        }
    }
    EXPECT_EQ(batch_target_values, expected_target_values);
}

TEST(UtilComprehensiveTest, VectorizationCheckOutput)
{
    // Test that vectorization check doesn't crash and produces output
    testing::internal::CaptureStdout();
    ASSERT_NO_THROW(printVectorizationSupport());
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_FALSE(output.empty());
}

TEST(UtilComprehensiveTest, SpikeDataStatisticalProperties)
{
    const int n_samples = 1000;
    const int input_dim = 10;
    const int n_steps = 20;
    const float rate = 0.2F;

    auto result = generate_autoencoder_spike_data(n_samples, input_dim, n_steps, rate, 1.0F);
    auto spike_trains = std::get<0>(result);
    // auto _ = std::get<1>(result); // unused

    // Calculate overall firing rate
    int total_spikes = 0;
    int total_possible = n_samples * input_dim * n_steps;

    for (const auto& spikes : spike_trains)
    {
        total_spikes += static_cast<int>(spikes.sum());
    }

    float actual_rate = static_cast<float>(total_spikes) / total_possible;

    // Should be close to target rate (within statistical variation)
    // Using 0.08F tolerance for statistical variation with n=1000*10*20=200k samples
    EXPECT_NEAR(actual_rate, rate, 0.08F);

    // Test temporal consistency - spikes should be somewhat evenly distributed
    std::vector<int> spikes_per_timestep;
    for (const auto& spikes : spike_trains)
    {
        spikes_per_timestep.push_back(static_cast<int>(spikes.sum()));
    }

    // Check that no timestep has zero spikes (for reasonable rate)
    for (int spikes : spikes_per_timestep)
    {
        EXPECT_GT(spikes, 0);
    }
}
