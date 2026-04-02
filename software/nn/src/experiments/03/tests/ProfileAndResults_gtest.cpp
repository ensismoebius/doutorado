#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "ProfileLoader.hpp"
#include "ResultsWriter.hpp"

using experiment03::Summary;

TEST(Experiment03ProfilesTest, LoadsDefaultProfile)
{
    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config("default", config, error);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(config.autoencoder_hidden_size, 64);
    EXPECT_EQ(config.autoencoder_latent_size, 32);
    EXPECT_EQ(config.autoencoder_depth, 2);
}

TEST(Experiment03ProfilesTest, LoadsProfileFromAbsolutePath)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_abs_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"audio-window\",\n"
               "  \"autoencoder_type\": \"audio-window-ann\",\n"
               "  \"autoencoder_hidden_size\": 64\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(config.dataset_type, Experiment03DatasetType::AudioWindow);
    EXPECT_EQ(config.autoencoder_type, Experiment03AutoencoderType::AudioWindowAnn);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ProfilesTest, LoadsInputAndLayerOverrides)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_inputs_layers_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"autoencoder_input_features\": 777,\n"
               "  \"eeg_features\": 333,\n"
               "  \"audio_features\": 444,\n"
               "  \"layers\": 5\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(config.autoencoder_input_features, 777);
    EXPECT_EQ(config.autoencoder_eeg_features, 333);
    EXPECT_EQ(config.autoencoder_audio_features, 444);
    EXPECT_EQ(config.autoencoder_depth, 5);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ResultsWriterTest, WritesSummaryJson)
{
    Summary summary{};
    summary.profile_name = "test-profile";
    summary.dataset_type = "fused-window";
    summary.autoencoder_type = "fused-window-ann";
    summary.exit_code = 0;
    summary.total_samples = 100;
    summary.processed_samples = 20;
    summary.seen_batches = 5;
    summary.epoch_mean_losses = {1.0F, 0.9F};

    std::string out_path;
    std::string out_error;

    const bool ok = experiment03::write_run_summary_json(summary, out_path, out_error);

    ASSERT_TRUE(ok) << out_error;
    ASSERT_FALSE(out_path.empty());
    ASSERT_TRUE(std::filesystem::exists(out_path));

    std::ifstream ifs(out_path);
    ASSERT_TRUE(ifs.good());
    std::string body((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    EXPECT_NE(body.find("\"profile\": \"test-profile\""), std::string::npos);
    EXPECT_NE(body.find("\"epoch_mean_losses\": [1, 0.9]"), std::string::npos);
}
