/**
 * @file src/experiments/03/tests/ProfileAndResults_gtest.cpp
 * @brief Implementation for Profileandresults gtest.
 *

 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "ProfileLoader.hpp"
#include "ResultsWriter.hpp"
#include "cli.hpp"

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
    EXPECT_EQ(config.dataset_subject_filter_regex, "^S(\\d+)$");
    EXPECT_EQ(config.dataset_root_path,
        "/home/ensismoebius/Documentos/UNESP/doutorado/databases/BaseDeDatosHablaImaginada");
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

TEST(Experiment03ProfilesTest, LoadsDeviceOverride)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_device_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"device\": \"cpu\"\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(config.device, "cpu");

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ProfilesTest, RejectsUnknownTopLevelKey)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_unknown_key_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"audio-window\",\n"
               "  \"autoencoder_type\": \"audio-window-ann\",\n"
               "  \"batch_size\": 32\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_FALSE(ok);
    EXPECT_NE(error.find("unknown profile key(s):"), std::string::npos);
    EXPECT_NE(error.find("batch_size"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03CliTest, ParsesDeviceOverride)
{
    const auto dataset_root = std::filesystem::temp_directory_path();

    Config defaults{};
    defaults.profile_name = "";
    defaults.device = "opencl";
    defaults.dataset_subject_filter_regex = "^S(\\d+)$";
    defaults.dataset_root_path = dataset_root.string();
    defaults.training_batch_size = 10;
    defaults.training_max_batches_per_epoch = 0;
    defaults.sampler_shuffle_samples = true;
    defaults.sampler_shuffle_seed = 42U;
    defaults.sampler_default_type = "";
    defaults.sampler_weights = {};
    defaults.sampler_weighted_num_samples = std::nullopt;
    defaults.sampler_distributed_num_replicas = 1;
    defaults.sampler_distributed_rank = 0;
    defaults.sampler_distributed_shuffle = true;
    defaults.sampler_distributed_drop_last = false;
    defaults.dataset_input_mode = Protocol101117InputMode::Concatenated;
    defaults.dataset_type = Experiment03DatasetType::FusedWindow;
    defaults.autoencoder_type = Experiment03AutoencoderType::FusedWindowAnn;
    defaults.autoencoder_hidden_size = 64;
    defaults.autoencoder_latent_size = 32;
    defaults.autoencoder_depth = 2;
    defaults.autoencoder_layer_sizes = {};
    defaults.autoencoder_input_features = 0;
    defaults.autoencoder_eeg_features = 0;
    defaults.autoencoder_audio_features = 0;
    defaults.autoencoder_architecture = AutoencoderArchitecture::Auto;
    defaults.autoencoder_branch_hidden_size = 0;
    defaults.autoencoder_fusion_hidden_size = 0;
    defaults.autoencoder_residual_blocks = 1;
    defaults.autoencoder_time_step = 1.0F;
    defaults.autoencoder_resistance = 1.0F;
    defaults.autoencoder_capacitance = 1.0F;
    defaults.training_optimizer_type = "adam";
    defaults.training_learning_rate = 1e-2F;
    defaults.training_optimizer_momentum = 0.0F;
    defaults.training_optimizer_adam_beta1 = 0.9F;
    defaults.training_optimizer_adam_beta2 = 0.999F;
    defaults.training_optimizer_adam_epsilon = 1e-8F;
    defaults.training_epochs = 1;
    defaults.window_eeg_config = {.window_size = 256, .overlap = 0.5F, .sample_rate = 1024};
    defaults.window_audio_config = {.window_size = 11025, .overlap = 0.5F, .sample_rate = 44100};
    defaults.prefetch_lookahead = 1;
    defaults.prefetch_ram_cap_mb = 64;

    std::string arg0 = "experiment03";
    std::string arg1 = "--dataset-root";
    std::string arg2 = dataset_root.string();
    std::string arg3 = "--device";
    std::string arg4 = "cpu";
    char* argv[] = {arg0.data(), arg1.data(), arg2.data(), arg3.data(), arg4.data()};

    const Config parsed = parseCliParams(5, argv, defaults);

    EXPECT_EQ(parsed.device, "cpu");
}

TEST(Experiment03ProfilesTest, LoadsOptimizerOverrides)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_optimizer_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"training_optimizer_type\": \"sgd\",\n"
               "  \"training_learning_rate\": 0.02,\n"
               "  \"training_optimizer_momentum\": 0.7\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(config.training_optimizer_type, "sgd");
    EXPECT_FLOAT_EQ(config.training_learning_rate, 0.02F);
    EXPECT_FLOAT_EQ(config.training_optimizer_momentum, 0.7F);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03CliTest, ParsesOptimizerOverride)
{
    const auto dataset_root = std::filesystem::temp_directory_path();

    Config defaults{};
    defaults.profile_name = "";
    defaults.device = "opencl";
    defaults.dataset_subject_filter_regex = "^S(\\d+)$";
    defaults.dataset_root_path = dataset_root.string();
    defaults.training_batch_size = 10;
    defaults.training_max_batches_per_epoch = 0;
    defaults.sampler_shuffle_samples = true;
    defaults.sampler_shuffle_seed = 42U;
    defaults.sampler_default_type = "";
    defaults.sampler_weights = {};
    defaults.sampler_weighted_num_samples = std::nullopt;
    defaults.sampler_distributed_num_replicas = 1;
    defaults.sampler_distributed_rank = 0;
    defaults.sampler_distributed_shuffle = true;
    defaults.sampler_distributed_drop_last = false;
    defaults.dataset_input_mode = Protocol101117InputMode::Concatenated;
    defaults.dataset_type = Experiment03DatasetType::FusedWindow;
    defaults.autoencoder_type = Experiment03AutoencoderType::FusedWindowAnn;
    defaults.autoencoder_hidden_size = 64;
    defaults.autoencoder_latent_size = 32;
    defaults.autoencoder_depth = 2;
    defaults.autoencoder_layer_sizes = {};
    defaults.autoencoder_input_features = 0;
    defaults.autoencoder_eeg_features = 0;
    defaults.autoencoder_audio_features = 0;
    defaults.autoencoder_architecture = AutoencoderArchitecture::Auto;
    defaults.autoencoder_branch_hidden_size = 0;
    defaults.autoencoder_fusion_hidden_size = 0;
    defaults.autoencoder_residual_blocks = 1;
    defaults.autoencoder_time_step = 1.0F;
    defaults.autoencoder_resistance = 1.0F;
    defaults.autoencoder_capacitance = 1.0F;
    defaults.training_optimizer_type = "adam";
    defaults.training_learning_rate = 1e-2F;
    defaults.training_optimizer_momentum = 0.0F;
    defaults.training_optimizer_adam_beta1 = 0.9F;
    defaults.training_optimizer_adam_beta2 = 0.999F;
    defaults.training_optimizer_adam_epsilon = 1e-8F;
    defaults.training_epochs = 1;
    defaults.window_eeg_config = {.window_size = 256, .overlap = 0.5F, .sample_rate = 1024};
    defaults.window_audio_config = {.window_size = 11025, .overlap = 0.5F, .sample_rate = 44100};
    defaults.prefetch_lookahead = 1;
    defaults.prefetch_ram_cap_mb = 64;

    std::string arg0 = "experiment03";
    std::string arg1 = "--dataset-root";
    std::string arg2 = dataset_root.string();
    std::string arg3 = "--optimizer";
    std::string arg4 = "sgd";
    std::string arg5 = "--optimizer-momentum";
    std::string arg6 = "0.8";
    char* argv[] = {
        arg0.data(), arg1.data(), arg2.data(), arg3.data(), arg4.data(), arg5.data(), arg6.data()};

    const Config parsed = parseCliParams(7, argv, defaults);

    EXPECT_EQ(parsed.training_optimizer_type, "sgd");
    EXPECT_FLOAT_EQ(parsed.training_optimizer_momentum, 0.8F);
}

TEST(Experiment03ResultsWriterTest, WritesSummaryJson)
{
    Summary summary{};
    summary.profile_name = "test-profile";
    summary.dataset_type = "fused-window";
    summary.autoencoder_type = "fused-window-ann";
    summary.optimizer_type = "adam";
    summary.optimizer_learning_rate = 1e-4F;
    summary.optimizer_momentum = 0.0F;
    summary.optimizer_adam_beta1 = 0.9F;
    summary.optimizer_adam_beta2 = 0.999F;
    summary.optimizer_adam_epsilon = 1e-8F;
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
    EXPECT_NE(body.find("\"optimizer\":"), std::string::npos);
    EXPECT_NE(body.find("\"type\": \"adam\""), std::string::npos);
    EXPECT_NE(body.find("\"epoch_mean_losses\": [1, 0.9]"), std::string::npos);
}
