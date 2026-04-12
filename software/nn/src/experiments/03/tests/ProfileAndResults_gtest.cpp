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
               "  \"neural_network_type\": \"audio-window-ann\",\n"
               "  \"neural_network_hidden_size\": 64\n"
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
               "  \"neural_network_input_features\": 777,\n"
               "  \"neural_network_eeg_features\": 333,\n"
               "  \"neural_network_audio_features\": 444,\n"
               "  \"neural_network_depth\": 5\n"
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
               "  \"program_device\": \"cpu\"\n"
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
               "  \"neural_network_type\": \"audio-window-ann\",\n"
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

TEST(Experiment03ProfilesTest, RejectsLegacyProfileKeys)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_legacy_key_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"audio-window\",\n"
               "  \"autoencoder_type\": \"audio-window-ann\"\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_FALSE(ok);
    EXPECT_NE(error.find("unknown profile key(s):"), std::string::npos);
    EXPECT_NE(error.find("autoencoder_type"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03CliTest, LoadsDeviceFromSelectedProfile)
{
    const auto dataset_root = std::filesystem::temp_directory_path();
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_cli_device_profile_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"program_device\": \"cpu\"\n"
               "}\n";
    }

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
    std::string arg1 = "--profile";
    std::string arg2 = profile_path.string();
    char* argv[] = {arg0.data(), arg1.data(), arg2.data()};

    const Config parsed = parseCliParams(3, argv, defaults);

    EXPECT_EQ(parsed.device, "cpu");

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
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

TEST(Experiment03ProfilesTest, LoadsDeclarativeArchitectureAndTrainingOverrides)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_declarative_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"neural_network_layer\": [\n"
               "    \"encoder:linear:64:relu\",\n"
               "    \"encoder:linear:latent:identity\",\n"
               "    \"decoder:linear:64:relu\",\n"
               "    \"decoder:linear:output:identity\",\n"
               "    \"branch_encoder:linear:branch_hidden:relu\",\n"
               "    \"fusion_decoder:linear:output:identity\"\n"
               "  ],\n"
               "  \"training_optimizer_type\": \"sgd\",\n"
               "  \"training_loss_type\": \"mae\",\n"
               "  \"training_lr_plateau_enabled\": true,\n"
               "  \"training_lr_plateau_factor\": 0.25,\n"
               "  \"training_lr_plateau_patience\": 4,\n"
               "  \"training_lr_plateau_min_delta\": 0.0005,\n"
               "  \"validation_modality_diagnostics_enabled\": true\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(config.autoencoder_encoder_layer_spec.size(), 2U);
    EXPECT_EQ(config.autoencoder_encoder_layer_spec[0], "linear:64:relu");
    EXPECT_EQ(config.autoencoder_encoder_layer_spec[1], "linear:latent:identity");
    EXPECT_EQ(config.autoencoder_decoder_layer_spec.size(), 2U);
    EXPECT_EQ(config.autoencoder_decoder_layer_spec[1], "linear:output:identity");
    EXPECT_EQ(config.autoencoder_branch_encoder_layer_spec.size(), 1U);
    EXPECT_EQ(config.autoencoder_branch_encoder_layer_spec[0], "linear:branch_hidden:relu");
    EXPECT_EQ(config.autoencoder_fusion_decoder_layer_spec.size(), 1U);
    EXPECT_EQ(config.autoencoder_fusion_decoder_layer_spec[0], "linear:output:identity");
    EXPECT_EQ(config.training_optimizer_type, "sgd");
    EXPECT_EQ(config.training_loss_type, "mae");
    EXPECT_TRUE(config.training_lr_plateau_enabled);
    EXPECT_FLOAT_EQ(config.training_lr_plateau_factor, 0.25F);
    EXPECT_EQ(config.training_lr_plateau_patience, 4U);
    EXPECT_FLOAT_EQ(config.training_lr_plateau_min_delta, 0.0005F);
    EXPECT_TRUE(config.validation_modality_diagnostics_enabled);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ProfilesTest, RejectsSplitLayerSpecKeys)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_split_layer_keys_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"neural_network_decoder_layer_spec\": [\"linear:output:identity\"]\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_FALSE(ok);
    EXPECT_NE(error.find("unknown profile key(s):"), std::string::npos);
    EXPECT_NE(error.find("neural_network_decoder_layer_spec"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ProfilesTest, RejectsLegacyNeuralNetworkFamilyAlias)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_family_alias_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"audio-window\",\n"
               "  \"neural_network_family\": \"snn\"\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_FALSE(ok);
    EXPECT_NE(error.find("unknown profile key(s):"), std::string::npos);
    EXPECT_NE(error.find("neural_network_family"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ProfilesTest, RejectsShortNeuralNetworkTypeAlias)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_type_alias_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"protocol\",\n"
               "  \"neural_network_type\": \"ann\"\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_FALSE(ok);
    EXPECT_NE(error.find("unsupported neural_network_type"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ProfilesTest, RejectsUnsupportedAutoencoderType)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_profile_bad_type_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"fused-window\",\n"
               "  \"neural_network_type\": \"transformer-snn\"\n"
               "}\n";
    }

    Config config{};
    std::string error;

    const bool ok = experiment03::load_profile_to_config(profile_path.string(), config, error);

    ASSERT_FALSE(ok);
    EXPECT_NE(error.find("unsupported neural_network_type"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03CliTest, LoadsOptimizerFromSelectedProfile)
{
    const auto dataset_root = std::filesystem::temp_directory_path();
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_cli_optimizer_profile_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"training_optimizer_type\": \"sgd\",\n"
               "  \"training_optimizer_momentum\": 0.8\n"
               "}\n";
    }

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
    std::string arg1 = "--profile";
    std::string arg2 = profile_path.string();
    char* argv[] = {arg0.data(), arg1.data(), arg2.data()};

    const Config parsed = parseCliParams(3, argv, defaults);

    EXPECT_EQ(parsed.training_optimizer_type, "sgd");
    EXPECT_FLOAT_EQ(parsed.training_optimizer_momentum, 0.8F);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03CliTest, PreservesProfileSeededValuesWhenNotOverridden)
{
    const auto dataset_root = std::filesystem::temp_directory_path();
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_cli_profile_seeded_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"audio-window\",\n"
               "  \"neural_network_type\": \"fused-window-snn\",\n"
               "  \"dataset_input_mode\": 1,\n"
               "  \"training_optimizer_type\": \"sgd\",\n"
               "  \"training_loss_type\": \"mae\",\n"
               "  \"training_learning_rate\": 0.005,\n"
               "  \"training_optimizer_momentum\": 0.4,\n"
               "  \"training_lr_plateau_enabled\": true,\n"
               "  \"training_lr_plateau_factor\": 0.3,\n"
               "  \"training_lr_plateau_patience\": 6,\n"
               "  \"training_lr_plateau_min_delta\": 0.0001,\n"
               "  \"validation_modality_diagnostics_enabled\": true,\n"
               "  \"program_prefetch_lookahead\": 3,\n"
               "  \"program_prefetch_ram_cap_mb\": 96\n"
               "}\n";
    }

    Config defaults{};
    defaults.profile_name = "";
    defaults.device = "cpu";
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
    defaults.kfold_enabled = false;
    defaults.kfold_n_splits = 5;
    defaults.kfold_shuffle = true;
    defaults.kfold_seed = 42U;
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
    defaults.training_loss_type = "mse";
    defaults.training_learning_rate = 0.001F;
    defaults.training_optimizer_momentum = 0.0F;
    defaults.training_optimizer_adam_beta1 = 0.9F;
    defaults.training_optimizer_adam_beta2 = 0.999F;
    defaults.training_optimizer_adam_epsilon = 1e-8F;
    defaults.training_epochs = 1;
    defaults.training_lr_plateau_enabled = true;
    defaults.training_lr_plateau_factor = 0.5F;
    defaults.training_lr_plateau_patience = 3;
    defaults.training_lr_plateau_min_delta = 1e-6F;
    defaults.validation_modality_diagnostics_enabled = false;
    defaults.window_eeg_config = {.window_size = 256, .overlap = 0.5F, .sample_rate = 1024};
    defaults.window_audio_config = {.window_size = 11025, .overlap = 0.5F, .sample_rate = 44100};
    defaults.prefetch_lookahead = 1;
    defaults.prefetch_ram_cap_mb = 64;

    std::string arg0 = "experiment03";
    std::string arg1 = "--profile";
    std::string arg2 = profile_path.string();
    char* argv[] = {arg0.data(), arg1.data(), arg2.data()};

    const Config parsed = parseCliParams(3, argv, defaults);

    EXPECT_EQ(parsed.dataset_input_mode, Protocol101117InputMode::EegOnly);
    EXPECT_EQ(parsed.dataset_type, Experiment03DatasetType::AudioWindow);
    EXPECT_EQ(parsed.autoencoder_type, Experiment03AutoencoderType::FusedWindowSnn);
    EXPECT_EQ(parsed.training_optimizer_type, "sgd");
    EXPECT_EQ(parsed.training_loss_type, "mae");
    EXPECT_FLOAT_EQ(parsed.training_learning_rate, 0.005F);
    EXPECT_FLOAT_EQ(parsed.training_optimizer_momentum, 0.4F);
    EXPECT_TRUE(parsed.training_lr_plateau_enabled);
    EXPECT_FLOAT_EQ(parsed.training_lr_plateau_factor, 0.3F);
    EXPECT_EQ(parsed.training_lr_plateau_patience, 6U);
    EXPECT_FLOAT_EQ(parsed.training_lr_plateau_min_delta, 1e-4F);
    EXPECT_TRUE(parsed.validation_modality_diagnostics_enabled);
    EXPECT_EQ(parsed.prefetch_lookahead, 3U);
    EXPECT_EQ(parsed.prefetch_ram_cap_mb, 96U);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03CliTest, RejectsMissingSamplerSeed)
{
    const std::filesystem::path profile_path =
        std::filesystem::temp_directory_path() / "experiment03_cli_missing_seed_profile_test.json";

    {
        std::ofstream ofs(profile_path);
        ASSERT_TRUE(ofs.good());
        ofs << "{\n"
               "  \"dataset_type\": \"fused-window\",\n"
               "  \"neural_network_type\": \"fused-window-ann\",\n"
               "  \"sampler_shuffle_samples\": true\n"
               "}\n";
    }

    Config defaults{};
    defaults.profile_name = "";
    defaults.device = "cpu";
    defaults.dataset_subject_filter_regex = "^S(\\d+)$";
    defaults.dataset_root_path = std::filesystem::temp_directory_path().string();
    defaults.training_batch_size = 10;
    defaults.training_max_batches_per_epoch = 0;
    defaults.sampler_shuffle_samples = true;
    defaults.sampler_shuffle_seed = std::nullopt;
    defaults.sampler_default_type = "";
    defaults.sampler_weights = {};
    defaults.sampler_weighted_num_samples = std::nullopt;
    defaults.sampler_distributed_num_replicas = 1;
    defaults.sampler_distributed_rank = 0;
    defaults.sampler_distributed_shuffle = true;
    defaults.sampler_distributed_drop_last = false;
    defaults.kfold_enabled = false;
    defaults.kfold_n_splits = 5;
    defaults.kfold_shuffle = true;
    defaults.kfold_seed = 42U;
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
    defaults.training_loss_type = "mse";
    defaults.training_learning_rate = 0.001F;
    defaults.training_optimizer_momentum = 0.0F;
    defaults.training_optimizer_adam_beta1 = 0.9F;
    defaults.training_optimizer_adam_beta2 = 0.999F;
    defaults.training_optimizer_adam_epsilon = 1e-8F;
    defaults.training_epochs = 1;
    defaults.training_lr_plateau_enabled = true;
    defaults.training_lr_plateau_factor = 0.5F;
    defaults.training_lr_plateau_patience = 3;
    defaults.training_lr_plateau_min_delta = 1e-6F;
    defaults.validation_modality_diagnostics_enabled = false;
    defaults.window_eeg_config = {.window_size = 256, .overlap = 0.5F, .sample_rate = 1024};
    defaults.window_audio_config = {.window_size = 11025, .overlap = 0.5F, .sample_rate = 44100};
    defaults.prefetch_lookahead = 1;
    defaults.prefetch_ram_cap_mb = 64;

    std::string arg0 = "experiment03";
    std::string arg1 = "--profile";
    std::string arg2 = profile_path.string();
    char* argv[] = {arg0.data(), arg1.data(), arg2.data()};

    EXPECT_THROW((void) parseCliParams(3, argv, defaults), std::runtime_error);

    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
}

TEST(Experiment03ResultsWriterTest, WritesSummaryJson)
{
    Summary summary{};
    summary.profile_name = "test-profile";
    summary.dataset_type = "fused-window";
    summary.autoencoder_type = "fused-window-ann";
    summary.optimizer_type = "adam";
    summary.loss_type = "mae";
    summary.optimizer_learning_rate = 1e-4F;
    summary.optimizer_momentum = 0.0F;
    summary.optimizer_adam_beta1 = 0.9F;
    summary.optimizer_adam_beta2 = 0.999F;
    summary.optimizer_adam_epsilon = 1e-8F;
    summary.optimizer_final_learning_rate = 5e-5F;
    summary.exit_code = 0;
    summary.total_samples = 100;
    summary.processed_samples = 20;
    summary.seen_batches = 5;
    summary.epoch_mean_losses = {1.0F, 0.9F};
    summary.fold_epoch_val_losses = {{0.8F, 0.7F}};
    summary.fold_epoch_val_eeg_losses = {{0.3F, 0.2F}};
    summary.fold_epoch_val_audio_losses = {{1.3F, 1.2F}};

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
    EXPECT_NE(body.find("\"loss_type\": \"mae\""), std::string::npos);
    EXPECT_NE(body.find("\"optimizer\":"), std::string::npos);
    EXPECT_NE(body.find("\"type\": \"adam\""), std::string::npos);
    EXPECT_NE(body.find("\"final_learning_rate\": 5e-05"), std::string::npos);
    EXPECT_NE(body.find("\"epoch_mean_losses\": [1, 0.9]"), std::string::npos);
    EXPECT_NE(body.find("\"fold_epoch_val_eeg_losses\": [[0.3, 0.2]]"), std::string::npos);
    EXPECT_NE(body.find("\"fold_epoch_val_audio_losses\": [[1.3, 1.2]]"), std::string::npos);
}
