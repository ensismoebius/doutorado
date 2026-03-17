#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "experiments/03/lib/include/DemoProbeModel.hpp"
#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/10.1117/SubjectDiscovery.hpp"
#include "utils/MockImaginedSpeechDatasetGenerator.hpp"

namespace
{
constexpr std::size_t kStackedConcatRows =
    nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels + 1U;
constexpr std::size_t kStackedConcatCols =
    nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();
constexpr std::size_t kStackedConcatFeatures = kStackedConcatRows * kStackedConcatCols;

class Protocol101117DatasetModesTest : public ::testing::Test
{
   protected:
    std::filesystem::path tmp_root_;

    void SetUp() override
    {
        tmp_root_ = std::filesystem::temp_directory_path() /
                    ("protocol101117_modes_" + std::to_string(getpid()));
        std::filesystem::remove_all(tmp_root_);

        const auto subject_dir = tmp_root_ / "S01";
        std::filesystem::create_directories(subject_dir);

        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateEEGMatFile(
            subject_dir / "S01_EEG.mat", 3U);
        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateAudioMatFile(
            subject_dir / "S01_Audio.mat", 3U);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(tmp_root_);
    }

    [[nodiscard]] auto discoveredSubjects() const -> std::vector<SubjectFiles>
    {
        return discoverSubjects(tmp_root_.string(), "^S(\\d+)$");
    }

    static void writeAlignedSubjectMats(
        const std::filesystem::path& subject_dir, std::size_t trials)
    {
        const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
        const std::size_t eeg_rows = trials;
        const std::size_t eeg_cols = schema.eegTotalColumns();
        const std::size_t audio_rows = trials;
        const std::size_t audio_cols = schema.audioTotalColumns();

        std::vector<double> eeg_data(eeg_rows * eeg_cols, 0.0);
        std::vector<double> audio_data(audio_rows * audio_cols, 0.0);

        for (std::size_t r = 0; r < trials; ++r)
        {
            for (std::size_t c = 0; c < schema.eegSignalColumns(); ++c)
            {
                eeg_data[(c * eeg_rows) + r] = static_cast<double>((r + 1U) * 1000U + c);
            }

            const double stimulus = static_cast<double>((r % 5U) + 1U);
            eeg_data[(schema.eegModeColumn() * eeg_rows) + r] = 1.0;
            eeg_data[(schema.eegStimulusColumn() * eeg_rows) + r] = stimulus;
            eeg_data[(schema.eegBlinkColumn() * eeg_rows) + r] = 1.0;

            for (std::size_t c = 0; c < schema.audioSamples(); ++c)
            {
                audio_data[(c * audio_rows) + r] = static_cast<double>((r + 1U) * 10U + c);
            }

            audio_data[(schema.audioStimulusColumn() * audio_rows) + r] = stimulus;
            // 1-based mapping to avoid any 0-based/1-based ambiguity in tests.
            audio_data[(schema.audioEEGIndexColumn() * audio_rows) + r] =
                static_cast<double>(r + 1U);
        }

        matioCpp::File eeg_file = matioCpp::File::Create((subject_dir / "S99_EEG.mat").string());
        matioCpp::MultiDimensionalArray<double> eeg_matrix(
            nn::dataLoaders::EEG_MAT_VARIABLE_NAME, {eeg_rows, eeg_cols}, eeg_data.data());
        eeg_file.write(eeg_matrix);
        eeg_file.close();

        matioCpp::File audio_file =
            matioCpp::File::Create((subject_dir / "S99_Audio.mat").string());
        matioCpp::MultiDimensionalArray<double> audio_matrix(
            nn::dataLoaders::AUDIO_MAT_VARIABLE_NAME, {audio_rows, audio_cols}, audio_data.data());
        audio_file.write(audio_matrix);
        audio_file.close();
    }
};
} // namespace

TEST_F(Protocol101117DatasetModesTest, ConstructorAndSetterControlConcatenationMode)
{
    auto dataset = Protocol101117Dataset(discoveredSubjects());
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::Concatenated);

    dataset.set_input_mode(Protocol101117InputMode::EegOnly);
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::EegOnly);

    dataset.set_input_mode(Protocol101117InputMode::Concatenated);
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::Concatenated);

    dataset.set_input_mode(Protocol101117InputMode::AudioOnly);
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::AudioOnly);
}

TEST_F(Protocol101117DatasetModesTest, GetSampleReturnsSeparatedTensorsWhenConfigured)
{
    auto dataset = Protocol101117Dataset(discoveredSubjects(), Protocol101117InputMode::EegOnly);

    const auto sample = dataset.get_sample(0);

    EXPECT_EQ(sample.input_mode, Protocol101117InputMode::EegOnly);
    EXPECT_EQ(sample.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));
    EXPECT_EQ(sample.eeg.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));
    EXPECT_EQ(sample.audio.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(sample.targets.cols(), 5);
}

TEST_F(Protocol101117DatasetModesTest, GetSampleOverrideCanReturnConcatenated)
{
    auto dataset = Protocol101117Dataset(discoveredSubjects(), Protocol101117InputMode::EegOnly);

    const auto sample = dataset.get_sample(0, Protocol101117InputMode::Concatenated);

    EXPECT_EQ(sample.input_mode, Protocol101117InputMode::Concatenated);
    EXPECT_EQ(sample.inputs.rows(), static_cast<int>(kStackedConcatRows));
    EXPECT_EQ(sample.inputs.cols(), static_cast<int>(kStackedConcatCols));
}

TEST_F(Protocol101117DatasetModesTest, GetSampleReturnsAudioOnlyWhenConfigured)
{
    auto dataset = Protocol101117Dataset(discoveredSubjects(), Protocol101117InputMode::AudioOnly);

    const auto sample = dataset.get_sample(0);

    EXPECT_EQ(sample.input_mode, Protocol101117InputMode::AudioOnly);
    EXPECT_EQ(sample.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(sample.audio.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(sample.eeg.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));
}

TEST_F(Protocol101117DatasetModesTest, GetItemFollowsModeConfiguration)
{
    auto dataset = Protocol101117Dataset(discoveredSubjects(), Protocol101117InputMode::EegOnly);

    const Batch item = dataset.get_item(0);
    EXPECT_EQ(item.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));

    dataset.set_input_mode(Protocol101117InputMode::Concatenated);
    const Batch concat_batch = dataset.get_item(0);
    EXPECT_EQ(concat_batch.inputs.rows(), static_cast<int>(kStackedConcatRows));
    EXPECT_EQ(concat_batch.inputs.cols(), static_cast<int>(kStackedConcatCols));

    dataset.set_input_mode(Protocol101117InputMode::AudioOnly);
    const Batch audio_only_batch = dataset.get_item(0);
    EXPECT_EQ(audio_only_batch.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
}

TEST_F(Protocol101117DatasetModesTest, CollateFollowsModeConfigurationWithAlignedMapping)
{
    const auto subject_dir = tmp_root_ / "S99";
    std::filesystem::create_directories(subject_dir);
    writeAlignedSubjectMats(subject_dir, 3U);

    SubjectFiles only_subject{};
    only_subject.subject_id = 99;
    only_subject.subject_name = "S99";
    only_subject.eeg_mat_path = (subject_dir / "S99_EEG.mat").string();
    only_subject.audio_mat_path = (subject_dir / "S99_Audio.mat").string();
    only_subject.eeg_rows = 3U;
    only_subject.audio_rows = 3U;

    Protocol101117Dataset dataset({only_subject}, Protocol101117InputMode::EegOnly);

    const Batch eeg_only_batch = dataset.collate({0U, 1U, 2U});
    EXPECT_EQ(eeg_only_batch.inputs.rows(), 3);
    EXPECT_EQ(eeg_only_batch.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));
    EXPECT_EQ(eeg_only_batch.targets.rows(), 3);
    EXPECT_EQ(eeg_only_batch.targets.cols(), 5);

    dataset.set_input_mode(Protocol101117InputMode::Concatenated);
    const Batch concat_batch = dataset.collate({0U, 1U, 2U});
    EXPECT_EQ(concat_batch.inputs.rows(), 3);
    EXPECT_EQ(concat_batch.inputs.cols(), static_cast<int>(kStackedConcatFeatures));
    EXPECT_EQ(concat_batch.targets.rows(), 3);
    EXPECT_EQ(concat_batch.targets.cols(), 5);

    dataset.set_input_mode(Protocol101117InputMode::AudioOnly);
    const Batch audio_only_batch = dataset.collate({0U, 1U, 2U});
    EXPECT_EQ(audio_only_batch.inputs.rows(), 3);
    EXPECT_EQ(audio_only_batch.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(audio_only_batch.targets.rows(), 3);
    EXPECT_EQ(audio_only_batch.targets.cols(), 5);
}

TEST_F(Protocol101117DatasetModesTest,
    ConcatenatedModeStacksAudioThenEegWithLinearResamplingAndPreservesTargets)
{
    const auto subject_dir = tmp_root_ / "S99";
    std::filesystem::create_directories(subject_dir);
    writeAlignedSubjectMats(subject_dir, 3U);

    SubjectFiles only_subject{};
    only_subject.subject_id = 99;
    only_subject.subject_name = "S99";
    only_subject.eeg_mat_path = (subject_dir / "S99_EEG.mat").string();
    only_subject.audio_mat_path = (subject_dir / "S99_Audio.mat").string();
    only_subject.eeg_rows = 3U;
    only_subject.audio_rows = 3U;

    Protocol101117Dataset dataset({only_subject}, Protocol101117InputMode::Concatenated);
    const auto sample = dataset.get_sample(1U);

    const std::size_t eeg_channels = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels;
    const std::size_t eeg_source_width =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    const std::size_t audio_source_width =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();

    ASSERT_EQ(sample.input_mode, Protocol101117InputMode::Concatenated);
    ASSERT_EQ(sample.inputs.rows(), static_cast<int>(kStackedConcatRows));
    ASSERT_EQ(sample.inputs.cols(), static_cast<int>(audio_source_width));
    ASSERT_EQ(sample.audio.cols(), static_cast<int>(audio_source_width));
    ASSERT_EQ(sample.eeg.cols(), static_cast<int>(eeg_channels * eeg_source_width));

    // Targets must remain untouched by the concatenation/stacking transform.
    EXPECT_FLOAT_EQ(sample.targets.at(0, 0), 99.0f); // subject id
    EXPECT_FLOAT_EQ(sample.targets.at(0, 1), 1.0f);  // modality
    EXPECT_FLOAT_EQ(sample.targets.at(0, 2), 2.0f);  // stimulus for row index 1
    EXPECT_FLOAT_EQ(sample.targets.at(0, 3), 1.0f);  // artifact
    EXPECT_FLOAT_EQ(sample.targets.at(0, 4), 2.0f);  // eeg index label (1-based)

    const auto expected_audio_resampled = [&](std::size_t target_idx)
    { return sample.audio.at(0, target_idx); };

    const auto expected_eeg_resampled = [&](std::size_t channel, std::size_t target_idx)
    {
        const std::size_t channel_offset = channel * eeg_source_width;
        const double scale = static_cast<double>(eeg_source_width - 1U) /
                             static_cast<double>(audio_source_width - 1U);
        const double source_pos = static_cast<double>(target_idx) * scale;
        const std::size_t left = static_cast<std::size_t>(std::floor(source_pos));
        const std::size_t right = std::min(left + 1U, eeg_source_width - 1U);
        const double alpha = source_pos - static_cast<double>(left);

        const double left_value = static_cast<double>(sample.eeg.at(0, channel_offset + left));
        const double right_value = static_cast<double>(sample.eeg.at(0, channel_offset + right));
        return static_cast<float>((1.0 - alpha) * left_value + alpha * right_value);
    };

    // Audio must occupy row 0.
    const std::array<std::size_t, 3> probe_indices = {
        0UL,
        audio_source_width / 2UL,
        audio_source_width - 1UL,
    };
    for (const std::size_t idx : probe_indices)
    {
        EXPECT_NEAR(sample.inputs.at(0, idx), expected_audio_resampled(idx), 1e-6f);
    }

    // EEG channels must occupy rows 1..6 in order, each resampled to audio width.
    for (std::size_t ch = 0; ch < eeg_channels; ++ch)
    {
        for (const std::size_t idx : probe_indices)
        {
            EXPECT_NEAR(sample.inputs.at(ch + 1U, idx), expected_eeg_resampled(ch, idx), 1e-3f);
        }
    }
}

TEST_F(
    Protocol101117DatasetModesTest, CollateConcatenatedModePreservesStackOrderAndLinearResampling)
{
    const auto subject_dir = tmp_root_ / "S99";
    std::filesystem::create_directories(subject_dir);
    writeAlignedSubjectMats(subject_dir, 3U);

    SubjectFiles only_subject{};
    only_subject.subject_id = 99;
    only_subject.subject_name = "S99";
    only_subject.eeg_mat_path = (subject_dir / "S99_EEG.mat").string();
    only_subject.audio_mat_path = (subject_dir / "S99_Audio.mat").string();
    only_subject.eeg_rows = 3U;
    only_subject.audio_rows = 3U;

    Protocol101117Dataset dataset({only_subject}, Protocol101117InputMode::Concatenated);
    const Batch batch = dataset.collate({0U, 1U, 2U});

    const std::size_t eeg_channels = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels;
    const std::size_t eeg_source_width =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    const std::size_t audio_source_width =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();

    ASSERT_EQ(batch.inputs.rows(), 3);
    ASSERT_EQ(batch.inputs.cols(), static_cast<int>(kStackedConcatFeatures));
    ASSERT_EQ(batch.targets.rows(), 3);
    ASSERT_EQ(batch.targets.cols(), 5);

    // Validate labels for row 1 are preserved after collate transformation.
    EXPECT_FLOAT_EQ(batch.targets.at(1, 0), 99.0f);
    EXPECT_FLOAT_EQ(batch.targets.at(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(batch.targets.at(1, 2), 2.0f);
    EXPECT_FLOAT_EQ(batch.targets.at(1, 3), 1.0f);
    EXPECT_FLOAT_EQ(batch.targets.at(1, 4), 2.0f);

    const std::array<std::size_t, 3> probe_indices = {
        0UL,
        audio_source_width / 2UL,
        audio_source_width - 1UL,
    };

    // For row index 1 in writeAlignedSubjectMats: audio source is 20 + c.
    for (const std::size_t idx : probe_indices)
    {
        const float expected_audio = static_cast<float>(20.0 + static_cast<double>(idx));
        EXPECT_NEAR(batch.inputs.at(1, idx), expected_audio, 1e-6f);
    }

    // EEG channel blocks are flattened after the audio row: [audio][eeg1]...[eeg6].
    const auto expected_eeg_resampled = [&](std::size_t channel, std::size_t target_idx)
    {
        const std::size_t channel_offset = channel * eeg_source_width;
        const double scale = static_cast<double>(eeg_source_width - 1U) /
                             static_cast<double>(audio_source_width - 1U);
        const double source_pos = static_cast<double>(target_idx) * scale;
        const double expected_source_value =
            2000.0 + static_cast<double>(channel_offset) + source_pos;
        return static_cast<float>(expected_source_value);
    };

    for (std::size_t ch = 0; ch < eeg_channels; ++ch)
    {
        const std::size_t block_offset = (ch + 1U) * audio_source_width;
        for (const std::size_t idx : probe_indices)
        {
            EXPECT_NEAR(
                batch.inputs.at(1, block_offset + idx), expected_eeg_resampled(ch, idx), 1e-3f);
        }
    }
}

TEST(DemoProbeModelModesTest, AcceptsConcatenatedAndEegOnlyInputs)
{
    constexpr std::size_t eeg_features =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns();
    constexpr std::size_t audio_features =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();
    constexpr std::size_t stacked_rows =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels + 1U;
    constexpr std::size_t stacked_concat_features = stacked_rows * audio_features;

    DemoProbeModel model;

    nn::Tensor concatenated(1, stacked_concat_features);
    for (std::size_t i = 0; i < stacked_concat_features; ++i)
    {
        concatenated.at(0, i) = 1.0f;
    }
    nn::Tensor out_concat = model.forward(concatenated);
    EXPECT_EQ(out_concat.rows(), 1);
    EXPECT_EQ(out_concat.cols(), 2);
    EXPECT_GT(out_concat.at(0, 1), 0.0f);

    nn::Tensor stacked(stacked_rows, audio_features);
    for (std::size_t r = 0; r < stacked_rows; ++r)
    {
        for (std::size_t c = 0; c < audio_features; ++c)
        {
            stacked.at(r, c) = 1.0f;
        }
    }
    nn::Tensor out_stacked = model.forward(stacked);
    EXPECT_EQ(out_stacked.rows(), 1);
    EXPECT_EQ(out_stacked.cols(), 2);
    EXPECT_GT(out_stacked.at(0, 1), 0.0f);

    nn::Tensor eeg_only(1, eeg_features);
    for (std::size_t i = 0; i < eeg_features; ++i)
    {
        eeg_only.at(0, i) = 1.0f;
    }
    nn::Tensor out_eeg = model.forward(eeg_only);
    EXPECT_EQ(out_eeg.rows(), 1);
    EXPECT_EQ(out_eeg.cols(), 2);
    EXPECT_EQ(out_eeg.at(0, 1), 0.0f);

    nn::Tensor invalid(1, 7);
    EXPECT_THROW((void) model.forward(invalid), std::runtime_error);
}
