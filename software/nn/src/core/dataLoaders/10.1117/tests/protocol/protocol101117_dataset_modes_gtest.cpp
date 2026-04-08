/**
 * @file src/core/dataLoaders/10.1117/tests/protocol/protocol101117_dataset_modes_gtest.cpp
 * @brief Implementation for Protocol101117 dataset modes gtest.
 *

 */

#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp"
#include "nn/dataLoaders/10.1117/datasets/raw/SamplePacking.hpp"
#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/NAMES.hpp"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/testing/SqliteTestHelpers.hpp"
#include "utils/MockImaginedSpeechDatasetGenerator.hpp"

namespace
{
constexpr std::size_t kStackedConcatRows =
    nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels + 1U;
constexpr std::size_t kStackedConcatCols =
    nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();
constexpr std::size_t kStackedConcatFeatures = kStackedConcatRows * kStackedConcatCols;

class Dataset101117ModesTest : public ::testing::Test
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
            nn::dataLoaders::kEegMatVariableName, {eeg_rows, eeg_cols}, eeg_data.data());
        eeg_file.write(eeg_matrix);
        eeg_file.close();

        matioCpp::File audio_file =
            matioCpp::File::Create((subject_dir / "S99_Audio.mat").string());
        matioCpp::MultiDimensionalArray<double> audio_matrix(
            nn::dataLoaders::kAudioMatVariableName, {audio_rows, audio_cols}, audio_data.data());
        audio_file.write(audio_matrix);
        audio_file.close();
    }

    static void writeMismatchedStimulusSubjectMats(const std::filesystem::path& subject_dir)
    {
        const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
        const std::size_t eeg_rows = 1;
        const std::size_t eeg_cols = schema.eegTotalColumns();
        const std::size_t audio_rows = 1;
        const std::size_t audio_cols = schema.audioTotalColumns();

        std::vector<double> eeg_data(eeg_rows * eeg_cols, 0.0);
        std::vector<double> audio_data(audio_rows * audio_cols, 0.0);

        for (std::size_t c = 0; c < schema.eegSignalColumns(); ++c)
        {
            eeg_data[c * eeg_rows] = static_cast<double>(1000U + c);
        }
        for (std::size_t c = 0; c < schema.audioSamples(); ++c)
        {
            audio_data[c * audio_rows] = static_cast<double>(10U + c);
        }

        eeg_data[schema.eegModeColumn() * eeg_rows] = 1.0;
        eeg_data[schema.eegStimulusColumn() * eeg_rows] = 1.0;
        eeg_data[schema.eegBlinkColumn() * eeg_rows] = 0.0;

        // Intentionally mismatch audio stimulus label against EEG stimulus label.
        audio_data[schema.audioStimulusColumn() * audio_rows] = 2.0;
        audio_data[schema.audioEEGIndexColumn() * audio_rows] = 1.0;

        matioCpp::File eeg_file = matioCpp::File::Create((subject_dir / "S77_EEG.mat").string());
        matioCpp::MultiDimensionalArray<double> eeg_matrix(
            nn::dataLoaders::kEegMatVariableName, {eeg_rows, eeg_cols}, eeg_data.data());
        eeg_file.write(eeg_matrix);
        eeg_file.close();

        matioCpp::File audio_file =
            matioCpp::File::Create((subject_dir / "S77_Audio.mat").string());
        matioCpp::MultiDimensionalArray<double> audio_matrix(
            nn::dataLoaders::kAudioMatVariableName, {audio_rows, audio_cols}, audio_data.data());
        audio_file.write(audio_matrix);
        audio_file.close();
    }
};
} // namespace

TEST_F(Dataset101117ModesTest, ConstructorAndSetterControlConcatenationMode)
{
    auto dataset = Dataset101117(discoveredSubjects());
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::Concatenated);

    dataset.set_input_mode(Protocol101117InputMode::EegOnly);
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::EegOnly);

    dataset.set_input_mode(Protocol101117InputMode::Concatenated);
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::Concatenated);

    dataset.set_input_mode(Protocol101117InputMode::AudioOnly);
    EXPECT_EQ(dataset.input_mode(), Protocol101117InputMode::AudioOnly);
}

TEST_F(Dataset101117ModesTest, GetSampleReturnsSeparatedTensorsWhenConfigured)
{
    auto dataset = Dataset101117(discoveredSubjects(), Protocol101117InputMode::EegOnly);

    const auto sample = dataset.get_sample(0);

    EXPECT_EQ(sample.input_mode, Protocol101117InputMode::EegOnly);
    EXPECT_EQ(sample.inputs.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels));
    EXPECT_EQ(sample.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel()));
    EXPECT_EQ(sample.eeg.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels));
    EXPECT_EQ(sample.eeg.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel()));
    EXPECT_EQ(sample.audio.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(sample.audio.cols(), 1);
    EXPECT_EQ(sample.targets.cols(), 5);
}

TEST_F(Dataset101117ModesTest, GetSampleOverrideCanReturnConcatenated)
{
    auto dataset = Dataset101117(discoveredSubjects(), Protocol101117InputMode::EegOnly);

    const auto sample = dataset.get_sample(0, Protocol101117InputMode::Concatenated);

    EXPECT_EQ(sample.input_mode, Protocol101117InputMode::Concatenated);
    EXPECT_EQ(sample.inputs.rows(), static_cast<int>(kStackedConcatRows));
    EXPECT_EQ(sample.inputs.cols(), static_cast<int>(kStackedConcatCols));
}

TEST_F(Dataset101117ModesTest, GetSampleReturnsAudioOnlyWhenConfigured)
{
    auto dataset = Dataset101117(discoveredSubjects(), Protocol101117InputMode::AudioOnly);

    const auto sample = dataset.get_sample(0);

    EXPECT_EQ(sample.input_mode, Protocol101117InputMode::AudioOnly);
    EXPECT_EQ(sample.inputs.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(sample.inputs.cols(), 1);
    EXPECT_EQ(sample.audio.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(sample.audio.cols(), 1);
    EXPECT_EQ(sample.eeg.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels));
    EXPECT_EQ(sample.eeg.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel()));
}

TEST_F(Dataset101117ModesTest, GetItemFollowsModeConfiguration)
{
    auto dataset = Dataset101117(discoveredSubjects(), Protocol101117InputMode::EegOnly);

    const Batch item = dataset.get_item(0);
    EXPECT_EQ(item.inputs.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels));
    EXPECT_EQ(item.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel()));

    dataset.set_input_mode(Protocol101117InputMode::Concatenated);
    const Batch concat_batch = dataset.get_item(0);
    EXPECT_EQ(concat_batch.inputs.rows(), static_cast<int>(kStackedConcatRows));
    EXPECT_EQ(concat_batch.inputs.cols(), static_cast<int>(kStackedConcatCols));

    dataset.set_input_mode(Protocol101117InputMode::AudioOnly);
    const Batch audio_only_batch = dataset.get_item(0);
    EXPECT_EQ(audio_only_batch.inputs.rows(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(audio_only_batch.inputs.cols(), 1);
}

TEST_F(Dataset101117ModesTest, CollateFollowsModeConfigurationWithAlignedMapping)
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

    Dataset101117 dataset({only_subject}, Protocol101117InputMode::EegOnly);

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

TEST_F(Dataset101117ModesTest,
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

    Dataset101117 dataset({only_subject}, Protocol101117InputMode::Concatenated);
    const auto sample = dataset.get_sample(1U);

    const std::size_t eeg_channels = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels;
    const std::size_t eeg_source_width =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    const std::size_t audio_source_width =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();

    ASSERT_EQ(sample.input_mode, Protocol101117InputMode::Concatenated);
    ASSERT_EQ(sample.inputs.rows(), static_cast<int>(kStackedConcatRows));
    ASSERT_EQ(sample.inputs.cols(), static_cast<int>(audio_source_width));
    ASSERT_EQ(sample.audio.rows(), static_cast<int>(audio_source_width));
    ASSERT_EQ(sample.audio.cols(), 1);
    ASSERT_EQ(sample.eeg.rows(), static_cast<int>(eeg_channels));
    ASSERT_EQ(sample.eeg.cols(), static_cast<int>(eeg_source_width));

    // Targets must remain untouched by the concatenation/stacking transform.
    EXPECT_FLOAT_EQ(sample.targets.at(0, 0), 99.0f); // subject id
    EXPECT_FLOAT_EQ(sample.targets.at(0, 1), 1.0f);  // modality
    EXPECT_FLOAT_EQ(sample.targets.at(0, 2), 2.0f);  // stimulus for row index 1
    EXPECT_FLOAT_EQ(sample.targets.at(0, 3), 1.0f);  // artifact
    EXPECT_FLOAT_EQ(sample.targets.at(0, 4), 2.0f);  // eeg index label (1-based)

    const auto expected_audio_resampled = [&](std::size_t target_idx)
    { return sample.audio.at(target_idx, 0); };

    const auto expected_eeg_resampled = [&](std::size_t channel, std::size_t target_idx)
    {
        const double scale = static_cast<double>(eeg_source_width - 1U) /
                             static_cast<double>(audio_source_width - 1U);
        const double source_pos = static_cast<double>(target_idx) * scale;
        const std::size_t left = static_cast<std::size_t>(std::floor(source_pos));
        const std::size_t right = std::min(left + 1U, eeg_source_width - 1U);
        const double alpha = source_pos - static_cast<double>(left);

        const double left_value = static_cast<double>(sample.eeg.at(channel, left));
        const double right_value = static_cast<double>(sample.eeg.at(channel, right));
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
    Dataset101117ModesTest, CollateConcatenatedModePreservesStackOrderAndLinearResampling)
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

    Dataset101117 dataset({only_subject}, Protocol101117InputMode::Concatenated);
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

TEST_F(Dataset101117ModesTest, SamplePackingBuildInputAndTargetContracts)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;

    nn::Tensor eeg(schema.eeg_channels, schema.eegSamplesPerChannel());
    nn::Tensor audio(schema.audioSamples(), 1);
    eeg.setZero();
    audio.setZero();

    const auto merged = buildInputTensor(eeg, audio);
    EXPECT_EQ(merged.rows(), static_cast<int>(schema.eeg_channels + 1U));
    EXPECT_EQ(merged.cols(), static_cast<int>(schema.audioSamples()));

    const auto target = buildTargetTensor(7, {1, 2, 3}, 9);
    ASSERT_EQ(target.rows(), 1);
    ASSERT_EQ(target.cols(), 5);
    EXPECT_EQ(target.at(0, 0), 7.0F);
    EXPECT_EQ(target.at(0, 1), 1.0F);
    EXPECT_EQ(target.at(0, 2), 2.0F);
    EXPECT_EQ(target.at(0, 3), 3.0F);
    EXPECT_EQ(target.at(0, 4), 9.0F);
}

TEST_F(Dataset101117ModesTest, SamplePackingBuildInputThrowsOnWrongShapes)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    nn::Tensor good_eeg(schema.eeg_channels, schema.eegSamplesPerChannel());
    nn::Tensor good_audio(schema.audioSamples(), 1);

    nn::Tensor bad_eeg(schema.eeg_channels - 1U, schema.eegSamplesPerChannel());
    EXPECT_THROW((void) buildInputTensor(bad_eeg, good_audio), std::runtime_error);

    nn::Tensor bad_audio(schema.audioSamples(), 2);
    EXPECT_THROW((void) buildInputTensor(good_eeg, bad_audio), std::runtime_error);
}

TEST_F(Dataset101117ModesTest, GetSampleAndCollateRejectOutOfRangeIndexes)
{
    Dataset101117 dataset(discoveredSubjects(), Protocol101117InputMode::Concatenated);

    ASSERT_GT(dataset.size(), 0U);
    EXPECT_THROW((void) dataset.get_sample(dataset.size()), std::out_of_range);
    EXPECT_THROW((void) dataset.collate({dataset.size()}), std::out_of_range);

    Batch batch;
    EXPECT_THROW(dataset.collate_into({dataset.size()}, batch), std::out_of_range);
}

TEST_F(Dataset101117ModesTest, CollateIntoResizesBatchBuffersPerMode)
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

    Dataset101117 dataset({only_subject}, Protocol101117InputMode::Concatenated);
    Batch batch;

    dataset.collate_into({0U, 1U}, batch);
    EXPECT_EQ(batch.inputs.rows(), 2);
    EXPECT_EQ(batch.inputs.cols(), static_cast<int>(kStackedConcatFeatures));
    EXPECT_EQ(batch.targets.rows(), 2);
    EXPECT_EQ(batch.targets.cols(), 5);

    dataset.set_input_mode(Protocol101117InputMode::EegOnly);
    dataset.collate_into({0U, 1U}, batch);
    EXPECT_EQ(batch.inputs.rows(), 2);
    EXPECT_EQ(batch.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));

    dataset.set_input_mode(Protocol101117InputMode::AudioOnly);
    dataset.collate_into({0U, 1U}, batch);
    EXPECT_EQ(batch.inputs.rows(), 2);
    EXPECT_EQ(batch.inputs.cols(),
        static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));

    // Protocol get_item returns non-row-shaped inputs, so empty base collate_into
    // delegates to a reshape path that can throw for pre-sized buffers.
    EXPECT_THROW(dataset.collate_into({}, batch), std::exception);
}

TEST_F(Dataset101117ModesTest, SqliteBackedSubjectsInitializeSessionsAndReadSamples)
{
    int subject_id = -1;
    const std::string db_path = nn::testing::create_mock_imagined_db(subject_id,
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples(),
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());

    SubjectFiles subject{};
    subject.subject_id = subject_id;
    subject.subject_name = "sqlite_subject";
    subject.audio_mat_path = db_path;
    subject.eeg_mat_path = db_path;
    // Mock DB stores rows at audio_row={10,11}; use a range that includes them.
    subject.audio_rows = 12U;
    subject.eeg_rows = 2U;

    Dataset101117 dataset({subject}, Protocol101117InputMode::Concatenated);
    const auto sample = dataset.get_sample(11U);

    EXPECT_EQ(sample.input_mode, Protocol101117InputMode::Concatenated);
    EXPECT_EQ(sample.inputs.rows(), static_cast<int>(kStackedConcatRows));
    EXPECT_EQ(sample.inputs.cols(), static_cast<int>(kStackedConcatCols));
    EXPECT_EQ(sample.targets.rows(), 1);
    EXPECT_EQ(sample.targets.cols(), 5);

    std::filesystem::remove(db_path);
}

TEST_F(Dataset101117ModesTest, GetSampleThrowsWhenStimulusLabelsAreMismatched)
{
    const auto subject_dir = tmp_root_ / "S77";
    std::filesystem::create_directories(subject_dir);
    writeMismatchedStimulusSubjectMats(subject_dir);

    SubjectFiles only_subject{};
    only_subject.subject_id = 77;
    only_subject.subject_name = "S77";
    only_subject.eeg_mat_path = (subject_dir / "S77_EEG.mat").string();
    only_subject.audio_mat_path = (subject_dir / "S77_Audio.mat").string();
    only_subject.eeg_rows = 1U;
    only_subject.audio_rows = 1U;

    Dataset101117 dataset({only_subject}, Protocol101117InputMode::Concatenated);

    EXPECT_THROW((void) dataset.get_sample(0U), std::runtime_error);
}
