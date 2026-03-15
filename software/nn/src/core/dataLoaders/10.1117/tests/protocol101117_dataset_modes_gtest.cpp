#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>
#include <unistd.h>

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

    static void writeAlignedSubjectMats(const std::filesystem::path& subject_dir,
                                        std::size_t trials)
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
    EXPECT_TRUE(dataset.concatenate_modalities());

    dataset.set_concatenate_modalities(false);
    EXPECT_FALSE(dataset.concatenate_modalities());

    dataset.set_concatenate_modalities(true);
    EXPECT_TRUE(dataset.concatenate_modalities());
}

TEST_F(Protocol101117DatasetModesTest, GetSampleReturnsSeparatedTensorsWhenConfigured)
{
    auto dataset = Protocol101117Dataset(discoveredSubjects(), false);

    const auto sample = dataset.get_sample(0);

    EXPECT_FALSE(sample.concatenated);
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
    auto dataset = Protocol101117Dataset(discoveredSubjects(), false);

    const auto sample = dataset.get_sample(0, true);

    EXPECT_TRUE(sample.concatenated);
    EXPECT_EQ(sample.inputs.cols(),
              static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns() +
                               nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
}

TEST_F(Protocol101117DatasetModesTest, GetItemFollowsModeConfiguration)
{
    auto dataset = Protocol101117Dataset(discoveredSubjects(), false);

    const Batch item = dataset.get_item(0);
    EXPECT_EQ(item.inputs.cols(),
              static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));

    dataset.set_concatenate_modalities(true);
    const Batch concat_batch = dataset.get_item(0);
    EXPECT_EQ(concat_batch.inputs.cols(),
              static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns() +
                               nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
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

    Protocol101117Dataset dataset({only_subject}, false);

    const Batch eeg_only_batch = dataset.collate({0U, 1U, 2U});
    EXPECT_EQ(eeg_only_batch.inputs.rows(), 3);
    EXPECT_EQ(eeg_only_batch.inputs.cols(),
              static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns()));
    EXPECT_EQ(eeg_only_batch.targets.rows(), 3);
    EXPECT_EQ(eeg_only_batch.targets.cols(), 5);

    dataset.set_concatenate_modalities(true);
    const Batch concat_batch = dataset.collate({0U, 1U, 2U});
    EXPECT_EQ(concat_batch.inputs.rows(), 3);
    EXPECT_EQ(concat_batch.inputs.cols(),
              static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns() +
                               nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples()));
    EXPECT_EQ(concat_batch.targets.rows(), 3);
    EXPECT_EQ(concat_batch.targets.cols(), 5);
}

TEST(DemoProbeModelModesTest, AcceptsConcatenatedAndEegOnlyInputs)
{
    constexpr std::size_t eeg_features =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSignalColumns();
    constexpr std::size_t audio_features =
        nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();

    DemoProbeModel model;

    nn::Tensor concatenated(1, eeg_features + audio_features);
    for (std::size_t i = 0; i < eeg_features + audio_features; ++i)
    {
        concatenated.at(0, i) = 1.0f;
    }
    nn::Tensor out_concat = model.forward(concatenated);
    EXPECT_EQ(out_concat.rows(), 1);
    EXPECT_EQ(out_concat.cols(), 2);
    EXPECT_GT(out_concat.at(0, 1), 0.0f);

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
