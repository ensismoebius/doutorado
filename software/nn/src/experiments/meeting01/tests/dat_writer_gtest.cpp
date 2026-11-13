#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "../lib/include/Meeting01DatasetSplit.hpp"
#include "../lib/include/Meeting01Output.hpp"
#include "../lib/include/Meeting01PerWindow.hpp"
#include "../lib/include/Meeting01ResultRow.hpp"
#include "../lib/include/Meeting01RunMetrics.hpp"
#include "cnpy.h"

namespace fs = std::filesystem;
using namespace meeting01;

class DATWriterTest : public ::testing::Test
{
   protected:
    fs::path test_dir;

    void SetUp() override
    {
        test_dir = fs::temp_directory_path() / "dat_writer_tests" /
                   std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        fs::create_directories(test_dir);
    }

    void TearDown() override
    {
        if (fs::exists(test_dir))
        {
            fs::remove_all(test_dir);
        }
    }

    std::vector<std::string> read_file_lines(const fs::path& path)
    {
        std::vector<std::string> lines;
        std::ifstream file(path);
        EXPECT_TRUE(file.is_open()) << "Failed to open: " << path;
        std::string line;
        while (std::getline(file, line))
        {
            lines.push_back(line);
        }
        return lines;
    }
};

// ---- write_reference_inputs (inputs of the PCA / mean-frame reference baselines) ----

namespace
{

constexpr nn::Index kRefWidth = 8;

// Sample s of the window holds base + s, so a read-back value names both the window it
// came from (base) and its position in it (s): a transposed or reordered dump cannot pass.
auto ramp_window(float base, nn::Index width = kRefWidth) -> nn::Tensor
{
    nn::Tensor w(width, 1);
    for (nn::Index s = 0; s < width; ++s) w.at(s, 0) = base + static_cast<float>(s);
    return w;
}

auto ref_meta(int speaker_id, int recording_id, int window_id, int valid_length) -> WindowMetadata
{
    return WindowMetadata{.speaker = "s" + std::to_string(speaker_id),
        .speaker_id = speaker_id,
        .recording_id = recording_id,
        .window_id = window_id,
        .source_window_index = window_id % 2,
        .digit = 0,
        .valid_length = valid_length};
}

// train: 2 windows (one zero-padded after sample 5), val: 1, test: 3 (one padded after 3).
auto three_part_split() -> DatasetSplit
{
    DatasetSplit split;
    split.train_samples = {ramp_window(100.0f), ramp_window(200.0f)};
    split.train_meta = {ref_meta(1, 10, 0, 8), ref_meta(1, 11, 1, 5)};
    split.val_samples = {ramp_window(300.0f)};
    split.val_meta = {ref_meta(2, 20, 2, 8)};
    split.test_samples = {ramp_window(400.0f), ramp_window(500.0f), ramp_window(600.0f)};
    split.test_meta = {ref_meta(3, 30, 3, 8), ref_meta(3, 30, 4, 8), ref_meta(3, 31, 5, 3)};
    return split;
}

} // namespace

TEST_F(DATWriterTest, ReferenceInputsAreTheWindowsThemselvesInSampleOrder)
{
    write_reference_inputs(test_dir, "tag_fsdd_fold0", three_part_split());

    const std::vector<std::pair<std::string, std::vector<float>>> parts{
        {"train", {100.0f, 200.0f}}, {"val", {300.0f}}, {"test", {400.0f, 500.0f, 600.0f}}};
    for (const auto& [part, bases] : parts)
    {
        const cnpy::NpyArray arr = cnpy::npy_load(
            (test_dir / ("tag_fsdd_fold0_target_" + part + "_windows.npy")).string());
        ASSERT_EQ(arr.shape, (std::vector<size_t>{bases.size(), static_cast<size_t>(kRefWidth)}))
            << part;
        EXPECT_FALSE(arr.fortran_order) << part;
        ASSERT_EQ(arr.word_size, sizeof(float)) << part;
        const float* v = arr.data<float>();
        for (size_t r = 0; r < bases.size(); ++r)
            for (size_t s = 0; s < static_cast<size_t>(kRefWidth); ++s)
                EXPECT_FLOAT_EQ(v[r * kRefWidth + s], bases[r] + static_cast<float>(s))
                    << part << " row " << r << " sample " << s;
    }
}

TEST_F(DATWriterTest, ReferenceInputsMetaCarriesValidLengthPerWindow)
{
    write_reference_inputs(test_dir, "tag_fsdd_fold0", three_part_split());

    const auto train = read_file_lines(test_dir / "tag_fsdd_fold0_target_train_windows_meta.csv");
    ASSERT_EQ(train.size(), 3u);
    EXPECT_EQ(train[0], "speaker_id,recording_id,window_id,source_window_index,valid_length");
    EXPECT_EQ(train[1], "1,10,0,0,8");
    EXPECT_EQ(train[2], "1,11,1,1,5");

    const auto test = read_file_lines(test_dir / "tag_fsdd_fold0_target_test_windows_meta.csv");
    ASSERT_EQ(test.size(), 4u);
    EXPECT_EQ(test[3], "3,31,5,1,3");

    const auto val = read_file_lines(test_dir / "tag_fsdd_fold0_target_val_windows_meta.csv");
    ASSERT_EQ(val.size(), 2u);
    EXPECT_EQ(val[1], "2,20,2,0,8");
}

TEST_F(DATWriterTest, ReferenceInputsRefuseAnEmptyPart)
{
    DatasetSplit split = three_part_split();
    split.val_samples.clear();
    split.val_meta.clear();
    EXPECT_THROW(write_reference_inputs(test_dir, "t", split), std::runtime_error);
}

TEST_F(DATWriterTest, ReferenceInputsRefuseWindowsOfUnequalWidth)
{
    DatasetSplit split = three_part_split();
    split.test_samples[1] = ramp_window(500.0f, kRefWidth - 1);
    EXPECT_THROW(write_reference_inputs(test_dir, "t", split), std::runtime_error);
}

TEST_F(DATWriterTest, ReferenceInputsRefuseAValidLengthLongerThanTheWindow)
{
    DatasetSplit split = three_part_split();
    split.train_meta[0].valid_length = static_cast<int>(kRefWidth) + 1;
    EXPECT_THROW(write_reference_inputs(test_dir, "t", split), std::runtime_error);
}

TEST_F(DATWriterTest, ReferenceInputsRefuseMetadataNotParallelToWindows)
{
    DatasetSplit split = three_part_split();
    split.test_meta.pop_back();
    EXPECT_THROW(write_reference_inputs(test_dir, "t", split), std::runtime_error);
}

TEST_F(DATWriterTest, PerWindowErrorsOfARerunReplaceTheEarlierRunsRows)
{
    // The same fold run twice into one results directory: the file must hold the second
    // run's rows only. Appending (the old behaviour) left both, and the readers averaged an
    // older binary's errors with the new ones as if they were one run.
    const fs::path path = test_dir / "t_fsdd_fold0_per_window_errors.csv";
    PerWindowError earlier;
    earlier.model = "lstm-ae";
    earlier.seed = 42u;
    earlier.cv_fold = 0;
    earlier.split = "test";
    earlier.window_id = 7;
    earlier.mse = 9.0f;
    write_per_window_errors_csv(path, {earlier});

    PerWindowError rerun = earlier;
    rerun.encoding = "poisson";
    rerun.mse = 0.5f;
    write_per_window_errors_csv(path, {rerun});

    const auto lines = read_file_lines(path);
    ASSERT_EQ(lines.size(), 2u) << "header + the rerun's one row";
    EXPECT_EQ(lines[0].rfind("model,encoding,architecture,", 0), 0u);
    EXPECT_EQ(lines[1].rfind("lstm-ae,poisson,", 0), 0u) << lines[1];
    EXPECT_NE(lines[1].find(",0.50000000,"), std::string::npos) << lines[1];
}

TEST_F(DATWriterTest, ComparativeCsvReplacesTheOldFileWholeThroughATmpFile)
{
    // This file is 01_meeting01_run_loso.sh's RESUME=1 "fold finished" marker, so it goes
    // through <path>.tmp + rename: a kill mid-write can leave a stray .tmp, never a truncated
    // marker. Here: an older marker is replaced in full and no .tmp is left behind.
    const fs::path path = test_dir / "t_fsdd_fold0_comparative_metrics.csv";
    ResultRow older;
    older.model = "lstm-ae";
    older.split = "test";
    older.cv_fold = 0;
    write_rows_csv(path, {older, older, older});

    ResultRow newer = older;
    newer.model = "gru-ae";
    write_rows_csv(path, {newer});

    EXPECT_FALSE(fs::exists(fs::path(path.string() + ".tmp")));
    const auto lines = read_file_lines(path);
    ASSERT_EQ(lines.size(), 2u) << "header + the newer run's one row";
    EXPECT_EQ(lines[0].rfind("backend,profile,dataset,model,", 0), 0u) << lines[0];
    EXPECT_NE(lines[1].find(",gru-ae,"), std::string::npos) << lines[1];
}

TEST_F(DATWriterTest, ComparativeCsvRefusesADirectoryItCannotWrite)
{
    // It used to open the stream unchecked, so an unwritable path wrote nothing and returned:
    // the fold then looked unfinished to RESUME, with no error anywhere.
    EXPECT_THROW(write_rows_csv(test_dir / "no_such_dir" / "x_comparative_metrics.csv", {}),
        std::runtime_error);
}
