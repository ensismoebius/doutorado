/**
 * @file eeg_loader_gtest.cpp
 * @brief Basic correctness tests for loading an EEG row into (channels x samples) form.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "../../tests/MatTestUtils/MatTestUtils.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"

using namespace nn::dataLoaders;
using namespace nn::dataLoaders::test;

TEST(EEGLoaderTest, LoadSingleRow)
{
    std::string fname = std::filesystem::temp_directory_path().string() + "/tests_eeg_tmp.mat";
    const size_t rows = 1;
    const size_t cols = 24579; // 24576 samples + 3 labels

    std::vector<double> data(rows * cols, 0.0);
    for (size_t i = 0; i < 24576; ++i)
    {
        data[i * rows] = static_cast<double>(i + 1);
    }
    data[24576 * rows] = 7;        // modality
    data[(24576 + 1) * rows] = 42; // stimulus
    data[(24576 + 2) * rows] = 0;  // artifact

    ASSERT_TRUE(writeMatDouble(fname, "EEG", rows, cols, data.data()));

    auto [matrix, labels] = loadEEGFromMat(fname, 0);
    EXPECT_EQ(matrix.rows(), 6);
    EXPECT_EQ(matrix.cols(), 4096);
    EXPECT_FLOAT_EQ(matrix(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(matrix(0, 1), 2.0F);
    EXPECT_EQ(labels[0], 7);
    EXPECT_EQ(labels[1], 42);
    EXPECT_EQ(labels[2], 0);

    remove(fname.c_str());
}

TEST(EEGLoaderTest, SessionSupportsRowsFlatMetadataAndCache)
{
    std::string fname =
        std::filesystem::temp_directory_path().string() + "/tests_eeg_session_tmp.mat";
    const size_t rows = 3;
    const size_t cols = 24579;

    std::vector<double> data(rows * cols, 0.0);
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t i = 0; i < 24576; ++i)
        {
            data[(i * rows) + r] = static_cast<double>((r * 10000) + i);
        }
        data[(24576 * rows) + r] = static_cast<double>(r + 1);
        data[((24576 + 1) * rows) + r] = static_cast<double>(r + 11);
        data[((24576 + 2) * rows) + r] = static_cast<double>(r % 2);
    }

    ASSERT_TRUE(writeMatDouble(fname, "EEG", rows, cols, data.data()));

    nn::dataLoaders::EEGMatSession session(fname);
    EXPECT_EQ(session.filePath(), fname);
    EXPECT_EQ(session.rowCount(), rows);

    auto rows_data = session.readRows(0, rows);
    ASSERT_EQ(rows_data.size(), rows);

    auto [r1_tensor, r1_labels] = rows_data[1];
    EXPECT_EQ(r1_tensor.rows(), 6);
    EXPECT_EQ(r1_tensor.cols(), 4096);
    EXPECT_EQ(r1_labels[0], 2);
    EXPECT_EQ(r1_labels[1], 12);
    EXPECT_EQ(r1_labels[2], 1);

    auto rows_data_cached = session.readRows(0, rows);
    ASSERT_EQ(rows_data_cached.size(), rows);
    auto [r2_tensor, r2_labels] = rows_data_cached[2];
    EXPECT_EQ(r2_tensor.rows(), 6);
    EXPECT_EQ(r2_labels[0], 3);

    auto flat = session.readRowsFlat(0, rows);
    EXPECT_EQ(flat.signals.size(), rows * 24576U);
    ASSERT_EQ(flat.labels.size(), rows);
    EXPECT_EQ(flat.labels[0][1], 11);

    auto empty_rows = session.readRows(0, 0);
    EXPECT_TRUE(empty_rows.empty());

    auto empty_flat = session.readRowsFlat(0, 0);
    EXPECT_TRUE(empty_flat.signals.empty());
    EXPECT_TRUE(empty_flat.labels.empty());

    EXPECT_THROW((void) session.readRows(2, 2), std::runtime_error);
    EXPECT_THROW((void) session.readRowsFlat(2, 2), std::runtime_error);

    remove(fname.c_str());
}

TEST(EEGLoaderTest, StatefulLoaderOpenReadVariableAndNumericProbe)
{
    std::string fname =
        std::filesystem::temp_directory_path().string() + "/tests_eeg_stateful_tmp.mat";
    const size_t rows = 1;
    const size_t cols = 24579;
    std::vector<double> data(rows * cols, 1.0);
    ASSERT_TRUE(writeMatDouble(fname, "EEG", rows, cols, data.data()));

    nn::dataLoaders::EEGLoader loader;

    EXPECT_FALSE(loader.open("/tmp/nonexistent-eeg-loader-file.mat"));

    auto missing_var = loader.readVariable("EEG");
    EXPECT_EQ(missing_var.get(), nullptr);
    auto missing_numeric = loader.readFirstNumericVariable();
    EXPECT_FALSE(missing_numeric.has_value());

    ASSERT_TRUE(loader.open(fname));
    EXPECT_EQ(loader.filePath(), fname);

    auto eeg_var = loader.readVariable("EEG");
    ASSERT_NE(eeg_var.get(), nullptr);
    EXPECT_EQ(eeg_var->rank, 2);

    auto numeric = loader.readFirstNumericVariable();
    ASSERT_TRUE(numeric.has_value());
    ASSERT_NE(numeric->get(), nullptr);

    loader.close();

    const auto symlink_path = fname + ".symlink";
    std::error_code ec;
    std::filesystem::remove(symlink_path, ec);
    std::filesystem::create_symlink(fname, symlink_path, ec);
    if (!ec)
    {
        EXPECT_FALSE(loader.open(symlink_path));
        std::filesystem::remove(symlink_path, ec);
    }

    remove(fname.c_str());
}

TEST(EEGLoaderTest, SessionConstructorRejectsMissingVariable)
{
    std::string fname =
        std::filesystem::temp_directory_path().string() + "/tests_eeg_missing_var_session_tmp.mat";
    const size_t rows = 1;
    const size_t cols = 4;
    std::vector<double> data(rows * cols, 1.0);
    ASSERT_TRUE(writeMatDouble(fname, "NotEEG", rows, cols, data.data()));

    EXPECT_THROW((void) nn::dataLoaders::EEGMatSession(fname), std::runtime_error);
    remove(fname.c_str());
}

TEST(EEGLoaderTest, SessionConstructorRejectsWrongDimensionsAndType)
{
    {
        std::string bad_dims =
            std::filesystem::temp_directory_path().string() + "/tests_eeg_bad_dims_session_tmp.mat";
        const size_t rows = 1;
        const size_t cols = 24578;
        std::vector<double> data(rows * cols, 1.0);
        ASSERT_TRUE(writeMatDouble(bad_dims, "EEG", rows, cols, data.data()));
        EXPECT_THROW((void) nn::dataLoaders::EEGMatSession(bad_dims), std::runtime_error);
        remove(bad_dims.c_str());
    }

    {
        std::string bad_type =
            std::filesystem::temp_directory_path().string() + "/tests_eeg_bad_type_session_tmp.mat";
        const size_t rows = 1;
        const size_t cols = 24579;
        std::vector<float> data(rows * cols, 1.0F);
        ASSERT_TRUE(writeMatFloat(bad_type, "EEG", rows, cols, data.data()));
        EXPECT_THROW((void) nn::dataLoaders::EEGMatSession(bad_type), std::runtime_error);
        remove(bad_type.c_str());
    }
}

TEST(EEGLoaderTest, StatefulLoaderRejectsDirectoryPath)
{
    nn::dataLoaders::EEGLoader loader;
    EXPECT_FALSE(loader.open(std::filesystem::temp_directory_path().string()));
}

TEST(EEGLoaderTest, SessionCacheEvictionPath)
{
    std::string fname =
        std::filesystem::temp_directory_path().string() + "/tests_eeg_cache_evict_tmp.mat";
    const size_t rows = 40;
    const size_t cols = 24579;

    std::vector<double> data(rows * cols, 0.0);
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t i = 0; i < 24576; ++i)
        {
            data[(i * rows) + r] = static_cast<double>(((r * 7U) + i) % 2048U);
        }
        data[(24576 * rows) + r] = static_cast<double>((r % 5U) + 1U);
        data[((24576 + 1) * rows) + r] = static_cast<double>((r % 11U) + 1U);
        data[((24576 + 2) * rows) + r] = static_cast<double>(r % 2U);
    }

    ASSERT_TRUE(writeMatDouble(fname, "EEG", rows, cols, data.data()));

    nn::dataLoaders::EEGMatSession session(fname);
    for (size_t r = 0; r < rows; ++r)
    {
        auto [tensor, labels] = session.readRow(r);
        EXPECT_EQ(tensor.rows(), 6);
        EXPECT_EQ(tensor.cols(), 4096);
        EXPECT_EQ(labels[0], static_cast<int>((r % 5U) + 1U));
    }

    auto [tensor0, labels0] = session.readRow(0);
    EXPECT_EQ(tensor0.rows(), 6);
    EXPECT_EQ(labels0[0], 1);

    remove(fname.c_str());
}
