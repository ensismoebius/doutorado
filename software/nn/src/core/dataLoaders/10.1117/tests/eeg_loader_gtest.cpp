/**
 * @file eeg_loader_gtest.cpp
 * @brief Basic correctness tests for loading an EEG row into (channels x samples) form.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
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
