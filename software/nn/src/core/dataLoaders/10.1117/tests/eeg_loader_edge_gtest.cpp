#include <gtest/gtest.h>

#include <string>
#include <vector>
#include <filesystem>

#include "../../tests/MatTestUtils/MatTestUtils.h"
#include "../EEGLoader.h"

using namespace nn::dataLoaders;
using namespace nn::dataLoaders::test;

TEST(EEGLoaderEdgeTest, MultiRowFileLoadsCorrectRow)
{
    std::string fname = std::filesystem::temp_directory_path().string() + "/tests_eeg_multi_tmp.mat";
    const size_t rows = 3;
    const size_t cols = 24579;

    std::vector<double> data(rows * cols, 0.0);
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t i = 0; i < 24576; ++i)
        {
            data[(i * rows) + r] = static_cast<double>((i + 1) + (r * 1000));
        }
        data[(24576 * rows) + r] = static_cast<double>(r + 1);
        data[((24576 + 1) * rows) + r] = static_cast<double>(r + 10);
        data[((24576 + 2) * rows) + r] = static_cast<double>(r % 2);
    }

    ASSERT_TRUE(writeMatDouble(fname, "EEG", rows, cols, data.data()));

    auto [m1, labels1] = loadEEGFromMat(fname, 1);
    EXPECT_EQ(labels1[0], 2);
    EXPECT_EQ(labels1[1], 11);
    EXPECT_EQ(labels1[2], 1);

    auto [m2, labels2] = loadEEGFromMat(fname, 2);
    EXPECT_EQ(labels2[0], 3);
    EXPECT_EQ(labels2[1], 12);
    EXPECT_EQ(labels2[2], 0);

    remove(fname.c_str());
}

TEST(EEGLoaderEdgeTest, WrongTypeThrows)
{
    std::string fname = std::filesystem::temp_directory_path().string() + "/tests_eeg_badtype_tmp.mat";
    const size_t rows = 1;
    const size_t cols = 24579;
    std::vector<float> data(rows * cols, 1.0F);

    ASSERT_TRUE(writeMatFloat(fname, "EEG", rows, cols, data.data()));

    EXPECT_THROW({ auto p = loadEEGFromMat(fname, 0); }, std::runtime_error);

    remove(fname.c_str());
}

TEST(EEGLoaderEdgeTest, MissingVariableThrows)
{
    std::string fname = std::filesystem::temp_directory_path().string() + "/tests_eeg_missing_tmp.mat";
    const size_t rows = 1;
    const size_t cols = 10;
    std::vector<double> data(rows * cols, 1.0);

    ASSERT_TRUE(writeMatDouble(fname, "NotEEG", rows, cols, data.data()));

    EXPECT_THROW({ auto p = loadEEGFromMat(fname, 0); }, std::runtime_error);

    remove(fname.c_str());
}
