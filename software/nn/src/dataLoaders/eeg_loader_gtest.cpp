#include <gtest/gtest.h>
#include <matio.h>

#include <string>
#include <vector>

#include "EEGLoader.h"

using namespace nn::dataLoaders;

TEST(EEGLoaderTest, LoadSingleRow)
{
    // Create temporary filename
    std::string fname = "./tests_eeg_tmp.mat";

    // Prepare data: 1 row x 24579 columns
    const size_t rows = 1;
    const size_t cols = 24579; // 24576 samples + 3 labels

    std::vector<double> data(rows * cols, 0.0);

    // Fill sample region with incremental values and labels
    for (size_t i = 0; i < 24576; ++i) data[i * rows + 0] = static_cast<double>(i + 1);

    // labels
    data[24576 * rows + 0] = 7;        // modality
    data[(24576 + 1) * rows + 0] = 42; // stimulus
    data[(24576 + 2) * rows + 0] = 0;  // artifact

    // Write MAT file
    mat_t* mat = Mat_CreateVer(fname.c_str(), NULL, MAT_FT_MAT5);
    ASSERT_NE(mat, nullptr);

    size_t dims[2];
    dims[0] = rows;
    dims[1] = cols;
    matvar_t* matvar = Mat_VarCreate("EEG", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, data.data(), 0);
    ASSERT_NE(matvar, nullptr);

    int ok = Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
    EXPECT_EQ(ok, 0);

    Mat_VarFree(matvar);
    Mat_Close(mat);

    // Now load using our loader
    auto [matrix, labels] = loadEEGFromMat(fname, 0);

    // matrix should be 6 x 4096
    EXPECT_EQ(matrix.rows(), 6);
    EXPECT_EQ(matrix.cols(), 4096);

    // Check first few sample values for channel 0
    EXPECT_FLOAT_EQ(matrix(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(matrix(0, 1), 2.0f);

    // Check labels
    EXPECT_EQ(labels[0], 7);
    EXPECT_EQ(labels[1], 42);
    EXPECT_EQ(labels[2], 0);

    // Clean up
    remove(fname.c_str());
}
