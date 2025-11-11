#include <gtest/gtest.h>
#include <matio.h>

#include <string>
#include <vector>

#include "EEGLoader.h"

using namespace nn::dataLoaders;

TEST(EEGLoaderEdgeTest, MultiRowFileLoadsCorrectRow)
{
    std::string fname = "./tests_eeg_multi_tmp.mat";
    const size_t rows = 3;
    const size_t cols = 24579;

    std::vector<double> data(rows * cols, 0.0);

    // Fill data for three rows with different labels
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t i = 0; i < 24576; ++i)
            data[i * rows + r] = static_cast<double>(i + 1 + (r * 1000));
        data[24576 * rows + r] = static_cast<double>(r + 1);        // modality
        data[(24576 + 1) * rows + r] = static_cast<double>(r + 10); // stimulus
        data[(24576 + 2) * rows + r] = static_cast<double>(r % 2);  // artifact
    }

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

    // Load row 1 and check labels
    auto [m1, labels1] = loadEEGFromMat(fname, 1);
    EXPECT_EQ(labels1[0], 2);
    EXPECT_EQ(labels1[1], 11);
    EXPECT_EQ(labels1[2], 1);

    // Load row 2 and check labels
    auto [m2, labels2] = loadEEGFromMat(fname, 2);
    EXPECT_EQ(labels2[0], 3);
    EXPECT_EQ(labels2[1], 12);
    EXPECT_EQ(labels2[2], 0);

    remove(fname.c_str());
}

TEST(EEGLoaderEdgeTest, WrongTypeThrows)
{
    std::string fname = "./tests_eeg_badtype_tmp.mat";
    const size_t rows = 1;
    const size_t cols = 24579;
    std::vector<float> data(rows * cols, 1.0f);

    mat_t* mat = Mat_CreateVer(fname.c_str(), NULL, MAT_FT_MAT5);
    ASSERT_NE(mat, nullptr);
    size_t dims[2];
    dims[0] = rows;
    dims[1] = cols;
    // create single precision variable (MAT_C_SINGLE)
    matvar_t* matvar = Mat_VarCreate("EEG", MAT_C_SINGLE, MAT_T_SINGLE, 2, dims, data.data(), 0);
    ASSERT_NE(matvar, nullptr);
    int ok = Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
    EXPECT_EQ(ok, 0);
    Mat_VarFree(matvar);
    Mat_Close(mat);

    // Expect loader to throw because we require MAT_C_DOUBLE
    EXPECT_THROW({ auto p = loadEEGFromMat(fname, 0); }, std::runtime_error);

    remove(fname.c_str());
}

TEST(EEGLoaderEdgeTest, MissingVariableThrows)
{
    std::string fname = "./tests_eeg_missing_tmp.mat";
    const size_t rows = 1;
    const size_t cols = 10;
    std::vector<double> data(rows * cols, 1.0);

    mat_t* mat = Mat_CreateVer(fname.c_str(), NULL, MAT_FT_MAT5);
    ASSERT_NE(mat, nullptr);
    size_t dims[2];
    dims[0] = rows;
    dims[1] = cols;
    matvar_t* matvar = Mat_VarCreate("NotEEG", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, data.data(), 0);
    ASSERT_NE(matvar, nullptr);
    int ok = Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
    EXPECT_EQ(ok, 0);
    Mat_VarFree(matvar);
    Mat_Close(mat);

    // Since there is no numeric 2D var with correct columns, loader should throw
    EXPECT_THROW({ auto p = loadEEGFromMat(fname, 0); }, std::runtime_error);

    remove(fname.c_str());
}
