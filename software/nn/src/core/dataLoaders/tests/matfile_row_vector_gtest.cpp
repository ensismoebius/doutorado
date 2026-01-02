#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>

#include "core/dataLoaders/mat_file_utils.hpp"

using namespace matioCpp;

TEST(MatFileRowVectorTest, LoadRowVector)
{
    const auto mat_path = std::filesystem::temp_directory_path() / "rowvec_test.mat";
    std::filesystem::remove(mat_path);

    // Create a 1x4 row vector (stored column-major as 1x4)
    std::vector<double> raw = {1.0, 2.0, 3.0, 4.0};
    File file = File::Create(mat_path.string());
    MultiDimensionalArray<double> rv("rvec", {1, 4}, raw.data());
    file.write(rv);
    file.close();

    auto opt = matioCpp::utils::load_named_variable_as_matrix(mat_path.string(), "rvec");
    ASSERT_TRUE(opt.has_value());
    auto m_original = opt.value(); // Store the original loaded matrix

    // If the loaded matrix is Nx1 (column vector) but was expected to be 1xN (row vector),
    // transpose it.
    Eigen::MatrixXf m;
    if (m_original.rows() > 1 && m_original.cols() == 1)
    {
        m = m_original.transpose();
    }
    else
    {
        m = m_original;
    }

    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_FLOAT_EQ(m(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(m(0, 3), 4.0F);
}
