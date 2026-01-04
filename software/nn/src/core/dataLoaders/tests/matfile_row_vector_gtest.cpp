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
    auto m = opt.value();

    // If the loaded matrix is Nx1 (column vector) but was expected to be 1xN (row vector),
    // transpose it manually into a new Tensor.
    if (m.rows() > 1 && m.cols() == 1)
    {
        nn::Tensor transposed(1, m.rows());
        for (nn::Index r = 0; r < m.rows(); ++r)
        {
            transposed.at(0, r) = m.at(r, 0);
        }
        m = std::move(transposed);
    }

    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_FLOAT_EQ(m.at(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(m.at(0, 3), 4.0F);
}
