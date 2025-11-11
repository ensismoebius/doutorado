#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>

#include "../MatFileDataset.h"

using namespace matioCpp;

TEST(MatFileDataset, can_load_data)
{
    // Remove any leftover file from previous runs
    std::filesystem::remove("test.mat");

    // Prepare a small 2x3 inputs matrix (column-major layout expected by matio-cpp)
    std::vector<double> inputs_raw = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    File file = File::Create("test.mat");
    MultiDimensionalArray<double> inputs("inputs", {2, 3}, inputs_raw.data());
    file.write(inputs);

    // Prepare a 2x1 targets matrix
    std::vector<double> targets_raw = {1.0, 0.0};
    MultiDimensionalArray<double> targets("targets", {2, 1}, targets_raw.data());
    file.write(targets);

    file.close();

    MatFileDataset dataset("test.mat", "inputs", "targets");
    EXPECT_EQ(dataset.size(), 2);

    auto batch = dataset.get_item(0);
    EXPECT_EQ(batch.inputs.get_shape()[0], 1);
    EXPECT_EQ(batch.inputs.get_shape()[1], 3);
    EXPECT_EQ(batch.targets.get_shape()[0], 1);
    EXPECT_EQ(batch.targets.get_shape()[1], 1);
}
