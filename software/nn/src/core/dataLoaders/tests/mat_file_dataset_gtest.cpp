#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>
#include <stdexcept> // For std::runtime_error or std::invalid_argument

#include "../MatFileDataset.h"

using namespace matioCpp;

class MatFileDatasetTestFixture : public ::testing::Test
{
   protected:
    const std::string test_filename = "test_dataset.mat";

    void SetUp() override
    {
        // Ensure a clean slate before each test
        std::filesystem::remove(test_filename);
    }

    void TearDown() override
    {
        // Clean up after each test
        std::filesystem::remove(test_filename);
    }

    // Helper to create a simple .mat file
    void create_mat_file(const std::vector<double>& inputs_data,
                         const std::vector<size_t>& inputs_shape, const std::string& inputs_name,
                         const std::vector<double>& targets_data,
                         const std::vector<size_t>& targets_shape, const std::string& targets_name)
    {
        File file = File::Create(test_filename);
        MultiDimensionalArray<double> inputs(inputs_name, inputs_shape, inputs_data.data());
        file.write(inputs);
        MultiDimensionalArray<double> targets(targets_name, targets_shape, targets_data.data());
        file.write(targets);
        file.close();
    }
};

TEST_F(MatFileDatasetTestFixture, CanLoadData)
{
    // Prepare a small 2x3 inputs matrix (column-major layout expected by matio-cpp)
    std::vector<double> inputs_raw = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    std::vector<size_t> inputs_shape = {2, 3}; // 2 rows, 3 cols

    // Prepare a 2x1 targets matrix
    std::vector<double> targets_raw = {1.0, 0.0};
    std::vector<size_t> targets_shape = {2, 1}; // 2 rows, 1 col

    create_mat_file(inputs_raw, inputs_shape, "inputs", targets_raw, targets_shape, "targets");

    MatFileDataset dataset(test_filename, "inputs", "targets");
    EXPECT_EQ(dataset.size(), 2);

    auto batch = dataset.get_item(0);
    EXPECT_EQ(batch.inputs.get_shape()[0], 1);
    EXPECT_EQ(batch.inputs.get_shape()[1], 3);
    EXPECT_EQ(batch.targets.get_shape()[0], 1);
    EXPECT_EQ(batch.targets.get_shape()[1], 1);

    // Verify values
    EXPECT_FLOAT_EQ(batch.inputs.at(0, 0), 1.0);
    EXPECT_FLOAT_EQ(batch.inputs.at(0, 1), 2.0);
    EXPECT_FLOAT_EQ(batch.inputs.at(0, 2), 3.0);
    EXPECT_FLOAT_EQ(batch.targets.at(0, 0), 1.0);

    batch = dataset.get_item(1);
    EXPECT_EQ(batch.inputs.get_shape()[0], 1);
    EXPECT_EQ(batch.inputs.get_shape()[1], 3);
    EXPECT_EQ(batch.targets.get_shape()[0], 1);
    EXPECT_EQ(batch.targets.get_shape()[1], 1);

    // Verify values
    EXPECT_FLOAT_EQ(batch.inputs.at(0, 0), 4.0);
    EXPECT_FLOAT_EQ(batch.inputs.at(0, 1), 5.0);
    EXPECT_FLOAT_EQ(batch.inputs.at(0, 2), 6.0);
    EXPECT_FLOAT_EQ(batch.targets.at(0, 0), 0.0);
}

TEST_F(MatFileDatasetTestFixture, ThrowsOnFileDoesNotExist)
{
    // The file "test.mat" will not be created by create_mat_file,
    // so it should not exist.
    ASSERT_FALSE(std::filesystem::exists(test_filename));
    ASSERT_THROW(MatFileDataset dataset(test_filename, "inputs", "targets"), std::runtime_error);
}

TEST_F(MatFileDatasetTestFixture, ThrowsOnInputsVariableDoesNotExist)
{
    // Create a file with only targets
    std::vector<double> targets_raw = {1.0, 0.0};
    std::vector<size_t> targets_shape = {2, 1};
    // No inputs data
    create_mat_file({}, {}, "non_existent_inputs", targets_raw, targets_shape, "targets");

    ASSERT_THROW(MatFileDataset dataset(test_filename, "inputs", "targets"), std::runtime_error);
}

TEST_F(MatFileDatasetTestFixture, ThrowsOnTargetsVariableDoesNotExist)
{
    // Create a file with only inputs
    std::vector<double> inputs_raw = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    std::vector<size_t> inputs_shape = {2, 3};
    // No targets data
    create_mat_file(inputs_raw, inputs_shape, "inputs", {}, {}, "non_existent_targets");

    ASSERT_THROW(MatFileDataset dataset(test_filename, "inputs", "targets"), std::runtime_error);
}

TEST_F(MatFileDatasetTestFixture, ThrowsOnMismatchedSampleCounts)
{
    // Inputs: 3 samples, Targets: 2 samples
    std::vector<double> inputs_raw = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<size_t> inputs_shape = {3, 2}; // 3 samples

    std::vector<double> targets_raw = {1.0, 0.0};
    std::vector<size_t> targets_shape = {2, 1}; // 2 samples

    create_mat_file(inputs_raw, inputs_shape, "inputs", targets_raw, targets_shape, "targets");

    ASSERT_THROW(MatFileDataset dataset(test_filename, "inputs", "targets"), std::runtime_error);
}

TEST_F(MatFileDatasetTestFixture, GetItemOutOfBoundsThrows)
{
    std::vector<double> inputs_raw = {1.0, 4.0};
    std::vector<size_t> inputs_shape = {2, 1};

    std::vector<double> targets_raw = {1.0, 0.0};
    std::vector<size_t> targets_shape = {2, 1};

    create_mat_file(inputs_raw, inputs_shape, "inputs", targets_raw, targets_shape, "targets");

    MatFileDataset dataset(test_filename, "inputs", "targets");
    EXPECT_EQ(dataset.size(), 2);

    ASSERT_THROW(dataset.get_item(2), std::out_of_range); // Index 2 is out of bounds for size 2
    ASSERT_THROW(dataset.get_item(100), std::out_of_range);
}

TEST_F(MatFileDatasetTestFixture, CollateWithValidIndices)
{
    std::vector<double> inputs_raw = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    std::vector<size_t> inputs_shape = {3, 2}; // 3 samples
    std::vector<double> targets_raw = {1.0, 0.0, 1.0};
    std::vector<size_t> targets_shape = {3, 1}; // 3 samples
    create_mat_file(inputs_raw, inputs_shape, "inputs", targets_raw, targets_shape, "targets");

    MatFileDataset dataset(test_filename, "inputs", "targets");
    EXPECT_EQ(dataset.size(), 3);

    std::vector<size_t> indices = {0, 2};
    Batch batch = dataset.collate(indices);

    EXPECT_EQ(batch.inputs.get_shape()[0], 2);
    EXPECT_EQ(batch.inputs.get_shape()[1], 2);
    EXPECT_EQ(batch.targets.get_shape()[0], 2);
    EXPECT_EQ(batch.targets.get_shape()[1], 1);

    EXPECT_FLOAT_EQ(batch.inputs.at(0, 0), 1.0);  // Original inputs.at(0,0)
    EXPECT_FLOAT_EQ(batch.inputs.at(0, 1), 5.0);  // Original inputs.at(0,1)
    EXPECT_FLOAT_EQ(batch.inputs.at(1, 0), 2.0);  // Original inputs.at(2,0)
    EXPECT_FLOAT_EQ(batch.inputs.at(1, 1), 6.0);  // Original inputs.at(2,1)
    EXPECT_FLOAT_EQ(batch.targets.at(0, 0), 1.0); // Original targets.at(0,0)
    EXPECT_FLOAT_EQ(batch.targets.at(1, 0), 1.0); // Original targets.at(2,0)
}

TEST_F(MatFileDatasetTestFixture, CollateWithEmptyIndices)
{
    std::vector<double> inputs_raw = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0}; // From CanLoadData
    std::vector<size_t> inputs_shape = {2, 3};                       // From CanLoadData
    std::vector<double> targets_raw = {1.0, 2.0};                    // Made 2x1
    std::vector<size_t> targets_shape = {2, 1};                      // Made 2x1
    create_mat_file(inputs_raw, inputs_shape, "inputs", targets_raw, targets_shape, "targets");

    MatFileDataset dataset(test_filename, "inputs", "targets");
    EXPECT_EQ(dataset.size(), 2); // Corrected expected size

    std::vector<size_t> indices = {};
    Batch batch = dataset.collate(indices);

    EXPECT_EQ(batch.inputs.get_shape()[0], 0);
    EXPECT_EQ(batch.inputs.get_shape()[1], 3); // Corrected expected size
    EXPECT_EQ(batch.targets.get_shape()[0], 0);
    EXPECT_EQ(batch.targets.get_shape()[1], 1);
}

TEST_F(MatFileDatasetTestFixture, CollateWithOutOfBoundsIndicesThrows)
{
    std::vector<double> inputs_raw = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0}; // From CanLoadData
    std::vector<size_t> inputs_shape = {2, 3};                       // From CanLoadData
    std::vector<double> targets_raw = {1.0, 2.0};                    // Made 2x1
    std::vector<size_t> targets_shape = {2, 1};                      // Made 2x1
    create_mat_file(inputs_raw, inputs_shape, "inputs", targets_raw, targets_shape, "targets");

    MatFileDataset dataset(test_filename, "inputs", "targets");
    EXPECT_EQ(dataset.size(), 2); // Corrected expected size

    std::vector<size_t> indices = {0, 2}; // Index 2 is out of bounds for size 2
    ASSERT_THROW(dataset.collate(indices), std::out_of_range);
}

TEST_F(MatFileDatasetTestFixture, LoadMinimalTargetsDirectly)
{
    // Create a simple inputs and targets
    std::vector<double> inputs_raw = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    std::vector<size_t> inputs_shape = {2, 3}; // 2 rows, 3 cols
    std::vector<double> targets_raw = {1.0, 2.0};
    std::vector<size_t> targets_shape = {2, 1}; // 2 rows, 1 col

    create_mat_file(inputs_raw, inputs_shape, "inputs", targets_raw, targets_shape, "targets");

    // Check that file was created
    ASSERT_TRUE(std::filesystem::exists(test_filename)) << "Test file was not created";

    auto loaded_targets_opt =
        matioCpp::utils::load_named_variable_as_matrix(test_filename, "targets");
    ASSERT_TRUE(loaded_targets_opt.has_value());
    EXPECT_EQ(loaded_targets_opt->rows(), 2);
    EXPECT_EQ(loaded_targets_opt->cols(), 1);
    EXPECT_FLOAT_EQ((*loaded_targets_opt)(0, 0), 1.0);
    EXPECT_FLOAT_EQ((*loaded_targets_opt)(1, 0), 2.0);

    auto loaded_inputs_opt =
        matioCpp::utils::load_named_variable_as_matrix(test_filename, "inputs");
    ASSERT_TRUE(loaded_inputs_opt.has_value());
    EXPECT_EQ(loaded_inputs_opt->rows(), 2);
    EXPECT_EQ(loaded_inputs_opt->cols(), 3);
}
