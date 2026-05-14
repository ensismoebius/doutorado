#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "../lib/include/E04Output.hpp"
#include "../lib/include/E04EpochHistory.hpp"
#include "../lib/include/E04ResultRow.hpp"
#include "../lib/include/E04RunMetrics.hpp"

namespace fs = std::filesystem;
using namespace e04;

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

    EpochHistory create_sample_epoch_history()
    {
        EpochHistory history;
        // 3 epochs
        history.epoch_nums = {1.0f, 2.0f, 3.0f};
        history.train_losses = {0.5f, 0.3f, 0.2f};
        history.val_losses = {0.55f, 0.32f, 0.22f};

        // 30 batches (10 per epoch)
        for (int i = 0; i < 30; ++i)
        {
            history.batch_losses.push_back(0.5f - i * 0.01f);
            history.batch_epochs.push_back(static_cast<float>((i / 10) + 1));
        }

        return history;
    }

    RunMetrics create_sample_metrics()
    {
        RunMetrics metrics;
        metrics.mse = 0.025f;
        metrics.mae = 0.112f;
        metrics.r2 = 0.77f;
        metrics.f1 = 0.85f;
        metrics.spike_rate = 0.42f;
        metrics.energy = 650.0f;
        metrics.infer_ms = 420.0f;
        metrics.train_ms = 1200.0f;
        metrics.parameter_count = 18000;
        metrics.macs = 35000;
        return metrics;
    }

    ResultRow create_sample_result_row()
    {
        ResultRow row;
        row.backend = "xtensor";
        row.profile = "test_run";
        row.dataset = "fsdd";
        row.model = "snn-ae";
        row.encoding = "direct";
        row.architecture = "dense";
        row.layers = 2;
        row.v_th = 1.0f;
        row.alpha = 0.9f;
        row.run_id = 1;
        row.seed = 42u;
        row.config_hash = 12345u;
        row.metrics = create_sample_metrics();
        return row;
    }
};

// Test write_epoch_history_dat
TEST_F(DATWriterTest, EpochHistoryDatFormat)
{
    EpochHistory history = create_sample_epoch_history();
    fs::path output_file = test_dir / "epoch_history.dat";

    write_epoch_history_dat(output_file, "snn-ae", "direct", "dense", 1.0f, 0.9f, 1, history);

    ASSERT_TRUE(fs::exists(output_file)) << "DAT file not created";

    auto lines = read_file_lines(output_file);
    ASSERT_GT(lines.size(), 2) << "File should have header and data lines";

    // Check header line (starts with #)
    EXPECT_TRUE(lines[0].find('#') == 0) << "First line should be comment";
    EXPECT_TRUE(lines[0].find("model=snn-ae") != std::string::npos);
    EXPECT_TRUE(lines[0].find("encoding=direct") != std::string::npos);

    // Check column header
    EXPECT_EQ(lines[1], "epoch train_loss val_loss") << "Column header mismatch";

    // Check data rows match epoch count
    EXPECT_EQ(lines.size(), 5) << "Should have header comment + column header + 3 data rows";

    // Verify data format (should be parseable as floats)
    for (size_t i = 2; i < lines.size(); ++i)
    {
        std::istringstream iss(lines[i]);
        float epoch, train_loss, val_loss;
        EXPECT_TRUE(iss >> epoch >> train_loss >> val_loss)
            << "Line " << i << " not parseable as epoch train_loss val_loss";
    }
}

// Test write_batch_convergence_dat
TEST_F(DATWriterTest, BatchConvergenceDatFormat)
{
    EpochHistory history = create_sample_epoch_history();
    fs::path output_file = test_dir / "convergence.dat";

    write_batch_convergence_dat(output_file, "lstm-ae", "poisson", "", 0.0f, 0.0f, 1, history);

    ASSERT_TRUE(fs::exists(output_file)) << "DAT file not created";

    auto lines = read_file_lines(output_file);
    ASSERT_GT(lines.size(), 2) << "File should have header and data lines";

    // Check header
    EXPECT_TRUE(lines[0].find('#') == 0) << "First line should be comment";
    EXPECT_EQ(lines[1], "batch_num epoch batch_loss") << "Column header mismatch";

    // Batch count should match history
    int expected_rows = static_cast<int>(history.batch_losses.size());
    EXPECT_EQ(lines.size(), expected_rows + 2)
        << "Should have header + column header + " << expected_rows << " batch rows";

    // Verify pgfplots format (numeric columns)
    for (size_t i = 2; i < lines.size(); ++i)
    {
        std::istringstream iss(lines[i]);
        int batch_num;
        int epoch;
        float batch_loss;
        EXPECT_TRUE(iss >> batch_num >> epoch >> batch_loss) << "Line " << i << " not parseable";
    }
}

// Test write_pgfplots_summary_dat
TEST_F(DATWriterTest, PgfplotsSummaryDatFormat)
{
    std::vector<ResultRow> rows;
    // Create multiple rows with same config to ensure aggregation works
    for (int r = 0; r < 3; ++r)
    {
        ResultRow row = create_sample_result_row();
        row.run_id = r + 1;
        rows.push_back(row);
    }

    fs::path output_file = test_dir / "summary.dat";
    write_pgfplots_summary_dat(output_file, rows);

    ASSERT_TRUE(fs::exists(output_file)) << "DAT file not created";

    auto lines = read_file_lines(output_file);
    ASSERT_GE(lines.size(), 1) << "File should have at least header line";

    // First line is column header (NO comment line in this writer)
    auto& column_line = lines[0];
    EXPECT_TRUE(column_line.find("model") != std::string::npos);
    EXPECT_TRUE(column_line.find("encoding") != std::string::npos);
    EXPECT_TRUE(column_line.find("mse") != std::string::npos);

    // If there are data rows, verify format
    if (lines.size() > 1)
    {
        std::istringstream iss(lines[1]);
        std::string model, encoding, architecture;
        float v_th, alpha, mse;
        EXPECT_TRUE(iss >> model >> encoding >> architecture >> v_th >> alpha >> mse)
            << "First data row not parseable";
    }
}

// Test write_pgfplots_sweep_dat
TEST_F(DATWriterTest, PgfplotsSweepDatFormat)
{
    std::vector<ResultRow> rows;
    // Create results with varying alpha values (sweep only uses SNN-AE with direct encoding, v_th~1.0)
    float alpha_vals[] = {0.8f, 0.9f, 0.99f};
    std::string archs[] = {"dense", "conv1d", "recurrent"};

    for (float alpha : alpha_vals)
    {
        for (const auto& arch : archs)
        {
            ResultRow row = create_sample_result_row();
            row.model = "snn-ae";          // Only SNN-AE considered
            row.alpha = alpha;
            row.architecture = arch;
            row.encoding = "direct";       // Only direct encoding
            row.v_th = 1.0f;               // Only v_th ~= 1.0
            rows.push_back(row);
        }
    }

    fs::path output_file = test_dir / "sweep.dat";
    write_pgfplots_sweep_dat(output_file, rows);

    ASSERT_TRUE(fs::exists(output_file)) << "DAT file not created";

    auto lines = read_file_lines(output_file);
    ASSERT_GT(lines.size(), 1) << "File should have header and data lines";

    // First line is column header
    auto& column_line = lines[0];
    EXPECT_TRUE(column_line.find("alpha") != std::string::npos);
    EXPECT_TRUE(column_line.find("mse_dense") != std::string::npos);
    EXPECT_TRUE(column_line.find("energy_dense") != std::string::npos);

    // Check data rows
    for (size_t i = 1; i < lines.size(); ++i)
    {
        std::istringstream iss(lines[i]);
        float alpha, mse_dense, mse_conv1d, mse_recurrent, energy_dense, energy_conv1d, energy_recurrent;
        EXPECT_TRUE(iss >> alpha >> mse_dense >> mse_conv1d >> mse_recurrent >> energy_dense >>
                        energy_conv1d >> energy_recurrent)
            << "Line " << i << " not parseable";
    }
}

// Test file permissions and creation
TEST_F(DATWriterTest, FileCreation)
{
    EpochHistory history = create_sample_epoch_history();
    fs::path output_file = test_dir / "test_creation.dat";

    EXPECT_FALSE(fs::exists(output_file)) << "File should not exist before write";

    write_epoch_history_dat(output_file, "test", "direct", "", 0.0f, 0.0f, 1, history);

    EXPECT_TRUE(fs::exists(output_file)) << "File should exist after write";
    EXPECT_GT(fs::file_size(output_file), 0) << "File should have content";
}

// Test numeric precision
TEST_F(DATWriterTest, NumericPrecision)
{
    EpochHistory history;
    history.epoch_nums = {1.0f};
    history.train_losses = {0.123456789f};
    history.val_losses = {0.987654321f};
    history.batch_losses = {0.555555555f};
    history.batch_epochs = {1.0f};

    fs::path epoch_file = test_dir / "precision_epoch.dat";
    fs::path batch_file = test_dir / "precision_batch.dat";

    write_epoch_history_dat(epoch_file, "test", "direct", "", 0.0f, 0.0f, 1, history);
    write_batch_convergence_dat(batch_file, "test", "direct", "", 0.0f, 0.0f, 1, history);

    // Check epoch file precision
    auto epoch_lines = read_file_lines(epoch_file);
    std::istringstream epoch_iss(epoch_lines[2]);
    float epoch, train_loss, val_loss;
    epoch_iss >> epoch >> train_loss >> val_loss;

    // Should preserve at least 6 decimal places
    EXPECT_NEAR(train_loss, 0.123457f, 0.000001f);
    EXPECT_NEAR(val_loss, 0.987654f, 0.000001f);

    // Check batch file precision
    auto batch_lines = read_file_lines(batch_file);
    std::istringstream batch_iss(batch_lines[2]);
    int batch_num, epoch_num;
    float batch_loss;
    batch_iss >> batch_num >> epoch_num >> batch_loss;

    EXPECT_NEAR(batch_loss, 0.555556f, 0.000001f);
}

// Test empty history handling
TEST_F(DATWriterTest, EmptyHistoryHandling)
{
    EpochHistory empty_history;
    fs::path output_file = test_dir / "empty.dat";

    // Should not crash with empty history
    write_epoch_history_dat(output_file, "test", "direct", "", 0.0f, 0.0f, 1, empty_history);

    EXPECT_TRUE(fs::exists(output_file));
    auto lines = read_file_lines(output_file);

    // Should have header and column line only
    EXPECT_GE(lines.size(), 2);
}

// Test special characters in metadata
TEST_F(DATWriterTest, SpecialCharactersInMetadata)
{
    EpochHistory history = create_sample_epoch_history();
    fs::path output_file = test_dir / "special_chars.dat";

    // Write with various special characters in strings
    write_epoch_history_dat(output_file, "snn-ae", "direct", "conv1d", 1.0f, 0.9f, 1, history);

    auto lines = read_file_lines(output_file);
    EXPECT_TRUE(lines[0].find("conv1d") != std::string::npos);

    // Data should still be parseable
    std::istringstream iss(lines[2]);
    float e, t, v;
    EXPECT_TRUE(iss >> e >> t >> v);
}

// Test large dataset handling
TEST_F(DATWriterTest, LargeDataset)
{
    EpochHistory history;

    // Create large history (1000 batches over 10 epochs)
    for (int epoch = 1; epoch <= 10; ++epoch)
    {
        history.epoch_nums.push_back(static_cast<float>(epoch));
        history.train_losses.push_back(0.5f - epoch * 0.01f);
        history.val_losses.push_back(0.55f - epoch * 0.01f);
    }

    for (int i = 0; i < 1000; ++i)
    {
        history.batch_losses.push_back(0.5f - i * 0.0005f);
        history.batch_epochs.push_back(static_cast<float>((i / 100) + 1));
    }

    fs::path output_file = test_dir / "large.dat";
    write_batch_convergence_dat(output_file, "test", "direct", "", 0.0f, 0.0f, 1, history);

    auto lines = read_file_lines(output_file);
    // Comment + column header + 1000 data rows
    EXPECT_EQ(lines.size(), 1002);
}

// Test consistency between writes
TEST_F(DATWriterTest, ConsistencyBetweenWrites)
{
    EpochHistory history1 = create_sample_epoch_history();
    EpochHistory history2 = create_sample_epoch_history();

    fs::path file1 = test_dir / "consistency1.dat";
    fs::path file2 = test_dir / "consistency2.dat";

    write_epoch_history_dat(file1, "test", "direct", "", 0.0f, 0.0f, 1, history1);
    write_epoch_history_dat(file2, "test", "direct", "", 0.0f, 0.0f, 1, history2);

    auto lines1 = read_file_lines(file1);
    auto lines2 = read_file_lines(file2);

    // Skip comment line (different timestamps possible), but data should match
    ASSERT_EQ(lines1[1], lines2[1]) << "Column headers should match";

    for (size_t i = 2; i < lines1.size(); ++i)
    {
        EXPECT_EQ(lines1[i], lines2[i]) << "Data row " << i << " should match";
    }
}
