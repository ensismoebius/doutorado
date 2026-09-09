#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "../lib/include/GuayaquilCheckpoint.hpp"

namespace fs = std::filesystem;
using namespace guayaquil;

namespace
{

// Build a fully-populated ResultRow for round-trip testing.
ResultRow make_row()
{
    ResultRow r;
    r.backend = "xtensor";
    r.profile = "test-profile";
    r.dataset = "fsdd";
    r.model = "lstm-ae";
    r.encoding = "direct";
    r.architecture = "lstm";
    r.layers = 2;
    r.v_th = 0.0f;
    r.alpha = 0.0f;
    r.run_id = 1;
    r.seed = 42u;
    r.config_hash = 0xDEADBEEFCAFE1234ULL;

    r.metrics.mse = 0.021f;
    r.metrics.mae = 0.102f;
    r.metrics.r2 = 0.81f;
    r.metrics.precision = 0.90f;
    r.metrics.recall = 0.88f;
    r.metrics.f1 = 0.89f;
    r.metrics.spike_rate = 0.0f;
    r.metrics.energy = 85238400.0f;
    r.metrics.train_ms = 27421.9f;
    r.metrics.infer_ms = 337.7f;
    r.metrics.parameter_count = 120225u;
    r.metrics.macs = 8523840u;
    return r;
}

EpochHistory make_history()
{
    EpochHistory h;
    h.epoch_nums = {1.0f, 2.0f, 3.0f};
    h.train_losses = {0.50f, 0.30f, 0.20f};
    h.val_losses = {0.55f, 0.32f, 0.22f};
    h.batch_losses = {0.60f, 0.45f, 0.35f, 0.28f, 0.22f};
    h.batch_epochs = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f};
    return h;
}

CheckpointKey make_key()
{
    return {"test-run", "xtensor", "fsdd", "lstm-ae", "direct", "lstm", 0.0f, 0.0f, 1};
}

} // namespace

// ──────────────────────────────────────────────────────────────────────────────
// 1. Round-trip: save then load produces identical ResultRow
// ──────────────────────────────────────────────────────────────────────────────
TEST(CheckpointRoundTrip, MetricsAndFieldsMatch)
{
    const fs::path dir = fs::temp_directory_path() / "chk_test_roundtrip";
    fs::create_directories(dir);

    const ResultRow original = make_row();
    const EpochHistory hist = make_history();
    const std::size_t hash = original.config_hash;

    const auto path = checkpoint_path(dir, make_key());
    checkpoint_save(path, original, hist, hash);

    ASSERT_TRUE(fs::exists(path));

    const ResultRow loaded = checkpoint_load(path);

    EXPECT_EQ(loaded.backend, original.backend);
    EXPECT_EQ(loaded.profile, original.profile);
    EXPECT_EQ(loaded.dataset, original.dataset);
    EXPECT_EQ(loaded.model, original.model);
    EXPECT_EQ(loaded.encoding, original.encoding);
    EXPECT_EQ(loaded.architecture, original.architecture);
    EXPECT_EQ(loaded.layers, original.layers);
    EXPECT_FLOAT_EQ(loaded.v_th, original.v_th);
    EXPECT_FLOAT_EQ(loaded.alpha, original.alpha);
    EXPECT_EQ(loaded.run_id, original.run_id);
    EXPECT_EQ(loaded.seed, original.seed);
    EXPECT_EQ(loaded.config_hash, original.config_hash);

    EXPECT_FLOAT_EQ(loaded.metrics.mse, original.metrics.mse);
    EXPECT_FLOAT_EQ(loaded.metrics.mae, original.metrics.mae);
    EXPECT_FLOAT_EQ(loaded.metrics.r2, original.metrics.r2);
    EXPECT_FLOAT_EQ(loaded.metrics.precision, original.metrics.precision);
    EXPECT_FLOAT_EQ(loaded.metrics.recall, original.metrics.recall);
    EXPECT_FLOAT_EQ(loaded.metrics.f1, original.metrics.f1);
    EXPECT_FLOAT_EQ(loaded.metrics.train_ms, original.metrics.train_ms);
    EXPECT_FLOAT_EQ(loaded.metrics.infer_ms, original.metrics.infer_ms);
    EXPECT_EQ(loaded.metrics.parameter_count, original.metrics.parameter_count);
    EXPECT_EQ(loaded.metrics.macs, original.metrics.macs);

    fs::remove_all(dir);
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Hash mismatch: is_valid returns false
// ──────────────────────────────────────────────────────────────────────────────
TEST(CheckpointHashMismatch, ReturnsFalse)
{
    const fs::path dir = fs::temp_directory_path() / "chk_test_hash";
    fs::create_directories(dir);

    const std::size_t saved_hash = 0x1111111111111111ULL;
    const std::size_t wrong_hash = 0x2222222222222222ULL;

    const auto path = checkpoint_path(dir, make_key());
    checkpoint_save(path, make_row(), make_history(), saved_hash);

    EXPECT_FALSE(checkpoint_is_valid(path, wrong_hash));

    fs::remove_all(dir);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Missing file: is_valid returns false (no exception)
// ──────────────────────────────────────────────────────────────────────────────
TEST(CheckpointMissingFile, ReturnsFalse)
{
    const fs::path nonexistent =
        fs::temp_directory_path() / "chk_test_missing" / "no_such_file.json";
    EXPECT_FALSE(checkpoint_is_valid(nonexistent, 42u));
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. Atomic write: no .tmp file left after save; real file is valid JSON
// ──────────────────────────────────────────────────────────────────────────────
TEST(CheckpointAtomicWrite, NoTmpFileAfterSave)
{
    const fs::path dir = fs::temp_directory_path() / "chk_test_atomic";
    fs::create_directories(dir);

    const std::size_t hash = 0xABCDEF0123456789ULL;
    const auto path = checkpoint_path(dir, make_key());
    const auto tmp_path = fs::path(path.string() + ".tmp");

    checkpoint_save(path, make_row(), make_history(), hash);

    EXPECT_TRUE(fs::exists(path)) << "real checkpoint file should exist";
    EXPECT_FALSE(fs::exists(tmp_path)) << ".tmp file should not remain after save";

    // Real file must be parseable JSON containing correct hash
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)), {});
    // Basic sanity: contains expected hash token
    EXPECT_NE(content.find("config_hash"), std::string::npos);
    EXPECT_NE(content.find("version"), std::string::npos);

    fs::remove_all(dir);
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. Stale file replaced: re-saving with new hash overwrites cleanly
// ──────────────────────────────────────────────────────────────────────────────
TEST(CheckpointStaleFileReplaced, NewHashValidOldHashInvalid)
{
    const fs::path dir = fs::temp_directory_path() / "chk_test_stale";
    fs::create_directories(dir);

    const std::size_t old_hash = 0xAAAAAAAAAAAAAAAAULL;
    const std::size_t new_hash = 0xBBBBBBBBBBBBBBBBULL;

    const auto path = checkpoint_path(dir, make_key());

    // Save with old hash
    ResultRow row = make_row();
    row.config_hash = old_hash;
    checkpoint_save(path, row, make_history(), old_hash);
    EXPECT_TRUE(checkpoint_is_valid(path, old_hash));
    EXPECT_FALSE(checkpoint_is_valid(path, new_hash));

    // Re-save with new hash
    row.config_hash = new_hash;
    checkpoint_save(path, row, make_history(), new_hash);
    EXPECT_FALSE(checkpoint_is_valid(path, old_hash))
        << "old hash should no longer validate after re-save";
    EXPECT_TRUE(checkpoint_is_valid(path, new_hash)) << "new hash should validate";

    fs::remove_all(dir);
}
