/**
 * @file include/data_loaders/sources/SqliteBatchSource.hpp
 * @brief Sqlitebatchsource.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_SQLITEBATCHSOURCE_HPP
#define NN_DATALOADERS_SQLITEBATCHSOURCE_HPP

#include <sqlite3.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "data_loaders/10.1117/datasets/raw/Dataset101117.hpp"
#include "windowing/WindowSpec.hpp"

namespace nn::dataLoaders
{
enum class SqliteDatasetType
{
    Protocol,
    EegWindow,
    AudioWindow,
    FusedWindow,
};
} // namespace nn::dataLoaders

#include "data_loaders/interfaces/IBatchSource.hpp"

class SqliteBatchSource : public IBatchSource
{
   public:
    // db_root: directory where database.sqlite will be created/opened
    // SqliteBatchSource is DB-only: it does not fall back to an underlying
    // IBatchSource. If the DB cannot provide compatible batches `next()`
    // returns false.
    explicit SqliteBatchSource(const std::string& db_root,
        std::size_t batch_size = 1,
        nn::dataLoaders::SqliteDatasetType dataset_type =
            nn::dataLoaders::SqliteDatasetType::Protocol,
        const nn::windowing::WindowSpec& eeg_window = nn::windowing::WindowSpec{},
        const nn::windowing::WindowSpec& audio_window = nn::windowing::WindowSpec{},
        Protocol101117InputMode input_mode = Protocol101117InputMode::Concatenated,
        std::vector<int> selected_trial_ids = {});
    ~SqliteBatchSource() override;

    bool next(Batch& out) override;
    void reset_epoch(std::size_t epoch) override;

   private:
    bool emit_pending_window_batch(Batch& out);
    bool open_db();
    void close_db();

    // open_db() helpers, extracted with no behavior change.
    bool prepareCoreStatements();
    void logJoinedTrialCount() const;
    void loadTrialIds();

    // next() helpers, extracted with no behavior change.
    //
    // Both return nullopt to tell the caller to `continue` to the next trial (incomplete
    // data, no windows produced, or a stacking failure), or a value to `return` immediately
    // from next() — mirroring the multiple continue/return points in the original inline
    // per-trial logic.
    void appendProtocolTrialRows(const nn::Tensor& stacked_resampled);
    std::optional<bool> processProtocolConcatenatedTrial(
        const std::vector<float>& eeg_accum, const std::vector<float>& audio_accum, Batch& out);
    std::optional<bool> processWindowedTrial(
        const std::vector<float>& eeg_accum, const std::vector<float>& audio_accum, Batch& out);

    std::string db_path_;
    sqlite3* db_ = nullptr;
    sqlite3_stmt* pop_trial_stmt_ = nullptr;
    sqlite3_stmt* select_eeg_stmt_ = nullptr;
    sqlite3_stmt* select_audio_stmt_ = nullptr;
    std::size_t batch_size_ = 1;
    // Windowing/dataset parameters
    nn::dataLoaders::SqliteDatasetType dataset_type_{};
    nn::windowing::WindowSpec eeg_window_{};
    nn::windowing::WindowSpec audio_window_{};
    Protocol101117InputMode input_mode_ = Protocol101117InputMode::Concatenated;
    std::vector<int> selected_trial_ids_filter_{};
    std::vector<int> trial_ids_{};
    std::size_t next_trial_index_ = 0;
    std::vector<std::vector<float>> pending_window_samples_{};
    std::size_t next_pending_sample_index_ = 0;
};

#endif // NN_DATALOADERS_SQLITEBATCHSOURCE_HPP
