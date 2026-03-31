#ifndef NN_DATALOADERS_SQLITEBATCHSOURCE_HPP
#define NN_DATALOADERS_SQLITEBATCHSOURCE_HPP

#include <sqlite3.h>

#include <memory>
#include <string>

#include "nn/dataLoaders/10.1117/protocol/Protocol101117Dataset.hpp"
#include "nn/windowing/WindowSpec.hpp"

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

#include "nn/dataLoaders/IBatchSource.hpp"

class SqliteBatchSource : public IBatchSource
{
   public:
    // db_root: directory where database.sqlite will be created/opened
    // SqliteBatchSource is DB-only: it does not fall back to an underlying
    // IBatchSource. If the DB cannot provide compatible batches `next()`
    // returns false.
    SqliteBatchSource(const std::string& db_root,
        std::size_t batch_size = 1,
        nn::dataLoaders::SqliteDatasetType dataset_type =
            nn::dataLoaders::SqliteDatasetType::Protocol,
        const nn::windowing::WindowSpec& eeg_window = nn::windowing::WindowSpec{},
        const nn::windowing::WindowSpec& audio_window = nn::windowing::WindowSpec{},
        Protocol101117InputMode input_mode = Protocol101117InputMode::Concatenated);
    ~SqliteBatchSource() override;

    bool next(Batch& out) override;
    void reset_epoch(std::size_t epoch) override;

   private:
    bool open_db();
    void close_db();

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
};

#endif // NN_DATALOADERS_SQLITEBATCHSOURCE_HPP
