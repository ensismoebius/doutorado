#ifndef NN_DATALOADERS_SQLITEBATCHSOURCE_HPP
#define NN_DATALOADERS_SQLITEBATCHSOURCE_HPP

#include <sqlite3.h>

#include <memory>
#include <string>

#include "nn/dataLoaders/IBatchSource.hpp"

class SqliteBatchSource : public IBatchSource
{
   public:
    // db_root: directory where prefetch_cache.db will be created/opened
    // underlying: source used to fetch new batches when DB is empty
    SqliteBatchSource(const std::string& db_root, std::unique_ptr<IBatchSource> underlying);
    ~SqliteBatchSource() override;

    bool next(Batch& out) override;
    void reset_epoch(std::size_t epoch) override;

   private:
    bool open_db();
    void close_db();

    std::unique_ptr<IBatchSource> underlying_;
    std::string db_path_;
    sqlite3* db_ = nullptr;
    sqlite3_stmt* pop_trial_stmt_ = nullptr;
    sqlite3_stmt* select_eeg_stmt_ = nullptr;
    sqlite3_stmt* select_audio_stmt_ = nullptr;
};

#endif // NN_DATALOADERS_SQLITEBATCHSOURCE_HPP
