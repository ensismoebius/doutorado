#include "nn/dataLoaders/SqliteBatchSource.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "nn/utility/batching.hpp"

using std::string;

SqliteBatchSource::SqliteBatchSource(
    const string& db_root, std::unique_ptr<IBatchSource> underlying)
    : underlying_(std::move(underlying))
{
    db_path_ = (std::filesystem::path(db_root) / "database.sqlite").string();
    open_db();
}

SqliteBatchSource::~SqliteBatchSource()
{
    close_db();
}

bool SqliteBatchSource::open_db()
{
    if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK)
    {
        db_ = nullptr;
        return false;
    }

    // Prepare statements that operate on the provided schema.
    const char* pop_trial_sql = "SELECT id FROM trial ORDER BY id LIMIT 1;";
    sqlite3_prepare_v2(db_, pop_trial_sql, -1, &pop_trial_stmt_, nullptr);

    const char* select_eeg_sql =
        "SELECT F3,F4,C3,C4,P3,P4,blink FROM eeg_samples WHERE trial_id = ? ORDER BY id;";
    sqlite3_prepare_v2(db_, select_eeg_sql, -1, &select_eeg_stmt_, nullptr);

    const char* select_audio_sql =
        "SELECT samples FROM audio_samples WHERE trial_id = ? ORDER BY audio_row;";
    sqlite3_prepare_v2(db_, select_audio_sql, -1, &select_audio_stmt_, nullptr);

    return true;
}

void SqliteBatchSource::close_db()
{
    if (pop_trial_stmt_)
    {
        sqlite3_finalize(pop_trial_stmt_);
        pop_trial_stmt_ = nullptr;
    }
    if (select_eeg_stmt_)
    {
        sqlite3_finalize(select_eeg_stmt_);
        select_eeg_stmt_ = nullptr;
    }
    if (select_audio_stmt_)
    {
        sqlite3_finalize(select_audio_stmt_);
        select_audio_stmt_ = nullptr;
    }
    if (db_)
    {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

void SqliteBatchSource::reset_epoch(std::size_t epoch)
{
    if (underlying_) underlying_->reset_epoch(epoch);
}

bool SqliteBatchSource::next(Batch& out)
{
    // Try to consume an existing trial from the DB first.
    if (db_ && pop_trial_stmt_)
    {
        int rc = sqlite3_step(pop_trial_stmt_);
        if (rc == SQLITE_ROW)
        {
            const int trial_id = sqlite3_column_int(pop_trial_stmt_, 0);

            std::vector<float> eeg_accum;
            std::vector<float> audio_accum;

            // Read eeg_samples rows for this trial
            sqlite3_reset(select_eeg_stmt_);
            sqlite3_bind_int(select_eeg_stmt_, 1, trial_id);
            while ((rc = sqlite3_step(select_eeg_stmt_)) == SQLITE_ROW)
            {
                // Columns: F3,F4,C3,C4,P3,P4,blink (blobs may be NULL if not present)
                for (int c = 0; c < 6; ++c)
                {
                    const void* blob = sqlite3_column_blob(select_eeg_stmt_, c);
                    int bytes = sqlite3_column_bytes(select_eeg_stmt_, c);
                    if (blob && bytes > 0)
                    {
                        const size_t count = static_cast<size_t>(bytes) / sizeof(float);
                        const float* fptr = static_cast<const float*>(blob);
                        eeg_accum.insert(eeg_accum.end(), fptr, fptr + count);
                    }
                }
            }
            sqlite3_reset(select_eeg_stmt_);

            // Read audio_samples rows for this trial
            sqlite3_reset(select_audio_stmt_);
            sqlite3_bind_int(select_audio_stmt_, 1, trial_id);
            while ((rc = sqlite3_step(select_audio_stmt_)) == SQLITE_ROW)
            {
                const void* blob = sqlite3_column_blob(select_audio_stmt_, 0);
                int bytes = sqlite3_column_bytes(select_audio_stmt_, 0);
                if (blob && bytes > 0)
                {
                    const size_t count = static_cast<size_t>(bytes) / sizeof(float);
                    const float* fptr = static_cast<const float*>(blob);
                    audio_accum.insert(audio_accum.end(), fptr, fptr + count);
                }
            }
            sqlite3_reset(select_audio_stmt_);

            // Reset the pop-trial stmt for the next call
            sqlite3_reset(pop_trial_stmt_);

            // Build tensors: use single-row tensors with flattened data for now
            out.inputs = nn::Tensor(1, static_cast<int>(eeg_accum.size()));
            if (!eeg_accum.empty())
            {
                std::memcpy(out.inputs.mutable_data_ptr(),
                    eeg_accum.data(),
                    eeg_accum.size() * sizeof(float));
            }

            out.targets = nn::Tensor(1, static_cast<int>(audio_accum.size()));
            if (!audio_accum.empty())
            {
                std::memcpy(out.targets.mutable_data_ptr(),
                    audio_accum.data(),
                    audio_accum.size() * sizeof(float));
            }

            return true;
        }
        // No trial available
        sqlite3_reset(pop_trial_stmt_);
    }

    // Fall back to underlying source
    if (!underlying_) return false;
    return underlying_->next(out);
}
