/**
 * @file src/core/data_loaders/sources/SqliteBatchSource.cpp
 * @brief Implementation for Sqlitebatchsource.
 *

 */

#include "data_loaders/sources/SqliteBatchSource.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "data_loaders/10.1117/datasets/raw/SamplePacking.hpp"
#include "data_loaders/10.1117/schema/Metadata.hpp"
#include "logging/Logger.hpp"
#include "utility/batching.hpp"
#include "utility/path_expand.hpp"
#include "windowing/WindowSpec.hpp"

using std::string;

namespace
{
// Minimal internal diagnostics removed in non-debug builds to avoid noisy logs.

// Resolve a dataset root (a directory containing database.sqlite, or a direct
// .sqlite file path) to the actual sqlite file, expanding a leading `~/`.
std::string resolve_db_path(const std::string& root)
{
    const std::filesystem::path expanded(nn::utility::expand_home(root));
    if (expanded.extension() == ".sqlite")
    {
        return expanded.string();
    }
    return (expanded / "database.sqlite").string();
}
} // namespace

SqliteBatchSource::SqliteBatchSource(const string& db_root,
    std::size_t batch_size,
    nn::dataLoaders::SqliteDatasetType dataset_type,
    const nn::windowing::WindowSpec& eeg_window,
    const nn::windowing::WindowSpec& audio_window,
    Protocol101117InputMode input_mode,
    std::vector<int> selected_trial_ids)
    : db_path_(resolve_db_path(db_root)),
      batch_size_(batch_size),
      dataset_type_(dataset_type),
      eeg_window_(eeg_window),
      audio_window_(audio_window),
      input_mode_(input_mode),
      selected_trial_ids_filter_(std::move(selected_trial_ids))
{
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

    // Performance optimizations for faster I/O
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA cache_size=10000;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA mmap_size=268435456;", nullptr, nullptr, nullptr); // 256MB
    sqlite3_exec(db_, "PRAGMA temp_store=MEMORY;", nullptr, nullptr, nullptr);

    // Prepare statements that operate on the provided schema.
    // Select only trials that have both audio and eeg rows. Use DISTINCT
    // because joining with audio_samples and eeg_samples may produce
    // multiple rows per trial if there are multiple samples; we only need
    // a trial id that has both kinds of data.
    const char* pop_trial_sql =
        "SELECT DISTINCT t.id FROM trial t "
        "INNER JOIN audio_samples a ON a.trial_id = t.id "
        "INNER JOIN eeg_samples e ON e.trial_id = t.id "
        "ORDER BY t.id LIMIT 1;";
    int rc = sqlite3_prepare_v2(db_, pop_trial_sql, -1, &pop_trial_stmt_, nullptr);
    if (rc != SQLITE_OK)
    {
        NN_LOG_ERROR(std::string("SqliteBatchSource::open_db: prepare pop_trial_sql failed: ") +
                     sqlite3_errmsg(db_));
        pop_trial_stmt_ = nullptr;
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }

    const char* select_eeg_sql =
        "SELECT F3,F4,C3,C4,P3,P4,blink FROM eeg_samples WHERE trial_id = ? ORDER BY id;";
    rc = sqlite3_prepare_v2(db_, select_eeg_sql, -1, &select_eeg_stmt_, nullptr);
    if (rc != SQLITE_OK)
    {
        NN_LOG_ERROR(std::string("SqliteBatchSource::open_db: prepare select_eeg_sql failed: ") +
                     sqlite3_errmsg(db_));
        if (pop_trial_stmt_) sqlite3_finalize(pop_trial_stmt_);
        pop_trial_stmt_ = nullptr;
        sqlite3_close(db_);
        db_ = nullptr;
        select_eeg_stmt_ = nullptr;
        return false;
    }

    const char* select_audio_sql =
        "SELECT samples FROM audio_samples WHERE trial_id = ? ORDER BY audio_row;";
    rc = sqlite3_prepare_v2(db_, select_audio_sql, -1, &select_audio_stmt_, nullptr);
    if (rc != SQLITE_OK)
    {
        NN_LOG_ERROR(std::string("SqliteBatchSource::open_db: prepare select_audio_sql failed: ") +
                     sqlite3_errmsg(db_));
        if (pop_trial_stmt_) sqlite3_finalize(pop_trial_stmt_);
        if (select_eeg_stmt_) sqlite3_finalize(select_eeg_stmt_);
        pop_trial_stmt_ = nullptr;
        select_eeg_stmt_ = nullptr;
        select_audio_stmt_ = nullptr;
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }

    // Sanity: query how many joined trials exist and log for diagnostics.
    {
        sqlite3_stmt* chk = nullptr;
        const char* chk_sql =
            "SELECT COUNT(DISTINCT t.id) FROM trial t INNER JOIN audio_samples a ON a.trial_id = "
            "t.id INNER JOIN eeg_samples e ON e.trial_id = t.id;";
        int r2 = sqlite3_prepare_v2(db_, chk_sql, -1, &chk, nullptr);
        if (r2 == SQLITE_OK && chk)
        {
            int s = sqlite3_step(chk);
            if (s == SQLITE_ROW)
            {
                int joined = sqlite3_column_int(chk, 0);
                std::ostringstream oss;
                oss << "SqliteBatchSource::open_db: joined_trials=" << joined
                    << " db_path=" << db_path_;
                NN_LOG_INFO(oss.str());
            }
            sqlite3_finalize(chk);
        }
    }

    // Build a deterministic trial-id sequence to iterate in next().
    // When a k-fold subset filter is provided, keep only matching ids
    // while preserving DB ordering.
    {
        trial_ids_.clear();
        sqlite3_stmt* trials_stmt = nullptr;
        const char* trials_sql =
            "SELECT DISTINCT t.id FROM trial t "
            "INNER JOIN audio_samples a ON a.trial_id = t.id "
            "INNER JOIN eeg_samples e ON e.trial_id = t.id "
            "ORDER BY t.id;";

        if (sqlite3_prepare_v2(db_, trials_sql, -1, &trials_stmt, nullptr) == SQLITE_OK &&
            trials_stmt)
        {
            if (!selected_trial_ids_filter_.empty())
            {
                std::unordered_set<int> allowed(
                    selected_trial_ids_filter_.begin(), selected_trial_ids_filter_.end());
                while (sqlite3_step(trials_stmt) == SQLITE_ROW)
                {
                    const int trial_id = sqlite3_column_int(trials_stmt, 0);
                    if (allowed.contains(trial_id))
                    {
                        trial_ids_.push_back(trial_id);
                    }
                }
            }
            else
            {
                while (sqlite3_step(trials_stmt) == SQLITE_ROW)
                {
                    trial_ids_.push_back(sqlite3_column_int(trials_stmt, 0));
                }
            }
        }

        if (trials_stmt)
        {
            sqlite3_finalize(trials_stmt);
        }
        next_trial_index_ = 0;
    }

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
    // Restart deterministic trial iteration from the first selected id.
    (void) epoch;
    next_trial_index_ = 0;
    pending_window_samples_.clear();
    next_pending_sample_index_ = 0;
}

bool SqliteBatchSource::emit_pending_window_batch(Batch& out)
{
    if (next_pending_sample_index_ >= pending_window_samples_.size())
    {
        pending_window_samples_.clear();
        next_pending_sample_index_ = 0;
        return false;
    }

    const size_t sample_cols = pending_window_samples_[next_pending_sample_index_].size();
    const size_t remaining = pending_window_samples_.size() - next_pending_sample_index_;
    const size_t rows = std::min(batch_size_, remaining);

    out.inputs = nn::Tensor(rows, sample_cols);
    out.targets = nn::Tensor(rows, sample_cols);

    for (std::size_t row = 0; row < rows; ++row)
    {
        const auto& sample = pending_window_samples_[next_pending_sample_index_ + row];
        if (sample.empty())
        {
            continue;
        }

        for (std::size_t col = 0; col < sample_cols; ++col)
        {
            out.inputs.at(static_cast<nn::Index>(row), static_cast<nn::Index>(col)) = sample[col];
            out.targets.at(static_cast<nn::Index>(row), static_cast<nn::Index>(col)) = sample[col];
        }
    }

    next_pending_sample_index_ += rows;
    if (next_pending_sample_index_ >= pending_window_samples_.size())
    {
        pending_window_samples_.clear();
        next_pending_sample_index_ = 0;
    }

    return true;
}

bool SqliteBatchSource::next(Batch& out)
{
    // Normal operation: no verbose tracing here (kept concise for production).
    // Try to consume an existing trial from the DB first.
    if (db_)
    {
        try
        {
            if (emit_pending_window_batch(out))
            {
                return true;
            }

            while (next_trial_index_ < trial_ids_.size())
            {
                const int trial_id = trial_ids_[next_trial_index_++];
                std::vector<float> eeg_accum;
                std::vector<float> audio_accum;

                // Read eeg_samples rows for this trial
                sqlite3_reset(select_eeg_stmt_);
                sqlite3_bind_int(select_eeg_stmt_, 1, trial_id);
                while (sqlite3_step(select_eeg_stmt_) == SQLITE_ROW)
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
                while (sqlite3_step(select_audio_stmt_) == SQLITE_ROW)
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

                // Build windowed/fused samples according to requested dataset_type_.
                // For Protocol+Concatenated we keep the flattened trial semantics
                // (replicate full trial). For windowing datasets, extract aligned
                // windows from EEG and/or audio and produce samples which are
                // then packed into a batch of `batch_size_` rows (cycling if
                // necessary).

                // Quick fallbacks: if no data, use underlying source.
                if (eeg_accum.empty() && audio_accum.empty())
                {
                    continue;
                }

                const int eeg_channels = 6; // schema: F3,F4,C3,C4,P3,P4

                if (dataset_type_ == nn::dataLoaders::SqliteDatasetType::Protocol &&
                    input_mode_ == Protocol101117InputMode::Concatenated)
                {
                    // Protocol+Concatenated mode: match Dataset101117's stacked format.
                    // Each trial produces (7, 176400) when merged&resampled (audio + 6 EEG).
                    // Multiple trials are stacked vertically to fill batch_size.
                    // Example: batch_size=2 trials → (14, 176400) output.

                    try
                    {
                        if (eeg_accum.empty() || audio_accum.empty())
                        {
                            continue; // Skip trials with incomplete data
                        }

                        // EEG data is accumulated as [ch0_t0, ch1_t0, ..., ch5_t0, ch0_t1, ...]
                        // i.e., (samples_per_channel * channels) floats interleaved per timestep.
                        // Need to reshape to (channels, samples_per_channel) for
                        // mergeAudioAndEEGSignals.

                        if (eeg_accum.size() % eeg_channels != 0)
                        {
                            NN_LOG_WARN(
                                "SqliteBatchSource Protocol: EEG size not divisible by "
                                "channel count; skipping trial");
                            continue;
                        }

                        const size_t eeg_per_channel = eeg_accum.size() / eeg_channels;
                        nn::Tensor eeg_matrix(static_cast<size_t>(eeg_channels),
                            static_cast<size_t>(eeg_per_channel));

                        // Transpose from channel-interleaved to channel-separated layout.
                        // Input: eeg_accum[ch0_t0, ch1_t0, ..., ch5_t0, ch0_t1, ...]
                        // Output matrix[ch, t] = eeg_accum[t * channels + ch]
                        for (size_t t = 0; t < eeg_per_channel; ++t)
                        {
                            for (size_t ch = 0; ch < eeg_channels; ++ch)
                            {
                                eeg_matrix.at(ch, t) = eeg_accum[t * eeg_channels + ch];
                            }
                        }

                        // Build audio vector: (audio_samples, 1)
                        nn::Tensor audio_vector(audio_accum.size(), 1);
                        for (size_t i = 0; i < audio_accum.size(); ++i)
                        {
                            audio_vector.at(i, 0) = audio_accum[i];
                        }

                        // Stack and resample to produce (7, 176400)
                        nn::Tensor stacked_resampled =
                            mergeAudioAndEEGSignals(eeg_matrix, audio_vector);

                        // Accumulate row-by-row into pending samples for batching.
                        // Each row of the stacked result becomes a pending sample.
                        if (pending_window_samples_.empty())
                        {
                            pending_window_samples_.resize(stacked_resampled.rows());
                            for (size_t row = 0; row < stacked_resampled.rows(); ++row)
                            {
                                pending_window_samples_[row].resize(stacked_resampled.cols());
                                for (size_t col = 0; col < stacked_resampled.cols(); ++col)
                                {
                                    pending_window_samples_[row][col] =
                                        stacked_resampled.at(row, col);
                                }
                            }
                        }
                        else
                        {
                            // Append rows from this trial
                            for (size_t row = 0; row < stacked_resampled.rows(); ++row)
                            {
                                std::vector<float> row_data(stacked_resampled.cols());
                                for (size_t col = 0; col < stacked_resampled.cols(); ++col)
                                {
                                    row_data[col] = stacked_resampled.at(row, col);
                                }
                                pending_window_samples_.push_back(std::move(row_data));
                            }
                        }

                        // Check if we have enough rows to emit a batch
                        if (pending_window_samples_.size() >= batch_size_)
                        {
                            return emit_pending_window_batch(out);
                        }
                    }
                    catch (const std::exception& e)
                    {
                        NN_LOG_WARN(std::string("SqliteBatchSource Protocol: stacking/resampling "
                                                "failed: ") +
                                    e.what() + "; skipping trial");
                        continue;
                    }
                }
                else
                {
                    // Windowing behavior: compute per-channel lengths and number
                    // of windows for eeg and audio, then produce fused or single
                    // modality windows aligned by window index.
                    const int audio_len = static_cast<int>(audio_accum.size());
                    int per_channel_len = 0;
                    if (!eeg_accum.empty() && (eeg_accum.size() % eeg_channels == 0))
                    {
                        per_channel_len = static_cast<int>(eeg_accum.size() / eeg_channels);
                    }

                    const int num_windows_eeg =
                        per_channel_len > 0 ? eeg_window_.num_windows(per_channel_len) : 0;
                    const int num_windows_audio =
                        audio_len > 0 ? audio_window_.num_windows(audio_len) : 0;

                    int windows = 0;
                    switch (dataset_type_)
                    {
                        case nn::dataLoaders::SqliteDatasetType::EegWindow:
                            windows = num_windows_eeg;
                            if (windows <= 0 && per_channel_len > 0)
                                windows = 1; // allow padded partial window
                            break;
                        case nn::dataLoaders::SqliteDatasetType::AudioWindow:
                            windows = num_windows_audio;
                            if (windows <= 0 && audio_len > 0)
                                windows = 1; // allow padded partial window
                            break;
                        case nn::dataLoaders::SqliteDatasetType::FusedWindow:
                            windows = std::min(num_windows_eeg, num_windows_audio);
                            if (windows <= 0 && (num_windows_eeg > 0 || num_windows_audio > 0))
                            {
                                // If one modality has windows and the other is short, allow one
                                // padded fused window
                                windows = 1;
                            }
                            else if (windows <= 0 && (per_channel_len > 0 || audio_len > 0))
                            {
                                // Both modalities are short but present: produce one padded fused
                                // window
                                windows = 1;
                            }
                            break;
                        default:
                            windows = 0;
                            break;
                    }

                    if (windows <= 0)
                    {
                        // Nothing to produce in windowed mode for this trial.
                        continue;
                    }

                    // Build all sample vectors for this trial.
                    std::vector<std::vector<float>> samples;
                    samples.reserve(static_cast<size_t>(windows));

                    const int eeg_hop = eeg_window_.hop_size();
                    const int audio_hop = audio_window_.hop_size();
                    const size_t eeg_sample_cols =
                        (dataset_type_ == nn::dataLoaders::SqliteDatasetType::EegWindow ||
                            dataset_type_ == nn::dataLoaders::SqliteDatasetType::FusedWindow)
                            ? static_cast<size_t>(eeg_channels) *
                                  static_cast<size_t>(eeg_window_.window_size)
                            : 0;
                    const size_t audio_sample_cols =
                        (dataset_type_ == nn::dataLoaders::SqliteDatasetType::AudioWindow ||
                            dataset_type_ == nn::dataLoaders::SqliteDatasetType::FusedWindow)
                            ? static_cast<size_t>(audio_window_.window_size)
                            : 0;
                    const size_t expected_sample_cols = eeg_sample_cols + audio_sample_cols;

                    for (int w = 0; w < windows; ++w)
                    {
                        std::vector<float> samp;
                        samp.reserve(expected_sample_cols);
                        if (dataset_type_ == nn::dataLoaders::SqliteDatasetType::EegWindow ||
                            dataset_type_ == nn::dataLoaders::SqliteDatasetType::FusedWindow)
                        {
                            const int eeg_start = w * eeg_hop;
                            const int win = eeg_window_.window_size;
                            for (int ch = 0; ch < eeg_channels; ++ch)
                            {
                                const int ch_off = ch * per_channel_len;
                                if (eeg_start + win <= per_channel_len)
                                {
                                    samp.insert(samp.end(),
                                        eeg_accum.begin() + static_cast<size_t>(ch_off + eeg_start),
                                        eeg_accum.begin() +
                                            static_cast<size_t>(ch_off + eeg_start + win));
                                }
                                else
                                {
                                    // out-of-bounds: repeat last available sample to pad
                                    const size_t eeg_start_index =
                                        eeg_start >= 0 ? static_cast<size_t>(eeg_start) : 0;
                                    const size_t per_channel_len_u =
                                        static_cast<size_t>(per_channel_len);
                                    const size_t available =
                                        eeg_start >= 0 && eeg_start_index < per_channel_len_u
                                            ? (per_channel_len_u - eeg_start_index)
                                            : 0;
                                    if (available > 0)
                                    {
                                        samp.insert(samp.end(),
                                            eeg_accum.begin() +
                                                static_cast<size_t>(ch_off + eeg_start),
                                            eeg_accum.begin() +
                                                static_cast<size_t>(
                                                    ch_off + eeg_start + available));
                                        // repeat last value to reach window size
                                        float last = eeg_accum[static_cast<size_t>(
                                            ch_off + eeg_start + available - 1)];
                                        samp.insert(
                                            samp.end(), static_cast<size_t>(win - available), last);
                                    }
                                    else
                                    {
                                        samp.insert(samp.end(), win, 0.0f);
                                    }
                                }
                            }
                        }

                        if (dataset_type_ == nn::dataLoaders::SqliteDatasetType::AudioWindow ||
                            dataset_type_ == nn::dataLoaders::SqliteDatasetType::FusedWindow)
                        {
                            const int audio_start = w * audio_hop;
                            const int win = audio_window_.window_size;
                            if (audio_start + win <= audio_len)
                            {
                                samp.insert(samp.end(),
                                    audio_accum.begin() + static_cast<size_t>(audio_start),
                                    audio_accum.begin() + static_cast<size_t>(audio_start + win));
                            }
                            else
                            {
                                // partial tail: copy available and repeat last sample
                                const int available = std::max(0, audio_len - audio_start);
                                if (available > 0)
                                {
                                    samp.insert(samp.end(),
                                        audio_accum.begin() + static_cast<size_t>(audio_start),
                                        audio_accum.begin() +
                                            static_cast<size_t>(audio_start + available));
                                    float last = audio_accum[static_cast<size_t>(
                                        audio_start + available - 1)];
                                    samp.insert(
                                        samp.end(), static_cast<size_t>(win - available), last);
                                }
                                else
                                {
                                    samp.insert(samp.end(), win, 0.0f);
                                }
                            }
                        }

                        samples.push_back(std::move(samp));
                    }

                    pending_window_samples_ = std::move(samples);
                    next_pending_sample_index_ = 0;
                    if (emit_pending_window_batch(out))
                    {
                        return true;
                    }

                    continue;
                }

                // Production: do not emit verbose debug logs here.

                return true;
            }

            return false;
        }
        catch (const std::exception& e)
        {
            NN_LOG_ERROR(std::string("EXCEPTION[SqliteBatchSource::next]: ") + e.what());
        }
        catch (...)
        {
            NN_LOG_ERROR("EXCEPTION[SqliteBatchSource::next]: unknown");
        }
    }

    // DB-only source: no underlying fallback. If we reach here, signal
    // that no data could be produced from the DB.
    NN_LOG_INFO("SqliteBatchSource: no data available from DB");
    return false;
}
