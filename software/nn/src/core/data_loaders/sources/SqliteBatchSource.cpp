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

/// The six EEG channel columns the schema stores (F3,F4,C3,C4,P3,P4).
/// Column 6, `blink`, exists but is deliberately not read.
constexpr int kEegChannels = 6;

/// Read one trial's EEG and audio sample blobs into flat float vectors.
///
/// Both tables store float arrays as BLOBs, so the loops are the same shape:
/// step the prepared statement, append every non-empty blob's floats. EEG has
/// six channel columns per row (F3,F4,C3,C4,P3,P4 -- `blink` is column 6 and
/// is deliberately not read); audio has one.
void read_trial_signals(sqlite3_stmt* eeg_stmt,
    sqlite3_stmt* audio_stmt,
    int trial_id,
    std::vector<float>& eeg_accum,
    std::vector<float>& audio_accum)
{
    sqlite3_reset(eeg_stmt);
    sqlite3_bind_int(eeg_stmt, 1, trial_id);
    while (sqlite3_step(eeg_stmt) == SQLITE_ROW)
    {
        for (int c = 0; c < kEegChannels; ++c)
        {
            const void* blob = sqlite3_column_blob(eeg_stmt, c);
            int bytes = sqlite3_column_bytes(eeg_stmt, c);
            if (blob && bytes > 0)
            {
                const size_t count = static_cast<size_t>(bytes) / sizeof(float);
                const float* fptr = static_cast<const float*>(blob);
                eeg_accum.insert(eeg_accum.end(), fptr, fptr + count);
            }
        }
    }
    sqlite3_reset(eeg_stmt);

    sqlite3_reset(audio_stmt);
    sqlite3_bind_int(audio_stmt, 1, trial_id);
    while (sqlite3_step(audio_stmt) == SQLITE_ROW)
    {
        const void* blob = sqlite3_column_blob(audio_stmt, 0);
        int bytes = sqlite3_column_bytes(audio_stmt, 0);
        if (blob && bytes > 0)
        {
            const size_t count = static_cast<size_t>(bytes) / sizeof(float);
            const float* fptr = static_cast<const float*>(blob);
            audio_accum.insert(audio_accum.end(), fptr, fptr + count);
        }
    }
    sqlite3_reset(audio_stmt);
}

/// How many windows this trial yields, for the requested dataset type.
///
/// A count of zero with a non-empty signal still yields ONE window: a signal
/// shorter than the window is padded rather than dropped, so short trials are
/// not silently lost. (`append_window` does the padding.)
int window_count(nn::dataLoaders::SqliteDatasetType dataset_type,
    int num_windows_eeg,
    int num_windows_audio,
    int per_channel_len,
    int audio_len)
{
    switch (dataset_type)
    {
        case nn::dataLoaders::SqliteDatasetType::EegWindow:
            if (num_windows_eeg > 0) return num_windows_eeg;
            return per_channel_len > 0 ? 1 : 0;
        case nn::dataLoaders::SqliteDatasetType::AudioWindow:
            if (num_windows_audio > 0) return num_windows_audio;
            return audio_len > 0 ? 1 : 0;
        case nn::dataLoaders::SqliteDatasetType::FusedWindow:
        {
            const int both = std::min(num_windows_eeg, num_windows_audio);
            if (both > 0) return both;
            // One modality has windows and the other is short, or both are
            // short but present: either way, one padded fused window.
            if (num_windows_eeg > 0 || num_windows_audio > 0) return 1;
            return (per_channel_len > 0 || audio_len > 0) ? 1 : 0;
        }
        default:
            return 0;
    }
}

/// Append `win` samples of `src` starting at `base + start` to `out`.
///
/// When the window runs past the end of the signal the tail is padded by
/// repeating the last available sample -- and with zeros when the window
/// starts past the end entirely, since there is no last sample to repeat.
/// `base` is the channel offset into a channel-major buffer (0 for audio).
void append_window(const std::vector<float>& src,
    int base,
    int length,
    int start,
    int win,
    std::vector<float>& out)
{
    if (start + win <= length)
    {
        out.insert(out.end(),
            src.begin() + static_cast<size_t>(base + start),
            src.begin() + static_cast<size_t>(base + start + win));
        return;
    }

    const int available = std::max(0, std::min(length, length - start));
    if (available <= 0)
    {
        out.insert(out.end(), win, 0.0f);
        return;
    }

    out.insert(out.end(),
        src.begin() + static_cast<size_t>(base + start),
        src.begin() + static_cast<size_t>(base + start + available));
    const float last = src[static_cast<size_t>(base + start + available - 1)];
    out.insert(out.end(), static_cast<size_t>(win - available), last);
}

/// One Protocol+Concatenated trial as a stacked, resampled (7, N) tensor:
/// the audio row followed by the six EEG channel rows.
///
/// Throws `std::invalid_argument` when the EEG buffer is not a whole number
/// of channels -- the caller logs and skips that trial rather than guessing a
/// channel count.
nn::Tensor stack_protocol_trial(
    const std::vector<float>& eeg_accum, const std::vector<float>& audio_accum)
{
    if (eeg_accum.size() % kEegChannels != 0)
    {
        throw std::invalid_argument(
            "SqliteBatchSource: EEG buffer size is not divisible by the channel count");
    }

    // EEG data is accumulated as [ch0_t0, ch1_t0, ..., ch5_t0, ch0_t1, ...]
    // i.e., (samples_per_channel * channels) floats interleaved per timestep.
    // Need to reshape to (channels, samples_per_channel) for
    // mergeAudioAndEEGSignals.

    const size_t eeg_per_channel = eeg_accum.size() / kEegChannels;
    nn::Tensor eeg_matrix(static_cast<size_t>(kEegChannels), static_cast<size_t>(eeg_per_channel));

    // Transpose from channel-interleaved to channel-separated layout.
    // Input: eeg_accum[ch0_t0, ch1_t0, ..., ch5_t0, ch0_t1, ...]
    // Output matrix[ch, t] = eeg_accum[t * channels + ch]
    for (size_t t = 0; t < eeg_per_channel; ++t)
    {
        for (size_t ch = 0; ch < kEegChannels; ++ch)
        {
            eeg_matrix.at(ch, t) = eeg_accum[t * kEegChannels + ch];
        }
    }

    // Build audio vector: (audio_samples, 1)
    nn::Tensor audio_vector(audio_accum.size(), 1);
    for (size_t i = 0; i < audio_accum.size(); ++i)
    {
        audio_vector.at(i, 0) = audio_accum[i];
    }

    // Stack and resample to produce (7, 176400)
    nn::Tensor stacked_resampled = mergeAudioAndEEGSignals(eeg_matrix, audio_vector);
    return stacked_resampled;
}

/// Every window this trial yields, as one flat sample vector per window.
///
/// A fused sample is the six EEG channel windows followed by the audio
/// window, in that order -- the two `append_window` groups below must stay in
/// this sequence, because the emitted batch's column layout IS this order and
/// a consumer reading `[0, window_size)` as channel F3 depends on it.
///
/// Returns empty when the trial yields nothing (no signal, or a dataset type
/// with no windows); the caller skips such a trial.
std::vector<std::vector<float>> build_trial_windows(const std::vector<float>& eeg_accum,
    const std::vector<float>& audio_accum,
    nn::dataLoaders::SqliteDatasetType dataset_type,
    const nn::windowing::WindowSpec& eeg_window,
    const nn::windowing::WindowSpec& audio_window)
{
    // Windowing behavior: compute per-channel lengths and number
    // of windows for eeg and audio, then produce fused or single
    // modality windows aligned by window index.
    const int audio_len = static_cast<int>(audio_accum.size());
    int per_channel_len = 0;
    if (!eeg_accum.empty() && (eeg_accum.size() % kEegChannels == 0))
    {
        per_channel_len = static_cast<int>(eeg_accum.size() / kEegChannels);
    }

    const int num_windows_eeg = per_channel_len > 0 ? eeg_window.num_windows(per_channel_len) : 0;
    const int num_windows_audio = audio_len > 0 ? audio_window.num_windows(audio_len) : 0;

    const int windows =
        window_count(dataset_type, num_windows_eeg, num_windows_audio, per_channel_len, audio_len);

    if (windows <= 0) return {};

    // Build all sample vectors for this trial.
    std::vector<std::vector<float>> samples;
    samples.reserve(static_cast<size_t>(windows));

    const int eeg_hop = eeg_window.hop_size();
    const int audio_hop = audio_window.hop_size();
    const size_t eeg_sample_cols =
        (dataset_type == nn::dataLoaders::SqliteDatasetType::EegWindow ||
            dataset_type == nn::dataLoaders::SqliteDatasetType::FusedWindow)
            ? static_cast<size_t>(kEegChannels) * static_cast<size_t>(eeg_window.window_size)
            : 0;
    const size_t audio_sample_cols =
        (dataset_type == nn::dataLoaders::SqliteDatasetType::AudioWindow ||
            dataset_type == nn::dataLoaders::SqliteDatasetType::FusedWindow)
            ? static_cast<size_t>(audio_window.window_size)
            : 0;
    const size_t expected_sample_cols = eeg_sample_cols + audio_sample_cols;

    for (int w = 0; w < windows; ++w)
    {
        std::vector<float> samp;
        samp.reserve(expected_sample_cols);
        if (dataset_type == nn::dataLoaders::SqliteDatasetType::EegWindow ||
            dataset_type == nn::dataLoaders::SqliteDatasetType::FusedWindow)
        {
            const int eeg_start = w * eeg_hop;
            const int win = eeg_window.window_size;
            for (int ch = 0; ch < kEegChannels; ++ch)
            {
                append_window(
                    eeg_accum, ch * per_channel_len, per_channel_len, eeg_start, win, samp);
            }
        }

        if (dataset_type == nn::dataLoaders::SqliteDatasetType::AudioWindow ||
            dataset_type == nn::dataLoaders::SqliteDatasetType::FusedWindow)
        {
            const int audio_start = w * audio_hop;
            const int win = audio_window.window_size;
            append_window(audio_accum, 0, audio_len, audio_start, win, samp);
        }

        samples.push_back(std::move(samp));
    }
    return samples;
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

                read_trial_signals(
                    select_eeg_stmt_, select_audio_stmt_, trial_id, eeg_accum, audio_accum);

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

                        nn::Tensor stacked_resampled = stack_protocol_trial(eeg_accum, audio_accum);

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
                    std::vector<std::vector<float>> samples = build_trial_windows(
                        eeg_accum, audio_accum, dataset_type_, eeg_window_, audio_window_);
                    if (samples.empty())
                    {
                        // Nothing to produce in windowed mode for this trial.
                        continue;
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
