#include "nn/dataLoaders/SqliteBatchSource.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/utility/batching.hpp"
#include "nn/windowing/WindowSpec.hpp"

using std::string;

namespace
{
// Minimal internal diagnostics removed in non-debug builds to avoid noisy logs.
} // namespace

SqliteBatchSource::SqliteBatchSource(const string& db_root,
    std::size_t batch_size,
    nn::dataLoaders::SqliteDatasetType dataset_type,
    const nn::windowing::WindowSpec& eeg_window,
    const nn::windowing::WindowSpec& audio_window,
    Protocol101117InputMode input_mode)
    : batch_size_(batch_size),
      dataset_type_(dataset_type),
      eeg_window_(eeg_window),
      audio_window_(audio_window),
      input_mode_(input_mode)
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
        std::cerr << "SqliteBatchSource::open_db: prepare pop_trial_sql failed: "
                  << sqlite3_errmsg(db_) << std::endl;
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
        std::cerr << "SqliteBatchSource::open_db: prepare select_eeg_sql failed: "
                  << sqlite3_errmsg(db_) << std::endl;
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
        std::cerr << "SqliteBatchSource::open_db: prepare select_audio_sql failed: "
                  << sqlite3_errmsg(db_) << std::endl;
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
                std::string msg = oss.str();
                std::cerr << msg << std::endl;
            }
            sqlite3_finalize(chk);
        }
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
    // No underlying source to reset for DB-only SqliteBatchSource.
    (void) epoch;
}

bool SqliteBatchSource::next(Batch& out)
{
    // Normal operation: no verbose tracing here (kept concise for production).
    // Try to consume an existing trial from the DB first.
    if (db_ && pop_trial_stmt_)
    {
        try
        {
            // Prepare and step a transient pop_trial statement each call
            const char* pop_trial_sql =
                "SELECT DISTINCT t.id FROM trial t "
                "INNER JOIN audio_samples a ON a.trial_id = t.id "
                "INNER JOIN eeg_samples e ON e.trial_id = t.id "
                "ORDER BY t.id LIMIT 1;";
            sqlite3_stmt* local_pop = nullptr;
            int prc = sqlite3_prepare_v2(db_, pop_trial_sql, -1, &local_pop, nullptr);
            if (prc != SQLITE_OK || !local_pop)
            {
                std::ostringstream oss;
                oss << "SqliteBatchSource::next: prepare(pop_trial) failed: "
                    << sqlite3_errmsg(db_);
                std::string s = oss.str();
                std::cerr << s << std::endl;
                if (local_pop) sqlite3_finalize(local_pop);
                if (pop_trial_stmt_) sqlite3_reset(pop_trial_stmt_);
                return false;
            }

            int rc = sqlite3_step(local_pop);
            if (rc == SQLITE_ROW)
            {
                const int trial_id = sqlite3_column_int(local_pop, 0);

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

                // Finalize the transient pop stmt
                sqlite3_finalize(local_pop);

                // Build windowed/fused samples according to requested dataset_type_.
                // For Protocol+Concatenated we keep the flattened trial semantics
                // (replicate full trial). For windowing datasets, extract aligned
                // windows from EEG and/or audio and produce samples which are
                // then packed into a batch of `batch_size_` rows (cycling if
                // necessary).

                // Quick fallbacks: if no data, use underlying source.
                if (eeg_accum.empty() && audio_accum.empty())
                {
                    if (pop_trial_stmt_) sqlite3_reset(pop_trial_stmt_);
                    return false;
                }

                const int eeg_channels = 6; // schema: F3,F4,C3,C4,P3,P4

                if (dataset_type_ == nn::dataLoaders::SqliteDatasetType::Protocol &&
                    input_mode_ == Protocol101117InputMode::Concatenated)
                {
                    // Keep previous behavior: flattened trial replicated.
                    const int in_cols = static_cast<int>(eeg_accum.size());
                    const int tgt_cols = static_cast<int>(audio_accum.size());

                    out.inputs =
                        nn::Tensor(static_cast<size_t>(batch_size_), static_cast<size_t>(in_cols));
                    if (!eeg_accum.empty())
                    {
                        for (std::size_t r = 0; r < batch_size_; ++r)
                        {
                            std::memcpy(
                                out.inputs.mutable_data_ptr() + r * static_cast<size_t>(in_cols),
                                eeg_accum.data(),
                                static_cast<size_t>(in_cols) * sizeof(float));
                        }
                    }

                    out.targets =
                        nn::Tensor(static_cast<size_t>(batch_size_), static_cast<size_t>(tgt_cols));
                    if (!audio_accum.empty())
                    {
                        for (std::size_t r = 0; r < batch_size_; ++r)
                        {
                            std::memcpy(
                                out.targets.mutable_data_ptr() + r * static_cast<size_t>(tgt_cols),
                                audio_accum.data(),
                                static_cast<size_t>(tgt_cols) * sizeof(float));
                        }
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
                        if (pop_trial_stmt_) sqlite3_reset(pop_trial_stmt_);
                        return false;
                    }

                    // Build all sample vectors for this trial.
                    std::vector<std::vector<float>> samples;
                    samples.reserve(static_cast<size_t>(windows));

                    const int eeg_hop = eeg_window_.hop_size();
                    const int audio_hop = audio_window_.hop_size();

                    for (int w = 0; w < windows; ++w)
                    {
                        std::vector<float> samp;
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
                                    const size_t available =
                                        per_channel_len > static_cast<size_t>(eeg_start)
                                            ? static_cast<size_t>(per_channel_len - eeg_start)
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

                    // Determine sample vector size and create output tensors.
                    const size_t sample_cols = samples.empty() ? 0 : samples[0].size();
                    out.inputs = nn::Tensor(batch_size_, sample_cols);
                    out.targets = nn::Tensor(batch_size_, sample_cols);

                    // Fill batch rows by cycling through generated samples.
                    for (std::size_t r = 0; r < batch_size_; ++r)
                    {
                        const auto& svec = samples[r % samples.size()];
                        if (!svec.empty())
                        {
                            std::memcpy(out.inputs.mutable_data_ptr() + r * sample_cols,
                                svec.data(),
                                sample_cols * sizeof(float));
                            std::memcpy(out.targets.mutable_data_ptr() + r * sample_cols,
                                svec.data(),
                                sample_cols * sizeof(float));
                        }
                    }
                }

                // Production: do not emit verbose debug logs here.

                return true;
            }
            // No trial available from transient pop stmt - log return code and fallback
            {
                std::ostringstream oss;
                oss << "SqliteBatchSource::next: local_pop rc=" << rc << " (" << sqlite3_errstr(rc)
                    << ") errmsg=" << (db_ ? sqlite3_errmsg(db_) : "(null db)");
                std::string s = oss.str();
                std::cerr << s << std::endl;
            }
            if (local_pop) sqlite3_finalize(local_pop);
            sqlite3_reset(pop_trial_stmt_);
        }
        catch (const std::exception& e)
        {
            std::cerr << "EXCEPTION[SqliteBatchSource::next]: " << e.what() << std::endl;
            // Reset statement to a known state and fall back to underlying source.
            if (pop_trial_stmt_) sqlite3_reset(pop_trial_stmt_);
        }
        catch (...)
        {
            std::cerr << "EXCEPTION[SqliteBatchSource::next]: unknown" << std::endl;
            if (pop_trial_stmt_) sqlite3_reset(pop_trial_stmt_);
        }
    }

    // DB-only source: no underlying fallback. If we reach here, signal
    // that no data could be produced from the DB.
    std::string s = "SqliteBatchSource: no data available from DB";
    std::cerr << s << std::endl;
    return false;
}
