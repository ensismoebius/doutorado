#include "ThesisDataset.hpp"

#include <sqlite3.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "EegPairing.hpp"
#include "data_loaders/10.1117/schema/SqliteDataset.hpp"
#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"
#include "logging/Logger.hpp"
#include "progress/ProgressManager.hpp"
#include "utility/path_expand.hpp"

namespace thesis
{

namespace
{
// ── Decoded-dataset binary cache ─────────────────────────────────────────────
// Loading a run means opening every subject's .mat files and decoding thousands
// of audio/EEG rows — the dominant cost of an autoencoder profile, repeated
// identically for every profile/repeat since the raw set never changes. This
// cache writes the fully-decoded sample set to one flat binary file the first
// time and reloads it directly afterwards (float blobs, no re-decode). The OS
// page cache keeps it in RAM across the runner's parallel workers, so on a
// machine with spare memory later loads are memory-speed.
//
// The cache stores the FULL set (before max_samples truncation), so one file
// serves every max_samples value; truncation is always applied after loading.
// Set THESIS_NO_DATASET_CACHE=1 to bypass entirely. A missing cache, another cache
// format version or another source signature is a cache miss: the source is decoded and
// the cache rewritten. A cache whose header matches but whose body cannot be read is
// corrupt and throws (it used to be decoded around in silence), as does a failed write.
// Version 2 (2026-11-12): text phrases come from the dataset's stimulus table.
// Version 3 (2026-11-12): EEG paired by trial identity (EegPairing.hpp). A version-2 cache
// holds every voice trial with the PREVIOUS trial's EEG; it must never load again.
constexpr char kCacheMagic[8] = {'E', '0', '5', 'D', 'S', 'C', '\0', '\3'}; // last byte = version

bool cache_disabled()
{
    const char* v = std::getenv("THESIS_NO_DATASET_CACHE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// One file per pairing: the two conditions hold different EEG, and runs of both read their
// caches concurrently on the cluster, so they must not take turns overwriting one file.
std::string cache_path_for(const std::string& root, EegCondition condition, std::uint32_t seed)
{
    return condition == EegCondition::Pronounced
               ? root + ".pronounced.e05dscache"
               : root + ".imagined-" + std::to_string(seed) + ".e05dscache";
}

// ── Trial metadata (SQLite) ──────────────────────────────────────────────────
struct SqliteDb
{
    sqlite3* db = nullptr;
    explicit SqliteDb(const std::string& path)
        : db(nn::dataLoaders::openSqliteDatasetReadOnly(path))
    {
    }
    ~SqliteDb()
    {
        sqlite3_close(db);
    }
    SqliteDb(const SqliteDb&) = delete;
    SqliteDb& operator=(const SqliteDb&) = delete;
};

struct Statement
{
    sqlite3_stmt* stmt = nullptr;
    Statement(sqlite3* db, const char* sql)
    {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
            throw std::runtime_error(std::string("ThesisDataset: cannot prepare \"") + sql +
                                     "\": " + sqlite3_errmsg(db));
    }
    ~Statement()
    {
        sqlite3_finalize(stmt);
    }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
};

auto read_modality_ids(sqlite3* db) -> ModalityIds
{
    Statement q(db, "SELECT id, name FROM modalities");
    std::map<std::string, int> ids;
    while (nn::dataLoaders::sqliteStepRow(db, q.stmt, "ThesisDataset: reading modalities"))
        ids[reinterpret_cast<const char*>(sqlite3_column_text(q.stmt, 1))] =
            sqlite3_column_int(q.stmt, 0);
    if (!ids.count("Pronounced") || !ids.count("Imagined"))
        throw std::runtime_error(
            "ThesisDataset: the modalities table lacks \"Pronounced\" or \"Imagined\"");
    return {ids.at("Pronounced"), ids.at("Imagined")};
}

auto read_eeg_trials(sqlite3* db, int subject_id) -> std::vector<EegTrialMeta>
{
    Statement q(db,
        "SELECT t.original_row, t.modality_id, t.stimulus_id FROM trial t JOIN eeg_samples e "
        "ON e.trial_id = t.id WHERE t.subject_id = ? ORDER BY t.original_row");
    if (sqlite3_bind_int(q.stmt, 1, subject_id) != SQLITE_OK)
        throw std::runtime_error("ThesisDataset: cannot bind the subject id");
    std::vector<EegTrialMeta> trials;
    while (nn::dataLoaders::sqliteStepRow(db, q.stmt, "ThesisDataset: reading EEG trials"))
    {
        for (int c = 0; c < 3; ++c)
            if (sqlite3_column_type(q.stmt, c) == SQLITE_NULL)
                throw std::runtime_error("ThesisDataset: subject " + std::to_string(subject_id) +
                                         " has an EEG trial with a NULL original_row, "
                                         "modality or stimulus");
        trials.push_back({sqlite3_column_int64(q.stmt, 0),
            sqlite3_column_int(q.stmt, 1),
            sqlite3_column_int(q.stmt, 2)});
    }
    return trials;
}

// Structural fingerprint of the source, cheap to compute (no row decode): the
// subject list plus each file's size+mtime. Any add/remove/edit of a source
// file changes it, invalidating the cache without a content re-read.
std::uint64_t source_signature(const std::vector<SubjectFiles>& subjects)
{
    std::uint64_t h = 1469598103934665603ULL; // FNV-1a offset
    auto mix = [&h](std::uint64_t x)
    {
        h ^= x;
        h *= 1099511628211ULL;
    };
    auto mix_str = [&](const std::string& s)
    {
        for (char c : s) mix(static_cast<unsigned char>(c));
        mix(0xff);
    };
    auto mix_file = [&](const std::string& p)
    {
        std::error_code ec;
        std::filesystem::path fp(p);
        auto sz = std::filesystem::file_size(fp, ec);
        mix(ec ? 0 : sz);
        auto mt = std::filesystem::last_write_time(fp, ec);
        mix(ec ? 0 : static_cast<std::uint64_t>(mt.time_since_epoch().count()));
    };
    for (const auto& sf : subjects)
    {
        mix(static_cast<std::uint64_t>(sf.subject_id));
        mix_str(sf.eeg_path);
        mix_str(sf.audio_path);
        if (!sf.eeg_path.empty()) mix_file(sf.eeg_path);
        if (!sf.audio_path.empty()) mix_file(sf.audio_path);
    }
    return h;
}

template <typename T>
void write_pod(std::ostream& os, const T& v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
bool read_pod(std::istream& is, T& v)
{
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

void write_tensor(std::ostream& os, const nn::Tensor& t)
{
    const std::uint64_t rows = t.rows();
    const std::uint64_t cols = t.cols();
    write_pod(os, rows);
    write_pod(os, cols);
    os.write(reinterpret_cast<const char*>(t.data_ptr()),
        static_cast<std::streamsize>(rows * cols * sizeof(float)));
}

bool read_tensor(std::istream& is, nn::Tensor& out)
{
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    if (!read_pod(is, rows) || !read_pod(is, cols)) return false;
    nn::Tensor t(static_cast<nn::Index>(rows), static_cast<nn::Index>(cols));
    if (!is.read(reinterpret_cast<char*>(t.mutable_data_ptr()),
            static_cast<std::streamsize>(rows * cols * sizeof(float))))
        return false;
    out = std::move(t);
    return true;
}

void write_string(std::ostream& os, const std::string& s)
{
    const std::uint32_t len = static_cast<std::uint32_t>(s.size());
    write_pod(os, len);
    os.write(s.data(), len);
}

bool read_string(std::istream& is, std::string& s)
{
    std::uint32_t len = 0;
    if (!read_pod(is, len)) return false;
    s.resize(len);
    return len == 0 || static_cast<bool>(is.read(s.data(), len));
}

// Returns true and fills `view` on a valid, signature-matching cache hit.
bool try_load_cache(const std::string& path, std::uint64_t signature, ThesisDatasetView& view)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    char magic[8];
    if (!is.read(magic, sizeof(magic)) || std::memcmp(magic, kCacheMagic, sizeof(magic)) != 0)
        return false;
    std::uint64_t sig = 0;
    if (!read_pod(is, sig) || sig != signature) return false;

    // Header matches this source: from here on a short read is a corrupt cache.
    const auto corrupt = [&path]()
    {
        return std::runtime_error("ThesisDataset: the decoded-dataset cache " + path +
                                  " matches this source but its body is unreadable; "
                                  "delete it to rebuild");
    };
    std::int32_t n_subjects = 0;
    std::int32_t n_stimuli = 0;
    std::uint64_t n_samples = 0;
    if (!read_pod(is, n_subjects) || !read_pod(is, n_stimuli) || !read_pod(is, n_samples))
        throw corrupt();

    ThesisDatasetView loaded;
    loaded.n_subjects = n_subjects;
    loaded.n_stimuli = n_stimuli;
    loaded.samples.reserve(n_samples);
    for (std::uint64_t i = 0; i < n_samples; ++i)
    {
        ThesisSample s;
        if (!read_pod(is, s.stimulus) || !read_pod(is, s.subject_id) ||
            !read_string(is, s.text_phrase) || !read_tensor(is, s.audio) || !read_tensor(is, s.eeg))
            throw corrupt();
        loaded.samples.push_back(std::move(s));
    }

    std::uint64_t n_files = 0;
    if (!read_pod(is, n_files)) throw corrupt();
    loaded.subject_files.reserve(n_files);
    for (std::uint64_t i = 0; i < n_files; ++i)
    {
        SubjectFiles sf;
        std::uint64_t eeg_rows = 0;
        std::uint64_t audio_rows = 0;
        if (!read_pod(is, sf.subject_id) || !read_string(is, sf.subject_name) ||
            !read_string(is, sf.eeg_path) || !read_string(is, sf.audio_path) ||
            !read_pod(is, eeg_rows) || !read_pod(is, audio_rows))
            throw corrupt();
        sf.eeg_rows = static_cast<std::size_t>(eeg_rows);
        sf.audio_rows = static_cast<std::size_t>(audio_rows);
        loaded.subject_files.push_back(std::move(sf));
    }

    view = std::move(loaded);
    return true;
}

// Atomic write (temp + rename) so a crash mid-write never leaves a torn cache
// that would fail the signature/read checks anyway but waste a decode proving it.
void write_cache(const std::string& path, std::uint64_t signature, const ThesisDatasetView& view)
{
    // Per-process temp name: the runner cold-starts several profiles in
    // parallel, so a shared "<path>.tmp" would be written by multiple workers
    // at once and corrupt. Each writes its own temp, then atomically renames
    // onto the final path (last writer wins; all payloads are identical).
    // The host name is part of it because a pid is unique only on one machine:
    // GridUnesp array tasks on different nodes share the same /home, so two of
    // them can hold the same pid and would write one temp file together.
    std::array<char, 256> host{};
    if (::gethostname(host.data(), host.size() - 1) != 0)
        throw std::runtime_error(
            "ThesisDataset: gethostname failed; cannot name the cache temp file");
    const std::string tmp =
        path + ".tmp." + std::string(host.data()) + "." + std::to_string(::getpid());
    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os)
            throw std::runtime_error("ThesisDataset: cannot open " + tmp + " to write the cache");
        os.write(kCacheMagic, sizeof(kCacheMagic));
        write_pod(os, signature);
        write_pod(os, static_cast<std::int32_t>(view.n_subjects));
        write_pod(os, static_cast<std::int32_t>(view.n_stimuli));
        write_pod(os, static_cast<std::uint64_t>(view.samples.size()));
        for (const auto& s : view.samples)
        {
            write_pod(os, s.stimulus);
            write_pod(os, s.subject_id);
            write_string(os, s.text_phrase);
            write_tensor(os, s.audio);
            write_tensor(os, s.eeg);
        }
        write_pod(os, static_cast<std::uint64_t>(view.subject_files.size()));
        for (const auto& sf : view.subject_files)
        {
            write_pod(os, sf.subject_id);
            write_string(os, sf.subject_name);
            write_string(os, sf.eeg_path);
            write_string(os, sf.audio_path);
            write_pod(os, static_cast<std::uint64_t>(sf.eeg_rows));
            write_pod(os, static_cast<std::uint64_t>(sf.audio_rows));
        }
        if (!os)
        {
            os.close();
            std::error_code rm_ec;
            std::filesystem::remove(tmp, rm_ec);
            throw std::runtime_error("ThesisDataset: writing the cache " + tmp + " failed");
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec)
    {
        std::error_code rm_ec;
        std::filesystem::remove(tmp, rm_ec);
        throw std::runtime_error(
            "ThesisDataset: cannot move the cache into place at " + path + ": " + ec.message());
    }
}

// Stimulus id → text phrase, from the dataset's own `stimulus` table (the one
// trial.stimulus_id references): 11 stimuli, the vowels A E I O U and the words Abajo,
// Adelante, Arriba, Atras, Derecha, Izquierda, numbered alphabetically. Until 2026-11-12
// this was a hard-coded table of 10 stimuli in another order (1-5 vowels, 6-10 words, no
// "atras"), so most phrases carried the wrong name and stimulus 11 became "unknown". The
// text split only needs the 11 groups to be distinct, which they were; the names were not.
std::string stimulus_to_phrase(const std::map<int, std::string>& names, int stim)
{
    const auto it = names.find(stim);
    if (it == names.end())
        throw std::runtime_error("ThesisDataset: stimulus " + std::to_string(stim) +
                                 " is not in the dataset's stimulus table");
    return it->second;
}

// Full decode from the SQLite source, ignoring max_samples (applied by the caller).
// Returns the complete decoded set — the unit the cache stores.
auto decode_full_dataset(const std::vector<SubjectFiles>& subjects,
    const std::map<int, std::string>& stimulus_names,
    EegCondition condition,
    std::uint32_t pairing_seed) -> ThesisDatasetView
{
    ThesisDatasetView view;
    view.subject_files = subjects;

    std::unordered_set<int> stimuli_seen;
    int subjects_with_both = 0;

    const auto n_subjects = static_cast<float>(subjects.size());
    uint32_t load_bar =
        nn::progress::ProgressManager::instance().create_bar("Loading dataset", n_subjects);
    nn::progress::ProgressManager::instance().set_description(
        load_bar, "Pairing audio+EEG trials per subject");

    int subjects_processed = 0;

    // Always load paired audio+EEG so every modality run operates on the
    // same set of trials. The modality field controls feature extraction, not
    // which samples are included. A subject missing either modality, or with an
    // empty one, is refused: it used to be dropped without a word, so a run could
    // hold fewer speakers than the dataset and nothing in its output said so.
    for (const auto& sf : subjects)
    {
        if (sf.audio_path.empty() || sf.eeg_path.empty())
            throw std::runtime_error("ThesisDataset: subject " + std::to_string(sf.subject_id) +
                                     " (" + sf.subject_name + ") has no " +
                                     (sf.audio_path.empty() ? "audio" : "EEG") + " file");
        // The pairing reads the trial table, which only the SQLite source has.
        if (std::filesystem::path(sf.eeg_path).extension() != ".sqlite")
            throw std::runtime_error("ThesisDataset: subject " + std::to_string(sf.subject_id) +
                                     " is read from " + sf.eeg_path +
                                     "; the thesis pairs voice and EEG through the SQLite trial "
                                     "table and does not read MAT sources");

        nn::dataLoaders::AudioSession audio_session(sf.audio_path, sf.subject_id);
        nn::dataLoaders::EEGSession eeg_session(sf.eeg_path, sf.subject_id);

        const size_t n_audio = audio_session.rowCount();
        if (n_audio == 0)
            throw std::runtime_error(
                "ThesisDataset: subject " + std::to_string(sf.subject_id) + " has no audio rows");

        // Voice trials first: their own original_row is the EEG of the same trial.
        std::vector<nn::Tensor> audio_tensors;
        std::vector<VoiceTrialMeta> voice;
        audio_tensors.reserve(n_audio);
        voice.reserve(n_audio);
        for (size_t row = 0; row < n_audio; ++row)
        {
            auto [audio_tensor, stimulus, original_row] = audio_session.readRow(row);
            // -1 is the loader's "NULL original_row": an audio-only trial, with no EEG.
            if (original_row < 0)
                throw std::runtime_error("ThesisDataset: subject " + std::to_string(sf.subject_id) +
                                         ", audio row " + std::to_string(row) +
                                         " has no EEG trial (NULL original_row)");
            audio_tensors.push_back(std::move(audio_tensor));
            voice.push_back({original_row, stimulus});
        }

        ModalityIds modality;
        std::vector<EegTrialMeta> eeg_trials;
        {
            const SqliteDb db(sf.eeg_path);
            modality = read_modality_ids(db.db);
            eeg_trials = read_eeg_trials(db.db, sf.subject_id);
        }
        const std::vector<long long> eeg_rows =
            pair_eeg_trials(voice, eeg_trials, modality, condition, pairing_seed, sf.subject_id);
        const int expected_modality =
            condition == EegCondition::Pronounced ? modality.pronounced : modality.imagined;

        int paired = 0;
        for (size_t i = 0; i < voice.size(); ++i)
        {
            auto [eeg_tensor, eeg_labels] = eeg_session.readRow(static_cast<size_t>(eeg_rows[i]));
            // What was read must be what was paired. Until 2026-11-12 the row read was
            // original_row - 1, the previous trial, and nothing checked it.
            if (eeg_labels[0] != expected_modality || eeg_labels[1] != voice[i].stimulus_id)
                throw std::runtime_error(
                    "ThesisDataset: subject " + std::to_string(sf.subject_id) + ", EEG row " +
                    std::to_string(eeg_rows[i]) + " is modality " + std::to_string(eeg_labels[0]) +
                    ", stimulus " + std::to_string(eeg_labels[1]) + "; expected modality " +
                    std::to_string(expected_modality) + ", stimulus " +
                    std::to_string(voice[i].stimulus_id));

            ThesisSample sample;
            sample.audio = std::move(audio_tensors[i]);
            sample.eeg = std::move(eeg_tensor);
            sample.stimulus = voice[i].stimulus_id;
            sample.subject_id = sf.subject_id;
            sample.text_phrase = stimulus_to_phrase(stimulus_names, voice[i].stimulus_id);
            stimuli_seen.insert(voice[i].stimulus_id);
            view.samples.push_back(std::move(sample));
            ++paired;
        }

        if (paired > 0) ++subjects_with_both;

        nn::progress::ProgressManager::instance().update_bar(
            load_bar, static_cast<float>(++subjects_processed));
    }

    nn::progress::ProgressManager::instance().complete_bar(load_bar);

    if (view.samples.empty())
        throw std::runtime_error(
            "ThesisDataset: no paired audio+EEG samples found. "
            "Check that each subject has both audio and EEG .mat files.");

    view.n_subjects = subjects_with_both;
    view.n_stimuli = static_cast<int>(stimuli_seen.size());
    return view;
}

// Round-robin truncation across subjects (in place). Samples are stored
// subject-contiguous, so a plain resize() to the first N would keep only the
// first 2-3 speakers — which breaks speaker-disjoint (GroupKFold) folds,
// especially nested CV. Round-robin so the truncated set spans every speaker.
void apply_max_samples(ThesisDatasetView& view, int max_samples)
{
    if (max_samples <= 0 || static_cast<int>(view.samples.size()) <= max_samples) return;

    std::map<int, std::vector<size_t>> by_subject;
    for (size_t i = 0; i < view.samples.size(); ++i)
        by_subject[view.samples[i].subject_id].push_back(i);

    std::vector<size_t> keep;
    keep.reserve(static_cast<size_t>(max_samples));
    size_t round = 0;
    bool added = true;
    while (static_cast<int>(keep.size()) < max_samples && added)
    {
        added = false;
        for (auto& [sid, idxs] : by_subject)
        {
            if (round < idxs.size())
            {
                keep.push_back(idxs[round]);
                added = true;
                if (static_cast<int>(keep.size()) >= max_samples) break;
            }
        }
        ++round;
    }
    std::sort(keep.begin(), keep.end());

    std::vector<ThesisSample> trimmed;
    trimmed.reserve(keep.size());
    for (size_t i : keep) trimmed.push_back(std::move(view.samples[i]));
    view.samples = std::move(trimmed);
}
} // namespace

auto require_rate(double rate_hz, const char* which) -> double
{
    if (!(rate_hz > 0.0) || !std::isfinite(rate_hz))
        throw std::invalid_argument(std::string("ThesisDataset: the ") + which +
                                    " sample rate of this dataset view is " +
                                    std::to_string(rate_hz) +
                                    "; load_dataset sets it from the base, a hand-built view "
                                    "must set it too");
    return rate_hz;
}

namespace
{
// The Coretto base's rates (Pressel Coretto et al., SPIE 2017, 10.1117/12.2255697): voice
// 176400 samples in 4 s, EEG 4096 samples in 4 s. Its SQLite does not store them.
constexpr double kCorettoVoiceRateHz = 44100.0;
constexpr double kCorettoEegRateHz = 1024.0;

// ── biomex2021: the paired SQLite of scripts/data/biomex_to_sqlite.py ──────────────────
//
//   recording(key, value)   format, source, rates, channels, epoch and window lengths
//   speaker(id, name)       F001 ... M026
//   stimulus(id, name)      1..9 = uno..nueve, 10 = cero
//   pair(..., voice, eeg)   one spoken digit: float32 voice window, float32 EEG
//                           (channels x samples, channel-major)
//
// The pairing was done (and checked) by the converter; here every blob is checked against
// the lengths the file declares, so a truncated or foreign file is refused, not read.
constexpr const char* kPairedFormat = "thesis-paired-v1";

auto recording_info(sqlite3* db) -> std::map<std::string, std::string>
{
    Statement q(db, "SELECT key, value FROM recording");
    std::map<std::string, std::string> info;
    while (nn::dataLoaders::sqliteStepRow(db, q.stmt, "ThesisDataset: reading recording"))
        info[reinterpret_cast<const char*>(sqlite3_column_text(q.stmt, 0))] =
            reinterpret_cast<const char*>(sqlite3_column_text(q.stmt, 1));
    return info;
}

auto info_value(
    const std::map<std::string, std::string>& info, const std::string& key, const std::string& path)
    -> const std::string&
{
    const auto it = info.find(key);
    if (it == info.end())
        throw std::runtime_error("ThesisDataset: " + path + " has no recording." + key +
                                 "; it was not written by scripts/data/biomex_to_sqlite.py");
    return it->second;
}

auto positive_count(const std::string& text, const std::string& what) -> std::size_t
{
    std::size_t used = 0;
    const long long v = std::stoll(text, &used);
    if (used != text.size() || v <= 0)
        throw std::runtime_error(
            "ThesisDataset: recording." + what + " is \"" + text + "\", not a positive integer");
    return static_cast<std::size_t>(v);
}

auto blob_floats(sqlite3_stmt* stmt, int col, std::size_t expected, const char* what, long long id)
    -> std::vector<float>
{
    const auto bytes = static_cast<std::size_t>(sqlite3_column_bytes(stmt, col));
    if (bytes != expected * sizeof(float))
        throw std::runtime_error("ThesisDataset: pair " + std::to_string(id) + " has a " + what +
                                 " of " + std::to_string(bytes) + " bytes; the base declares " +
                                 std::to_string(expected) + " float32 values");
    std::vector<float> v(expected);
    std::memcpy(v.data(), sqlite3_column_blob(stmt, col), bytes);
    return v;
}

auto load_paired(const std::string& path) -> ThesisDatasetView
{
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("ThesisDataset: " + path + " is not a file");
    const SqliteDb db(path);
    const auto info = recording_info(db.db);
    if (info_value(info, "format", path) != kPairedFormat ||
        info_value(info, "source", path) != "biomex2021")
        throw std::runtime_error("ThesisDataset: " + path + " is format \"" +
                                 info_value(info, "format", path) + "\", source \"" +
                                 info_value(info, "source", path) + "\"; dataset.source " +
                                 "biomex2021 needs " + kPairedFormat + " / biomex2021");

    ThesisDatasetView view;
    view.voice_sample_rate = std::stod(info_value(info, "voice_sample_rate", path));
    view.eeg_sample_rate = std::stod(info_value(info, "eeg_sample_rate", path));
    (void) require_rate(view.voice_sample_rate, "voice");
    (void) require_rate(view.eeg_sample_rate, "EEG");
    // The frozen pipeline splits 0..Nyquist into 2^level equal DTWPT sub-bands, so a feature
    // column is the same band in Hz on both bases only at the same rate. At its native
    // 16 kHz / 128 Hz, BIOMEX-DB spread over 9-10 of 16 sub-bands where Coretto holds 97-99%
    // of its power in sub-band 0 (measured 2026-11-13). The converter writes the Coretto
    // rates; a file at any other rate would feed the classifier columns that mean other Hz.
    if (view.voice_sample_rate != kCorettoVoiceRateHz || view.eeg_sample_rate != kCorettoEegRateHz)
        throw std::runtime_error("ThesisDataset: " + path + " is at " +
                                 info_value(info, "voice_sample_rate", path) + " Hz voice / " +
                                 info_value(info, "eeg_sample_rate", path) +
                                 " Hz EEG; both bases must reach the extractors at the Coretto "
                                 "rates (44100 / 1024 Hz) so the DTWPT sub-bands are the same Hz. "
                                 "Reconvert with scripts/data/biomex_to_sqlite.py");
    const std::size_t voice_len =
        positive_count(info_value(info, "voice_window_samples", path), "voice_window_samples");
    const std::size_t eeg_len =
        positive_count(info_value(info, "eeg_epoch_samples", path), "eeg_epoch_samples");
    const std::string& channel_list = info_value(info, "eeg_channels", path);
    const auto n_channels =
        static_cast<std::size_t>(std::count(channel_list.begin(), channel_list.end(), ',') + 1);

    std::map<int, std::string> names;
    {
        Statement q(db.db, "SELECT id, name FROM stimulus");
        while (nn::dataLoaders::sqliteStepRow(db.db, q.stmt, "ThesisDataset: reading stimuli"))
            names[sqlite3_column_int(q.stmt, 0)] =
                reinterpret_cast<const char*>(sqlite3_column_text(q.stmt, 1));
    }

    Statement q(db.db, "SELECT id, speaker_id, stimulus_id, voice, eeg FROM pair ORDER BY id");
    std::unordered_set<int> speakers;
    std::unordered_set<int> stimuli;
    while (nn::dataLoaders::sqliteStepRow(db.db, q.stmt, "ThesisDataset: reading pairs"))
    {
        const long long id = sqlite3_column_int64(q.stmt, 0);
        ThesisSample s;
        s.subject_id = sqlite3_column_int(q.stmt, 1);
        s.stimulus = sqlite3_column_int(q.stmt, 2);
        s.text_phrase = stimulus_to_phrase(names, s.stimulus);
        const auto voice = blob_floats(q.stmt, 3, voice_len, "voice", id);
        const auto eeg = blob_floats(q.stmt, 4, n_channels * eeg_len, "EEG", id);
        s.audio = nn::Tensor(static_cast<nn::Index>(voice_len), 1);
        std::copy(voice.begin(), voice.end(), s.audio.mutable_data_ptr());
        s.eeg = nn::Tensor(static_cast<nn::Index>(n_channels), static_cast<nn::Index>(eeg_len));
        std::copy(eeg.begin(), eeg.end(), s.eeg.mutable_data_ptr());
        speakers.insert(s.subject_id);
        stimuli.insert(s.stimulus);
        view.samples.push_back(std::move(s));
    }
    if (view.samples.empty()) throw std::runtime_error("ThesisDataset: " + path + " has no pairs");
    view.n_subjects = static_cast<int>(speakers.size());
    view.n_stimuli = static_cast<int>(stimuli.size());
    return view;
}
} // namespace

auto load_dataset(const ThesisConfig::Dataset& dataset_cfg) -> ThesisDatasetView
{
    const std::string root = nn::utility::expand_home(dataset_cfg.root);
    if (dataset_cfg.source == "biomex2021")
    {
        // ThesisConfig::validate refuses an imagined condition for this base.
        ThesisDatasetView view = load_paired(root);
        NN_LOG_INFO("ThesisDataset: loaded " + std::to_string(view.samples.size()) +
                    " BIOMEX-DB pairs from " + root);
        apply_max_samples(view, dataset_cfg.max_samples);
        return view;
    }
    if (dataset_cfg.source != "coretto2017")
        throw std::invalid_argument(
            "ThesisDataset: unknown dataset.source \"" + *dataset_cfg.source + "\"");

    auto subjects = discoverSubjects(root, "^S(\\d+)$");
    if (subjects.empty()) throw std::runtime_error("ThesisDataset: no subjects found in " + root);

    // A voice run reads no EEG, but the view always pairs (every modality runs on the
    // same trials): it loads the same-trial EEG and never looks at it.
    const bool reads_eeg = dataset_cfg.modality != "voice";
    const EegCondition condition =
        reads_eeg ? parse_eeg_condition(dataset_cfg.eeg_condition) : EegCondition::Pronounced;
    const std::uint32_t pairing_seed =
        condition == EegCondition::Imagined ? dataset_cfg.eeg_pairing_seed.get() : 0u;

    ThesisDatasetView view;
    bool from_cache = false;

    // Try the decoded-dataset cache: signature is derived from the (cheap)
    // subject discovery above, so a changed source invalidates it without a
    // content re-read. The pairing is part of the file name and of the signature.
    const std::string cache_file = cache_path_for(root, condition, pairing_seed);
    std::uint64_t signature = source_signature(subjects);
    signature ^= (static_cast<std::uint64_t>(condition == EegCondition::Imagined) << 63) ^
                 (static_cast<std::uint64_t>(pairing_seed) << 17);
    if (!cache_disabled() && try_load_cache(cache_file, signature, view))
    {
        from_cache = true;
        NN_LOG_INFO("ThesisDataset: loaded " + std::to_string(view.samples.size()) +
                    " samples from decoded cache (" + cache_file + ")");
    }
    else
    {
        view = decode_full_dataset(subjects, discoverStimulusNames(root), condition, pairing_seed);
        if (view.samples.empty())
            throw std::runtime_error("ThesisDataset: no paired audio+EEG samples found in " + root);
        if (!cache_disabled())
        {
            write_cache(cache_file, signature, view);
            NN_LOG_INFO("ThesisDataset: wrote decoded cache with " +
                        std::to_string(view.samples.size()) + " samples (" + cache_file + ")");
        }
    }
    (void) from_cache;

    view.voice_sample_rate = kCorettoVoiceRateHz;
    view.eeg_sample_rate = kCorettoEegRateHz;
    apply_max_samples(view, dataset_cfg.max_samples);
    return view;
}

auto make_text_split(
    const std::vector<ThesisSample>& samples, const std::string& text_mode, uint32_t seed)
    -> TextSplit
{
    if (text_mode != "dependent" && text_mode != "independent")
        throw std::invalid_argument("ThesisDataset: unknown text_mode " + text_mode);

    // The phrases, in a seeded order: the first half trains, the next half of the same size
    // is the "unseen" test set; with an odd count the last phrase is used by neither mode.
    std::vector<std::string> phrases;
    for (const auto& s : samples)
        if (std::find(phrases.begin(), phrases.end(), s.text_phrase) == phrases.end())
            phrases.push_back(s.text_phrase);
    if (phrases.size() < 2)
        throw std::invalid_argument("ThesisDataset: the text split needs at least 2 phrases, got " +
                                    std::to_string(phrases.size()));
    std::sort(phrases.begin(), phrases.end()); // order of first appearance must not matter
    std::mt19937 rng(seed);
    std::shuffle(phrases.begin(), phrases.end(), rng);
    const auto half = static_cast<ptrdiff_t>(phrases.size() / 2);
    const std::unordered_set<std::string> seen(phrases.begin(), phrases.begin() + half);
    const std::unordered_set<std::string> unseen(
        phrases.begin() + half, phrases.begin() + 2 * half);
    const auto& test_phrases = (text_mode == "dependent") ? seen : unseen;

    TextSplit ts;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (seen.count(samples[i].text_phrase)) ts.train_indices.push_back(i);
        if (test_phrases.count(samples[i].text_phrase)) ts.test_indices.push_back(i);
    }
    return ts;
}

auto build_speaker_map(const std::vector<ThesisSample>& samples,
    const std::vector<std::vector<double>>& feature_vectors) -> SpeakerFeatureMap
{
    if (samples.size() != feature_vectors.size())
        throw std::invalid_argument("ThesisDataset: samples/feature_vectors size mismatch");

    SpeakerFeatureMap map;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        std::string key = "subject_" + std::to_string(samples[i].subject_id);
        map[key].push_back(feature_vectors[i]);
    }
    return map;
}

} // namespace thesis
