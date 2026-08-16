#include "ThesisDataset.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>

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
// Set THESIS_NO_DATASET_CACHE=1 to bypass entirely. Any mismatch/read error falls
// back to a normal decode, so a stale or truncated cache is never fatal.
constexpr char kCacheMagic[8] = {'E', '0', '5', 'D', 'S', 'C', '\0', '\1'}; // last byte = version

bool cache_disabled()
{
    const char* v = std::getenv("THESIS_NO_DATASET_CACHE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

std::string cache_path_for(const std::string& root)
{
    return root + ".e05dscache";
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

    std::int32_t n_subjects = 0;
    std::int32_t n_stimuli = 0;
    std::uint64_t n_samples = 0;
    if (!read_pod(is, n_subjects) || !read_pod(is, n_stimuli) || !read_pod(is, n_samples))
        return false;

    ThesisDatasetView loaded;
    loaded.n_subjects = n_subjects;
    loaded.n_stimuli = n_stimuli;
    loaded.samples.reserve(n_samples);
    for (std::uint64_t i = 0; i < n_samples; ++i)
    {
        ThesisSample s;
        if (!read_pod(is, s.stimulus) || !read_pod(is, s.subject_id) ||
            !read_string(is, s.text_phrase) || !read_tensor(is, s.audio) || !read_tensor(is, s.eeg))
            return false;
        loaded.samples.push_back(std::move(s));
    }

    std::uint64_t n_files = 0;
    if (!read_pod(is, n_files)) return false;
    loaded.subject_files.reserve(n_files);
    for (std::uint64_t i = 0; i < n_files; ++i)
    {
        SubjectFiles sf;
        std::uint64_t eeg_rows = 0;
        std::uint64_t audio_rows = 0;
        if (!read_pod(is, sf.subject_id) || !read_string(is, sf.subject_name) ||
            !read_string(is, sf.eeg_path) || !read_string(is, sf.audio_path) ||
            !read_pod(is, eeg_rows) || !read_pod(is, audio_rows))
            return false;
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
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) return;
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
        if (!os) return; // write failed — leave no cache rather than a bad one
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
}

// Stimulus → text phrase mapping for the 10.1117 dataset.
// Stimuli 1-5: vowels /a/ /e/ /i/ /o/ /u/
// Stimuli 6-10: directional words
static const std::vector<std::string> kStimulToPhrase = {"", // index 0 unused
    "a",
    "e",
    "i",
    "o",
    "u",
    "arriba",
    "abajo",
    "izquierda",
    "derecha",
    "adelante"};

std::string stimulus_to_phrase(int stim)
{
    if (stim >= 1 && stim < static_cast<int>(kStimulToPhrase.size()))
        return kStimulToPhrase[static_cast<size_t>(stim)];
    return "unknown";
}

// Full decode from source .mat files, ignoring max_samples (applied by the
// caller). Returns the complete decoded set — the unit the cache stores.
auto decode_full_dataset(const std::vector<SubjectFiles>& subjects) -> ThesisDatasetView
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
    // which samples are included. Samples where either audio or EEG data is
    // absent are dropped to guarantee comparability across modality runs.
    for (const auto& sf : subjects)
    {
        if (sf.audio_path.empty() || sf.eeg_path.empty())
            continue; // skip subjects missing either modality

        nn::dataLoaders::AudioSession audio_session(sf.audio_path, sf.subject_id);
        nn::dataLoaders::EEGSession eeg_session(sf.eeg_path, sf.subject_id);

        const size_t n_audio = audio_session.rowCount();
        const size_t n_eeg = eeg_session.rowCount();
        if (n_audio == 0 || n_eeg == 0) continue;

        int paired = 0;
        for (size_t row = 0; row < n_audio; ++row)
        {
            auto [audio_tensor, stimulus, eeg_index] = audio_session.readRow(row);

            // eeg_index is 1-based in the dataset; convert to 0-based.
            const size_t eeg_row = (eeg_index > 0) ? static_cast<size_t>(eeg_index - 1)
                                                   : row; // fallback: same-row pairing

            if (eeg_row >= n_eeg) continue; // eeg_index out of range — drop trial

            auto [eeg_tensor, eeg_labels] = eeg_session.readRow(eeg_row);

            ThesisSample sample;
            sample.audio = std::move(audio_tensor);
            sample.eeg = std::move(eeg_tensor);
            sample.stimulus = stimulus;
            sample.subject_id = sf.subject_id;
            sample.text_phrase = stimulus_to_phrase(stimulus);
            stimuli_seen.insert(stimulus);
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

auto load_dataset(const ThesisConfig::Dataset& dataset_cfg) -> ThesisDatasetView
{
    const std::string root = nn::utility::expand_home(dataset_cfg.root);
    auto subjects = discoverSubjects(root, "^S(\\d+)$");
    if (subjects.empty()) throw std::runtime_error("ThesisDataset: no subjects found in " + root);

    ThesisDatasetView view;
    bool from_cache = false;

    // Try the decoded-dataset cache: signature is derived from the (cheap)
    // subject discovery above, so a changed source invalidates it without a
    // content re-read.
    const std::string cache_file = cache_path_for(root);
    const std::uint64_t signature = source_signature(subjects);
    if (!cache_disabled() && try_load_cache(cache_file, signature, view))
    {
        from_cache = true;
        NN_LOG_INFO("ThesisDataset: loaded " + std::to_string(view.samples.size()) +
                    " samples from decoded cache (" + cache_file + ")");
    }
    else
    {
        view = decode_full_dataset(subjects);
        if (view.samples.empty())
            throw std::runtime_error(
                "ThesisDataset: no paired audio+EEG samples found. "
                "Check that each subject has both audio and EEG .mat files.");
        if (!cache_disabled())
        {
            write_cache(cache_file, signature, view);
            NN_LOG_INFO("ThesisDataset: wrote decoded cache with " +
                        std::to_string(view.samples.size()) + " samples (" + cache_file + ")");
        }
    }
    (void) from_cache;

    apply_max_samples(view, dataset_cfg.max_samples);
    return view;
}

auto make_text_split(
    const std::vector<ThesisSample>& samples, const std::string& text_mode, uint32_t seed)
    -> TextSplit
{
    if (text_mode == "dependent")
    {
        // All phrases available at both train and test; split by utterance index.
        std::vector<size_t> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
        size_t split_point = indices.size() * 8 / 10; // 80/20
        TextSplit ts;
        ts.train_indices = {indices.begin(), indices.begin() + static_cast<ptrdiff_t>(split_point)};
        ts.test_indices = {indices.begin() + static_cast<ptrdiff_t>(split_point), indices.end()};
        return ts;
    }

    if (text_mode == "independent")
    {
        // Collect unique phrases, split phrases into train-set and test-set halves.
        std::vector<std::string> phrases;
        for (const auto& s : samples)
        {
            if (std::find(phrases.begin(), phrases.end(), s.text_phrase) == phrases.end())
                phrases.push_back(s.text_phrase);
        }
        std::mt19937 rng(seed);
        std::shuffle(phrases.begin(), phrases.end(), rng);
        size_t phrase_split = phrases.size() / 2;
        std::unordered_set<std::string> train_phrases(
            phrases.begin(), phrases.begin() + static_cast<ptrdiff_t>(phrase_split));

        TextSplit ts;
        for (size_t i = 0; i < samples.size(); ++i)
        {
            if (train_phrases.count(samples[i].text_phrase))
                ts.train_indices.push_back(i);
            else
                ts.test_indices.push_back(i);
        }
        return ts;
    }

    throw std::invalid_argument("ThesisDataset: unknown text_mode " + text_mode);
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
