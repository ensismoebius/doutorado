// The Phase 01 protocol pieces added on 2026-11-12, each tested through its public API:
//   - EEG pairing (same trial / imagined by subject+stimulus), also end to end on a SQLite
//     mock with the real schema, where every EEG row carries its own original_row as value;
//   - the text split (seen vs unseen phrases, identical training set);
//   - the simulated dysphonia (whisper mix);
//   - the cosine control and the degraded test features in run_classifier;
//   - the Phase 00 speaker subsets.

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <numbers>
#include <set>
#include <string>
#include <vector>

#include "../lib/include/EegPairing.hpp"
#include "../lib/include/ParaconsistentSubsets.hpp"
#include "../lib/include/ThesisClassifiers.hpp"
#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisDataset.hpp"
#include "../lib/include/VoiceDegradation.hpp"
#include "../lib/src/ThesisAnnClassifier.hpp"
#include "../lib/src/ThesisDsnnClassifier.hpp"
#include "ThesisTestConfig.hpp"
#include "statistics/GenuineImpostorEERScorer.hpp"

using namespace thesis;

namespace
{
constexpr ModalityIds kIds{2, 1}; // the real table: Imagined = 1, Pronounced = 2

// ── EEG pairing ────────────────────────────────────────────────────────────────
std::vector<EegTrialMeta> sample_trials()
{
    // rows 0..3 imagined (stimuli 1,1,2,2), rows 4..5 pronounced (stimuli 1,2)
    return {{0, 1, 1}, {1, 1, 1}, {2, 1, 2}, {3, 1, 2}, {4, 2, 1}, {5, 2, 2}};
}
} // namespace

TEST(EegPairing, PronouncedIsTheSameTrial)
{
    const std::vector<VoiceTrialMeta> voice{{4, 1}, {5, 2}};
    const auto rows =
        pair_eeg_trials(voice, sample_trials(), kIds, EegCondition::Pronounced, 7u, 1);
    EXPECT_EQ(rows, (std::vector<long long>{4, 5}));
}

// The 2026-11-12 defect: reading original_row - 1 gave the previous trial, with another
// stimulus and often another modality. A pairing that does not match is refused.
TEST(EegPairing, PronouncedRefusesAnotherStimulusOrModality)
{
    std::vector<EegTrialMeta> trials = sample_trials();
    trials[4].stimulus_id = 2; // the voice says stimulus 1
    EXPECT_THROW((void) pair_eeg_trials({{4, 1}}, trials, kIds, EegCondition::Pronounced, 7u, 1),
        std::runtime_error);
    EXPECT_THROW( // row 3 exists but is imagined
        (void) pair_eeg_trials({{3, 2}}, sample_trials(), kIds, EegCondition::Pronounced, 7u, 1),
        std::runtime_error);
}

TEST(EegPairing, ImaginedMatchesStimulusWithoutReplacement)
{
    const std::vector<VoiceTrialMeta> voice{{4, 1}, {10, 1}, {5, 2}};
    const auto rows = pair_eeg_trials(voice, sample_trials(), kIds, EegCondition::Imagined, 7u, 1);
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_TRUE(rows[0] == 0 || rows[0] == 1);
    EXPECT_TRUE(rows[1] == 0 || rows[1] == 1);
    EXPECT_NE(rows[0], rows[1]); // no imagined trial used twice
    EXPECT_TRUE(rows[2] == 2 || rows[2] == 3);
    // deterministic in the seed
    EXPECT_EQ(rows, pair_eeg_trials(voice, sample_trials(), kIds, EegCondition::Imagined, 7u, 1));
}

TEST(EegPairing, ImaginedRefusesTooFewTrials)
{
    const std::vector<VoiceTrialMeta> voice{{4, 1}, {10, 1}, {11, 1}}; // 3 needed, 2 exist
    EXPECT_THROW(
        (void) pair_eeg_trials(voice, sample_trials(), kIds, EegCondition::Imagined, 7u, 1),
        std::runtime_error);
}

TEST(EegPairing, UnknownConditionIsRefused)
{
    EXPECT_THROW((void) parse_eeg_condition("silent"), std::invalid_argument);
}

// ── End to end on a SQLite mock ──────────────────────────────────────────────────
namespace
{
constexpr std::size_t kAudio = 176400; // ImaginedSpeechSchema_10_1117: 4 s at 44.1 kHz
constexpr std::size_t kEeg = 4096;     // per channel: 4 s at 1024 Hz

void exec(sqlite3* db, const std::string& sql)
{
    char* err = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK)
    {
        const std::string msg = err != nullptr ? err : "?";
        sqlite3_free(err);
        throw std::runtime_error("mock db: " + msg + " in " + sql);
    }
}

// Two subjects. Per subject: imagined rows 0..3 (stimuli 1,1,2,2), pronounced rows 4,5
// (stimuli 1,2) with audio. Every EEG row's channels hold 1000*subject + original_row.
std::string make_mock_db()
{
    const auto dir =
        std::filesystem::temp_directory_path() / ("thesis_pairing_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "database.sqlite").string();
    std::filesystem::remove(path);
    sqlite3* db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) throw std::runtime_error("open");
    exec(db,
        "CREATE TABLE subject(id INTEGER PRIMARY KEY AUTOINCREMENT, subject_name TEXT UNIQUE);"
        "CREATE TABLE modalities(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);"
        "CREATE TABLE stimulus(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);"
        "CREATE TABLE trial(id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER, "
        "modality_id INTEGER, stimulus_id INTEGER, eeg_sample_count INTEGER, "
        "audio_sample_count INTEGER, original_row INTEGER);"
        "CREATE TABLE eeg_samples(id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id INTEGER, "
        "F3 BLOB, F4 BLOB, C3 BLOB, C4 BLOB, P3 BLOB, P4 BLOB, blink INTEGER);"
        "CREATE TABLE audio_samples(id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id INTEGER, "
        "samples BLOB, audio_row INTEGER);"
        "INSERT INTO modalities(name) VALUES ('Imagined'), ('Pronounced');"
        "INSERT INTO stimulus(name) VALUES ('A'), ('E');"
        "INSERT INTO subject(subject_name) VALUES ('S01'), ('S02');");
    const struct
    {
        int row, modality, stimulus;
    } trials[] = {{0, 1, 1}, {1, 1, 1}, {2, 1, 2}, {3, 1, 2}, {4, 2, 1}, {5, 2, 2}};
    sqlite3_stmt* t = nullptr;
    sqlite3_stmt* e = nullptr;
    sqlite3_stmt* a = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO trial(subject_id, modality_id, stimulus_id, original_row) VALUES (?,?,?,?)",
        -1,
        &t,
        nullptr);
    sqlite3_prepare_v2(db,
        "INSERT INTO eeg_samples(trial_id, F3, F4, C3, C4, P3, P4, blink) VALUES (?,?,?,?,?,?,?,0)",
        -1,
        &e,
        nullptr);
    sqlite3_prepare_v2(db,
        "INSERT INTO audio_samples(trial_id, samples, audio_row) VALUES (?,?,?)",
        -1,
        &a,
        nullptr);
    exec(db, "BEGIN");
    for (int subject = 1; subject <= 2; ++subject)
    {
        int audio_row = 0;
        for (const auto& tr : trials)
        {
            sqlite3_bind_int(t, 1, subject);
            sqlite3_bind_int(t, 2, tr.modality);
            sqlite3_bind_int(t, 3, tr.stimulus);
            sqlite3_bind_int(t, 4, tr.row);
            sqlite3_step(t);
            sqlite3_reset(t);
            const auto trial_id = sqlite3_last_insert_rowid(db);

            const std::vector<float> channel(kEeg, static_cast<float>(1000 * subject + tr.row));
            sqlite3_bind_int64(e, 1, trial_id);
            for (int c = 2; c <= 7; ++c)
                sqlite3_bind_blob(e,
                    c,
                    channel.data(),
                    static_cast<int>(channel.size() * sizeof(float)),
                    SQLITE_TRANSIENT);
            sqlite3_step(e);
            sqlite3_reset(e);

            if (tr.modality == 2)
            {
                const std::vector<float> audio(kAudio, static_cast<float>(tr.stimulus));
                sqlite3_bind_int64(a, 1, trial_id);
                sqlite3_bind_blob(a,
                    2,
                    audio.data(),
                    static_cast<int>(audio.size() * sizeof(float)),
                    SQLITE_TRANSIENT);
                sqlite3_bind_int(a, 3, audio_row++);
                sqlite3_step(a);
                sqlite3_reset(a);
            }
        }
    }
    exec(db, "COMMIT");
    sqlite3_finalize(t);
    sqlite3_finalize(e);
    sqlite3_finalize(a);
    sqlite3_close(db);
    return path;
}

ThesisConfig::Dataset mock_dataset(const std::string& root, const std::string& condition)
{
    ThesisConfig::Dataset d;
    d.root = root;
    d.results_dir = "results/thesis";
    d.modality = "eeg";
    d.max_samples = 0;
    d.eeg_condition = condition;
    if (condition == "imagined") d.eeg_pairing_seed = 11u;
    return d;
}

float eeg_code(const ThesisSample& s)
{
    return s.eeg.at(0, 0);
}
} // namespace

TEST(ThesisDatasetPairing, PronouncedReadsTheSameTrialNotThePreviousOne)
{
    ::setenv("THESIS_NO_DATASET_CACHE", "1", 1);
    const auto view = load_dataset(mock_dataset(make_mock_db(), "pronounced"));
    ASSERT_EQ(view.samples.size(), 4u);
    for (const auto& s : view.samples)
    {
        const float expected_row = s.stimulus == 1 ? 4.0f : 5.0f; // its own trial
        EXPECT_EQ(eeg_code(s), static_cast<float>(1000 * s.subject_id) + expected_row);
        EXPECT_EQ(s.audio.at(0, 0), static_cast<float>(s.stimulus)); // audio of the same trial
    }
}

TEST(ThesisDatasetPairing, ImaginedReadsAnImaginedTrialOfTheSameStimulus)
{
    ::setenv("THESIS_NO_DATASET_CACHE", "1", 1);
    const auto view = load_dataset(mock_dataset(make_mock_db(), "imagined"));
    ASSERT_EQ(view.samples.size(), 4u);
    for (const auto& s : view.samples)
    {
        const float row = eeg_code(s) - static_cast<float>(1000 * s.subject_id);
        if (s.stimulus == 1)
            EXPECT_TRUE(row == 0.0f || row == 1.0f) << row;
        else
            EXPECT_TRUE(row == 2.0f || row == 3.0f) << row;
    }
}

// ── Text split ───────────────────────────────────────────────────────────────────
namespace
{
std::vector<ThesisSample> phrase_samples(int n_phrases, int per_phrase)
{
    std::vector<ThesisSample> out;
    for (int p = 0; p < n_phrases; ++p)
        for (int k = 0; k < per_phrase; ++k)
        {
            ThesisSample s;
            s.subject_id = 1 + k % 3;
            s.stimulus = p + 1;
            s.text_phrase = "phrase" + std::to_string(p);
            out.push_back(std::move(s));
        }
    return out;
}

std::set<std::string> phrases_of(const std::vector<ThesisSample>& s, const std::vector<size_t>& idx)
{
    std::set<std::string> out;
    for (size_t i : idx) out.insert(s[i].text_phrase);
    return out;
}
} // namespace

TEST(TextSplit, BothModesTrainOnTheSameSamples)
{
    const auto samples = phrase_samples(11, 6);
    const auto dep = make_text_split(samples, "dependent", 42u);
    const auto ind = make_text_split(samples, "independent", 42u);
    EXPECT_EQ(dep.train_indices, ind.train_indices);
    EXPECT_EQ(phrases_of(samples, dep.train_indices).size(), 5u);
}

TEST(TextSplit, DependentTestsSeenPhrasesIndependentTestsUnseenOfEqualCount)
{
    const auto samples = phrase_samples(11, 6);
    const auto dep = make_text_split(samples, "dependent", 42u);
    const auto ind = make_text_split(samples, "independent", 42u);
    const auto train = phrases_of(samples, dep.train_indices);
    EXPECT_EQ(phrases_of(samples, dep.test_indices), train);
    const auto unseen = phrases_of(samples, ind.test_indices);
    EXPECT_EQ(unseen.size(), train.size());
    for (const auto& p : unseen) EXPECT_EQ(train.count(p), 0u) << p;
    EXPECT_EQ(dep.test_indices.size(), ind.test_indices.size());
}

TEST(TextSplit, RefusesFewerThanTwoPhrasesAndUnknownModes)
{
    EXPECT_THROW(
        (void) make_text_split(phrase_samples(1, 4), "dependent", 1u), std::invalid_argument);
    EXPECT_THROW((void) make_text_split(phrase_samples(4, 4), "both", 1u), std::invalid_argument);
}

// ── Simulated dysphonia ──────────────────────────────────────────────────────────
namespace
{
// A voiced vowel: a 150 Hz pulse train through two resonances, 0.5 s at 44.1 kHz.
std::vector<float> voiced(double fs)
{
    const auto n = static_cast<std::size_t>(0.5 * fs);
    std::vector<double> e(n, 0.0);
    const auto period = static_cast<std::size_t>(fs / 150.0);
    for (std::size_t i = 0; i < n; i += period) e[i] = 1.0;
    std::vector<double> y(n, 0.0);
    for (double f : {700.0, 1200.0})
    {
        const double r = 0.98;
        const double c = 2.0 * r * std::cos(2.0 * std::numbers::pi * f / fs);
        std::vector<double> z(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            z[i] = e[i] + (i >= 1 ? c * z[i - 1] : 0.0) - (i >= 2 ? r * r * z[i - 2] : 0.0);
        for (std::size_t i = 0; i < n; ++i) y[i] += z[i];
    }
    return {y.begin(), y.end()};
}

double power(const std::vector<float>& x)
{
    double p = 0.0;
    for (float v : x) p += static_cast<double>(v) * v;
    return p / static_cast<double>(x.size());
}

// Normalized autocorrelation at the pitch lag: near 1 for a voiced signal, near 0 for noise.
double periodicity(const std::vector<float>& x, std::size_t lag)
{
    double num = 0.0, den = 0.0;
    for (std::size_t i = lag; i < x.size(); ++i)
    {
        num += static_cast<double>(x[i]) * x[i - lag];
        den += static_cast<double>(x[i]) * x[i];
    }
    return num / den;
}
} // namespace

TEST(VoiceDegradation, WhisperKeepsThePowerAndRemovesThePitch)
{
    const double fs = 44100.0;
    const auto x = voiced(fs);
    const auto w = degrade_voice(x, fs, 1.0f, 3u);
    const auto lag = static_cast<std::size_t>(fs / 150.0);
    EXPECT_GT(periodicity(x, lag), 0.8);
    EXPECT_LT(std::abs(periodicity(w, lag)), 0.3);
    EXPECT_NEAR(power(w) / power(x), 1.0, 0.2);
}

TEST(VoiceDegradation, LevelIsTheBreathFractionAndThePowerStays)
{
    const double fs = 44100.0;
    const auto x = voiced(fs);
    const auto lag = static_cast<std::size_t>(fs / 150.0);
    double last = periodicity(x, lag);
    for (float level : {0.25f, 0.5f, 0.75f})
    {
        const auto y = degrade_voice(x, fs, level, 3u);
        EXPECT_NEAR(power(y) / power(x), 1.0, 0.2) << level;
        const double p = periodicity(y, lag);
        EXPECT_LT(p, last) << level; // more breath, less pitch
        last = p;
    }
}

TEST(VoiceDegradation, DeterministicSilentAndStrict)
{
    const double fs = 44100.0;
    const auto x = voiced(fs);
    EXPECT_EQ(degrade_voice(x, fs, 0.5f, 9u), degrade_voice(x, fs, 0.5f, 9u));
    EXPECT_NE(degrade_voice(x, fs, 0.5f, 9u), degrade_voice(x, fs, 0.5f, 10u));
    const std::vector<float> silence(4000, 0.0f);
    for (float v : whisper(silence, fs, 1u)) EXPECT_EQ(v, 0.0f);
    EXPECT_THROW((void) degrade_voice(x, fs, 0.0f, 1u), std::invalid_argument);
    EXPECT_THROW((void) degrade_voice(x, fs, 1.5f, 1u), std::invalid_argument);
}

// ── Cosine control and degraded test features ────────────────────────────────────
namespace
{
// 8 speakers x 8 samples, 2 phrases; speaker s has features centred on e_s.
ThesisDatasetView speakers_view()
{
    ThesisDatasetView v;
    v.n_subjects = 8;
    v.n_stimuli = 2;
    for (int s = 1; s <= 8; ++s)
        for (int k = 0; k < 8; ++k)
        {
            ThesisSample x;
            x.subject_id = s;
            x.stimulus = 1 + k % 2;
            x.text_phrase = k % 2 == 0 ? "a" : "e";
            v.samples.push_back(std::move(x));
        }
    return v;
}

std::vector<std::vector<double>> speaker_features(const ThesisDatasetView& v, double noise)
{
    std::vector<std::vector<double>> f;
    std::uint32_t state = 12345u;
    auto next = [&]()
    {
        state = state * 1664525u + 1013904223u;
        return (static_cast<double>(state >> 8) / 16777216.0) - 0.5;
    };
    for (const auto& s : v.samples)
    {
        std::vector<double> row(8, 0.0);
        row[static_cast<size_t>(s.subject_id - 1)] = 1.0;
        for (auto& x : row) x += noise * next();
        f.push_back(std::move(row));
    }
    return f;
}

ThesisConfig cosine_cfg()
{
    ThesisConfig c = test::phase00("voice");
    test::set_handcrafted(c);
    c.paraconsistent = ThesisConfig::Paraconsistent{};
    c.paraconsistent.enabled = false;
    test::set_classifier(c, "cosine");
    c.training.k_folds = 2;
    c.validate();
    return c;
}
} // namespace

TEST(CosineControl, ScoresTheFeaturesWithoutTraining)
{
    const auto view = speakers_view();
    const auto fvs = speaker_features(view, 0.2);
    const statistics::GenuineImpostorEERScorer scorer(1);
    const auto r = run_classifier(view, fvs, "hc", cosine_cfg(), scorer);
    EXPECT_EQ(r.param_count, 0u);
    EXPECT_EQ(r.mean_train_ms, 0.0);
    EXPECT_LT(r.mean_eer, 0.1); // separable speakers
    for (const auto& f : r.outer_folds) EXPECT_TRUE(f.history.empty());
}

TEST(CosineControl, NestedCvIsRefused)
{
    ThesisConfig c = cosine_cfg();
    c.training.nested_cv = true;
    c.training.early_stop_patience = 3;
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

// Degraded test rows hurt; with scope "probes" the enrolment rows stay clean, so the
// result differs from degrading everything. Pure noise as the "degraded" twin makes the
// direction unambiguous.
TEST(DegradedTestFeatures, ReplaceOnlyTestRowsAndKeepEnrolmentWhenAsked)
{
    const auto view = speakers_view();
    const auto clean = speaker_features(view, 0.2);
    auto noise = speaker_features(view, 0.0);
    std::uint32_t st = 99u;
    for (auto& row : noise)
        for (auto& x : row)
        {
            st = st * 1664525u + 1013904223u;
            x = static_cast<double>(st >> 8) / 16777216.0;
        }
    const statistics::GenuineImpostorEERScorer scorer(1);
    const ThesisConfig cfg = cosine_cfg();

    const auto r_clean = run_classifier(view, clean, "hc", cfg, scorer);
    DegradedTestFeatures all{&noise, false, 1};
    const auto r_all = run_classifier(view, clean, "hc", cfg, scorer, 0, nullptr, &all);
    DegradedTestFeatures probes{&noise, true, 1};
    const auto r_probes = run_classifier(view, clean, "hc", cfg, scorer, 0, nullptr, &probes);

    EXPECT_GT(r_all.mean_eer, r_clean.mean_eer + 0.2);
    EXPECT_GT(r_probes.mean_eer, r_clean.mean_eer + 0.2);
    EXPECT_NE(r_probes.mean_eer, r_all.mean_eer);

    DegradedTestFeatures misaligned{&clean, true, 0};
    EXPECT_THROW((void) run_classifier(view, clean, "hc", cfg, scorer, 0, nullptr, &misaligned),
        std::invalid_argument);
}

// ── Speaker subsets ──────────────────────────────────────────────────────────────
TEST(ParaconsistentSubsets, LeaveOneOutAndPhase01TrainingFolds)
{
    const auto view = speakers_view();
    const FeatureSet fs{"hc", speaker_features(view, 0.2)};
    const auto scores = score_speaker_subsets(view.samples, {fs}, 4, 42u, 3);
    std::size_t loso = 0, folds = 0;
    for (const auto& s : scores)
    {
        if (s.subset.rfind("loso:", 0) == 0) ++loso;
        if (s.subset.rfind("fold:", 0) == 0) ++folds;
        EXPECT_EQ(s.label, "hc");
        EXPECT_TRUE(std::isfinite(s.d_penalized));
    }
    EXPECT_EQ(loso, 8u);       // one per speaker
    EXPECT_EQ(folds, 4u * 3u); // k folds x repeats
    EXPECT_THROW(
        (void) score_speaker_subsets(view.samples, {fs}, 1, 42u, 3), std::invalid_argument);
}

// ── Config keys ──────────────────────────────────────────────────────────────────
TEST(ThesisProtocolConfig, EegConditionAndDegradationKeysFollowTheRun)
{
    ThesisConfig eeg = test::phase00("eeg");
    test::set_handcrafted(eeg);
    eeg.validate();
    eeg.dataset.eeg_condition = "imagined"; // needs its pairing seed
    EXPECT_THROW(eeg.validate(), std::invalid_argument);
    eeg.dataset.eeg_pairing_seed = 5u;
    eeg.validate();

    ThesisConfig voice = cosine_cfg();
    voice.degradation.enabled = true; // needs level, scope, seed
    EXPECT_THROW(voice.validate(), std::invalid_argument);
    voice.degradation.level = 0.5f;
    voice.degradation.scope = "probes";
    voice.degradation.seed = 3u;
    voice.validate();
    voice.degradation.scope = "everything";
    EXPECT_THROW(voice.validate(), std::invalid_argument);
}

// ── The ANN twin and the inference energy ───────────────────────────────────────

namespace
{
ThesisConfig network_cfg(const std::string& type)
{
    ThesisConfig c = test::phase00("voice");
    test::set_handcrafted(c);
    c.paraconsistent = ThesisConfig::Paraconsistent{};
    c.paraconsistent.enabled = false;
    test::set_classifier(c, type);
    // Each fold saves its model under results_dir: keep it out of the source tree.
    c.dataset.results_dir =
        (std::filesystem::temp_directory_path() / "thesis_protocol_gtest").string();
    c.training.k_folds = 2;
    c.training.epochs = 3;
    c.training.learning_rate = 0.001f;
    c.training.samples_per_batch = 8;
    c.training.optimizer_type = "adam";
    c.training.gradient_clip_norm = 0.0f;
    c.validate();
    return c;
}
} // namespace

// The twin differs from the DSNN in spiking only: same graph, same initial weights.
TEST(AnnTwin, StartsFromTheDsnnWeights)
{
    ThesisDsnnClassifier dsnn(6, 8, 3, 2, 42u, std::nullopt, std::nullopt);
    ThesisAnnClassifier ann(6, 8, 3, 2, 42u);
    const auto a = dsnn.state_dict();
    const auto b = ann.state_dict();
    std::size_t compared = 0;
    for (const auto& [key, tensor] : b)
    {
        ASSERT_TRUE(a.count(key)) << key << " has no DSNN counterpart";
        ASSERT_EQ(a.at(key).size(), tensor.size()) << key;
        for (nn::Index i = 0; i < tensor.rows(); ++i)
            for (nn::Index j = 0; j < tensor.cols(); ++j)
                ASSERT_EQ(a.at(key).at(i, j), tensor.at(i, j)) << key;
        ++compared;
    }
    EXPECT_EQ(compared, 6u); // fc_in, hidden_fc.0, fc_out: weight + bias each
    EXPECT_EQ(ann.params().size(), 6u);
}

// A dense net's arithmetic is its weight count: 6x8 + 8x8 + 8x3 = 136 MACs per sample.
TEST(InferenceEnergy, AnnCountsEveryWeightOnce)
{
    ThesisAnnClassifier ann(6, 8, 3, 2, 42u);
    (void) ann.forward(nn::Tensor::zeros(5, 6), false);
    const auto e = ann.inference_ledger();
    EXPECT_DOUBLE_EQ(e.fp_macs, 136.0);
    EXPECT_GT(e.total_nj(), 0.0);
}

// The DSNN's MACs are spike-independent: the first layer once (6x8), the leak of 2 LIF layers
// over T = 16 steps (2 x 16 x 8) and the readout scaling (3): 307. Computed at every step, the
// first layer costs 16x: 307 - 48 + 16 x 48 = 1027. Spikes add accumulates, not MACs.
TEST(InferenceEnergy, DsnnFirstLayerOnceOrAtEveryStep)
{
    ThesisDsnnClassifier dsnn(6, 8, 3, 2, 42u, std::nullopt, std::nullopt);
    nn::Tensor x = nn::Tensor::zeros(4, 6);
    for (nn::Index i = 0; i < 4; ++i)
        for (nn::Index j = 0; j < 6; ++j) x.at(i, j) = 3.0f * static_cast<float>((i + j) % 3);
    (void) dsnn.forward(x, false);
    const auto once = dsnn.inference_ledger(true);
    const auto every = dsnn.inference_ledger(false);
    EXPECT_DOUBLE_EQ(once.fp_macs, 307.0);
    EXPECT_DOUBLE_EQ(every.fp_macs, 1027.0);
    EXPECT_GT(every.total_nj(), once.total_nj());
    EXPECT_GT(dsnn.spikes_per_sample(), 0.0) << "the input must drive some spikes";
    // Every spike is accumulated into its fan-out: more input, more spikes, more adds.
    for (nn::Index i = 0; i < 4; ++i)
        for (nn::Index j = 0; j < 6; ++j) x.at(i, j) *= 4.0f;
    (void) dsnn.forward(x, false);
    EXPECT_GE(dsnn.spikes_per_sample(), 0.0);
    EXPECT_THROW((void) ThesisDsnnClassifier(6, 8, 3, 2, 1u, std::nullopt, std::nullopt)
                     .inference_ledger(true),
        std::logic_error); // no forward yet
    // tdBN adds one MAC per (neuron, step) before each of the 2 LIF layers: 307 + 2 x 16 x 8.
    ThesisDsnnClassifier bn(6, 8, 3, 2, 42u, std::nullopt, 1.0f);
    (void) bn.forward(x, false);
    EXPECT_DOUBLE_EQ(bn.inference_ledger(true).fp_macs, 307.0 + 256.0);
}

// Every network fold measures its test pass; the cosine control has none.
TEST(InferenceEnergy, RecordedPerFoldForNetworksOnly)
{
    const auto view = speakers_view();
    const auto fvs = speaker_features(view, 0.2);
    const statistics::GenuineImpostorEERScorer scorer(1);
    for (const std::string type : {"dsnn", "ann"})
    {
        const auto r = run_classifier(view, fvs, "hc", network_cfg(type), scorer);
        ASSERT_TRUE(r.mean_energy.has_value()) << type;
        for (const auto& f : r.outer_folds) ASSERT_TRUE(f.energy.has_value()) << type;
        EXPECT_GT(r.mean_energy->nj, 0.0) << type;
        if (type == "ann")
            EXPECT_DOUBLE_EQ(r.mean_energy->nj, r.mean_energy->nj_as_executed);
        else
            EXPECT_GT(r.mean_energy->nj_as_executed, r.mean_energy->nj);
    }
    const auto c = run_classifier(view, fvs, "hc", cosine_cfg(), scorer);
    EXPECT_FALSE(c.mean_energy.has_value());
}
