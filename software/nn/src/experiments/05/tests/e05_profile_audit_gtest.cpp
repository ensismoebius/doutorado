// Profile audit: every Experiment05 profile must parse cleanly and populate
// all required config fields. Catches silent key drift after profile edits.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "../lib/include/E05Config.hpp"
#include "nlohmann/json.hpp"
#include "optimizers/OptimizerFactory.hpp"

namespace fs = std::filesystem;
using e05::E05Config;

namespace
{

const std::vector<std::string>& all_profiles()
{
    static const std::vector<std::string> profiles = {
        "debug.json",
        "phase00/p00_ae_ann_base_eeg.json",
        "phase00/p00_ae_ann_base_voice.json",
        "phase00/p00_ae_ann_small_eeg.json",
        "phase00/p00_ae_ann_small_voice.json",
        "phase00/p00_ae_ann_tiny_eeg.json",
        "phase00/p00_ae_ann_tiny_voice.json",
        "phase00/p00_ae_snn_direct_base_eeg.json",
        "phase00/p00_ae_snn_direct_base_voice.json",
        "phase00/p00_ae_snn_direct_small_eeg.json",
        "phase00/p00_ae_snn_direct_small_voice.json",
        "phase00/p00_ae_snn_direct_tiny_eeg.json",
        "phase00/p00_ae_snn_direct_tiny_voice.json",
        "phase00/p00_ae_snn_latency_base_eeg.json",
        "phase00/p00_ae_snn_latency_base_voice.json",
        "phase00/p00_ae_snn_latency_small_eeg.json",
        "phase00/p00_ae_snn_latency_small_voice.json",
        "phase00/p00_ae_snn_latency_tiny_eeg.json",
        "phase00/p00_ae_snn_latency_tiny_voice.json",
        "phase00/p00_ae_snn_poisson_base_eeg.json",
        "phase00/p00_ae_snn_poisson_base_voice.json",
        "phase00/p00_ae_snn_poisson_small_eeg.json",
        "phase00/p00_ae_snn_poisson_small_voice.json",
        "phase00/p00_ae_snn_poisson_tiny_eeg.json",
        "phase00/p00_ae_snn_poisson_tiny_voice.json",
        "phase00/p00_hc_daub10_bark_c1_voice.json",
        "phase00/p00_hc_daub10_bark_c2_voice.json",
        "phase00/p00_hc_daub10_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub10_lfcc_c1_voice.json",
        "phase00/p00_hc_daub10_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub10_lfcc_c2_voice.json",
        "phase00/p00_hc_daub10_mel_c1_voice.json",
        "phase00/p00_hc_daub10_mel_c2_voice.json",
        "phase00/p00_hc_daub12_bark_c1_voice.json",
        "phase00/p00_hc_daub12_bark_c2_voice.json",
        "phase00/p00_hc_daub12_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub12_lfcc_c1_voice.json",
        "phase00/p00_hc_daub12_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub12_lfcc_c2_voice.json",
        "phase00/p00_hc_daub12_mel_c1_voice.json",
        "phase00/p00_hc_daub12_mel_c2_voice.json",
        "phase00/p00_hc_daub14_bark_c1_voice.json",
        "phase00/p00_hc_daub14_bark_c2_voice.json",
        "phase00/p00_hc_daub14_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub14_lfcc_c1_voice.json",
        "phase00/p00_hc_daub14_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub14_lfcc_c2_voice.json",
        "phase00/p00_hc_daub14_mel_c1_voice.json",
        "phase00/p00_hc_daub14_mel_c2_voice.json",
        "phase00/p00_hc_daub16_bark_c1_voice.json",
        "phase00/p00_hc_daub16_bark_c2_voice.json",
        "phase00/p00_hc_daub16_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub16_lfcc_c1_voice.json",
        "phase00/p00_hc_daub16_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub16_lfcc_c2_voice.json",
        "phase00/p00_hc_daub16_mel_c1_voice.json",
        "phase00/p00_hc_daub16_mel_c2_voice.json",
        "phase00/p00_hc_daub18_bark_c1_voice.json",
        "phase00/p00_hc_daub18_bark_c2_voice.json",
        "phase00/p00_hc_daub18_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub18_lfcc_c1_voice.json",
        "phase00/p00_hc_daub18_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub18_lfcc_c2_voice.json",
        "phase00/p00_hc_daub18_mel_c1_voice.json",
        "phase00/p00_hc_daub18_mel_c2_voice.json",
        "phase00/p00_hc_daub20_bark_c1_voice.json",
        "phase00/p00_hc_daub20_bark_c2_voice.json",
        "phase00/p00_hc_daub20_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub20_lfcc_c1_voice.json",
        "phase00/p00_hc_daub20_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub20_lfcc_c2_voice.json",
        "phase00/p00_hc_daub20_mel_c1_voice.json",
        "phase00/p00_hc_daub20_mel_c2_voice.json",
        "phase00/p00_hc_daub22_bark_c1_voice.json",
        "phase00/p00_hc_daub22_bark_c2_voice.json",
        "phase00/p00_hc_daub22_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub22_lfcc_c1_voice.json",
        "phase00/p00_hc_daub22_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub22_lfcc_c2_voice.json",
        "phase00/p00_hc_daub22_mel_c1_voice.json",
        "phase00/p00_hc_daub22_mel_c2_voice.json",
        "phase00/p00_hc_daub24_bark_c1_voice.json",
        "phase00/p00_hc_daub24_bark_c2_voice.json",
        "phase00/p00_hc_daub24_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub24_lfcc_c1_voice.json",
        "phase00/p00_hc_daub24_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub24_lfcc_c2_voice.json",
        "phase00/p00_hc_daub24_mel_c1_voice.json",
        "phase00/p00_hc_daub24_mel_c2_voice.json",
        "phase00/p00_hc_daub26_bark_c1_voice.json",
        "phase00/p00_hc_daub26_bark_c2_voice.json",
        "phase00/p00_hc_daub26_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub26_lfcc_c1_voice.json",
        "phase00/p00_hc_daub26_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub26_lfcc_c2_voice.json",
        "phase00/p00_hc_daub26_mel_c1_voice.json",
        "phase00/p00_hc_daub26_mel_c2_voice.json",
        "phase00/p00_hc_daub28_bark_c1_voice.json",
        "phase00/p00_hc_daub28_bark_c2_voice.json",
        "phase00/p00_hc_daub28_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub28_lfcc_c1_voice.json",
        "phase00/p00_hc_daub28_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub28_lfcc_c2_voice.json",
        "phase00/p00_hc_daub28_mel_c1_voice.json",
        "phase00/p00_hc_daub28_mel_c2_voice.json",
        "phase00/p00_hc_daub30_bark_c1_voice.json",
        "phase00/p00_hc_daub30_bark_c2_voice.json",
        "phase00/p00_hc_daub30_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub30_lfcc_c1_voice.json",
        "phase00/p00_hc_daub30_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub30_lfcc_c2_voice.json",
        "phase00/p00_hc_daub30_mel_c1_voice.json",
        "phase00/p00_hc_daub30_mel_c2_voice.json",
        "phase00/p00_hc_daub32_bark_c1_voice.json",
        "phase00/p00_hc_daub32_bark_c2_voice.json",
        "phase00/p00_hc_daub32_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub32_lfcc_c1_voice.json",
        "phase00/p00_hc_daub32_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub32_lfcc_c2_voice.json",
        "phase00/p00_hc_daub32_mel_c1_voice.json",
        "phase00/p00_hc_daub32_mel_c2_voice.json",
        "phase00/p00_hc_daub34_bark_c1_voice.json",
        "phase00/p00_hc_daub34_bark_c2_voice.json",
        "phase00/p00_hc_daub34_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub34_lfcc_c1_voice.json",
        "phase00/p00_hc_daub34_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub34_lfcc_c2_voice.json",
        "phase00/p00_hc_daub34_mel_c1_voice.json",
        "phase00/p00_hc_daub34_mel_c2_voice.json",
        "phase00/p00_hc_daub36_bark_c1_voice.json",
        "phase00/p00_hc_daub36_bark_c2_voice.json",
        "phase00/p00_hc_daub36_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub36_lfcc_c1_voice.json",
        "phase00/p00_hc_daub36_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub36_lfcc_c2_voice.json",
        "phase00/p00_hc_daub36_mel_c1_voice.json",
        "phase00/p00_hc_daub36_mel_c2_voice.json",
        "phase00/p00_hc_daub38_bark_c1_voice.json",
        "phase00/p00_hc_daub38_bark_c2_voice.json",
        "phase00/p00_hc_daub38_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub38_lfcc_c1_voice.json",
        "phase00/p00_hc_daub38_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub38_lfcc_c2_voice.json",
        "phase00/p00_hc_daub38_mel_c1_voice.json",
        "phase00/p00_hc_daub38_mel_c2_voice.json",
        "phase00/p00_hc_daub40_bark_c1_voice.json",
        "phase00/p00_hc_daub40_bark_c2_voice.json",
        "phase00/p00_hc_daub40_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub40_lfcc_c1_voice.json",
        "phase00/p00_hc_daub40_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub40_lfcc_c2_voice.json",
        "phase00/p00_hc_daub40_mel_c1_voice.json",
        "phase00/p00_hc_daub40_mel_c2_voice.json",
        "phase00/p00_hc_daub42_bark_c1_voice.json",
        "phase00/p00_hc_daub42_bark_c2_voice.json",
        "phase00/p00_hc_daub42_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub42_lfcc_c1_voice.json",
        "phase00/p00_hc_daub42_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub42_lfcc_c2_voice.json",
        "phase00/p00_hc_daub42_mel_c1_voice.json",
        "phase00/p00_hc_daub42_mel_c2_voice.json",
        "phase00/p00_hc_daub44_bark_c1_voice.json",
        "phase00/p00_hc_daub44_bark_c2_voice.json",
        "phase00/p00_hc_daub44_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub44_lfcc_c1_voice.json",
        "phase00/p00_hc_daub44_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub44_lfcc_c2_voice.json",
        "phase00/p00_hc_daub44_mel_c1_voice.json",
        "phase00/p00_hc_daub44_mel_c2_voice.json",
        "phase00/p00_hc_daub46_bark_c1_voice.json",
        "phase00/p00_hc_daub46_bark_c2_voice.json",
        "phase00/p00_hc_daub46_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub46_lfcc_c1_voice.json",
        "phase00/p00_hc_daub46_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub46_lfcc_c2_voice.json",
        "phase00/p00_hc_daub46_mel_c1_voice.json",
        "phase00/p00_hc_daub46_mel_c2_voice.json",
        "phase00/p00_hc_daub4_bark_c1_voice.json",
        "phase00/p00_hc_daub4_bark_c2_voice.json",
        "phase00/p00_hc_daub4_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub4_lfcc_c1_voice.json",
        "phase00/p00_hc_daub4_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub4_lfcc_c2_voice.json",
        "phase00/p00_hc_daub4_mel_c1_voice.json",
        "phase00/p00_hc_daub4_mel_c2_voice.json",
        "phase00/p00_hc_daub6_bark_c1_voice.json",
        "phase00/p00_hc_daub6_bark_c2_voice.json",
        "phase00/p00_hc_daub6_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub6_lfcc_c1_voice.json",
        "phase00/p00_hc_daub6_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub6_lfcc_c2_voice.json",
        "phase00/p00_hc_daub6_mel_c1_voice.json",
        "phase00/p00_hc_daub6_mel_c2_voice.json",
        "phase00/p00_hc_daub8_bark_c1_voice.json",
        "phase00/p00_hc_daub8_bark_c2_voice.json",
        "phase00/p00_hc_daub8_lfcc_c1_eeg.json",
        "phase00/p00_hc_daub8_lfcc_c1_voice.json",
        "phase00/p00_hc_daub8_lfcc_c2_eeg.json",
        "phase00/p00_hc_daub8_lfcc_c2_voice.json",
        "phase00/p00_hc_daub8_mel_c1_voice.json",
        "phase00/p00_hc_daub8_mel_c2_voice.json",
        "phase00/p00_hc_haar_bark_c1_voice.json",
        "phase00/p00_hc_haar_bark_c2_voice.json",
        "phase00/p00_hc_haar_lfcc_c1_eeg.json",
        "phase00/p00_hc_haar_lfcc_c1_voice.json",
        "phase00/p00_hc_haar_lfcc_c2_eeg.json",
        "phase00/p00_hc_haar_lfcc_c2_voice.json",
        "phase00/p00_hc_haar_mel_c1_voice.json",
        "phase00/p00_hc_haar_mel_c2_voice.json",
        "phase01/p01_dsnn_eeg_dep_flat_raw.json",
        "phase01/p01_dsnn_eeg_dep_flat_std.json",
        "phase01/p01_dsnn_eeg_dep_nested_raw.json",
        "phase01/p01_dsnn_eeg_dep_nested_std.json",
        "phase01/p01_dsnn_eeg_indep_flat_raw.json",
        "phase01/p01_dsnn_eeg_indep_flat_std.json",
        "phase01/p01_dsnn_eeg_indep_nested_raw.json",
        "phase01/p01_dsnn_eeg_indep_nested_std.json",
        "phase01/p01_dsnn_fused-early_dep_flat_raw.json",
        "phase01/p01_dsnn_fused-early_dep_flat_std.json",
        "phase01/p01_dsnn_fused-early_dep_nested_raw.json",
        "phase01/p01_dsnn_fused-early_dep_nested_std.json",
        "phase01/p01_dsnn_fused-early_indep_flat_raw.json",
        "phase01/p01_dsnn_fused-early_indep_flat_std.json",
        "phase01/p01_dsnn_fused-early_indep_nested_raw.json",
        "phase01/p01_dsnn_fused-early_indep_nested_std.json",
        "phase01/p01_dsnn_fused-late_dep_flat_raw.json",
        "phase01/p01_dsnn_fused-late_dep_flat_std.json",
        "phase01/p01_dsnn_fused-late_dep_nested_raw.json",
        "phase01/p01_dsnn_fused-late_dep_nested_std.json",
        "phase01/p01_dsnn_fused-late_indep_flat_raw.json",
        "phase01/p01_dsnn_fused-late_indep_flat_std.json",
        "phase01/p01_dsnn_fused-late_indep_nested_raw.json",
        "phase01/p01_dsnn_fused-late_indep_nested_std.json",
        "phase01/p01_dsnn_voice_dep_flat_raw.json",
        "phase01/p01_dsnn_voice_dep_flat_std.json",
        "phase01/p01_dsnn_voice_dep_nested_raw.json",
        "phase01/p01_dsnn_voice_dep_nested_std.json",
        "phase01/p01_dsnn_voice_indep_flat_raw.json",
        "phase01/p01_dsnn_voice_indep_flat_std.json",
        "phase01/p01_dsnn_voice_indep_nested_raw.json",
        "phase01/p01_dsnn_voice_indep_nested_std.json",
    };
    return profiles;
}

fs::path profiles_dir()
{
    fs::path here = fs::path(__FILE__).parent_path();
    return here.parent_path() / "profiles";
}

E05Config load(const std::string& name)
{
    const fs::path path = profiles_dir() / name;
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    return E05Config::from_json(j);
}

} // namespace

class E05ProfileAuditTest : public ::testing::TestWithParam<std::string>
{
};

TEST_P(E05ProfileAuditTest, ParsesCleanly)
{
    auto cfg = load(GetParam());
    cfg.validate();
    EXPECT_FALSE(cfg.experiment.run_tag.empty());
    EXPECT_NE(cfg.experiment.seed, 0u);
    EXPECT_GT(cfg.experiment.repeats, 0);
    EXPECT_FALSE(cfg.dataset.root.empty());
}

TEST_P(E05ProfileAuditTest, DatasetModalityValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.dataset.modality == "voice" || cfg.dataset.modality == "eeg" ||
                cfg.dataset.modality == "fused")
        << "unexpected modality: " << cfg.dataset.modality;
}

TEST_P(E05ProfileAuditTest, StrategyValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.feature_extraction.strategy == "handcrafted" ||
                cfg.feature_extraction.strategy == "autoencoder")
        << "unexpected strategy: " << cfg.feature_extraction.strategy;
}

TEST_P(E05ProfileAuditTest, ClassifierValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.classifier.type == "rnn" || cfg.classifier.type == "dsnn")
        << "unexpected classifier type: " << cfg.classifier.type;
    // layer_spec is only required when the classifier runs (Phase 01).
    if (cfg.classifier.enabled)
    {
        EXPECT_FALSE(cfg.classifier.layer_spec.empty());
    }
    EXPECT_TRUE(
        cfg.classifier.text_mode == "dependent" || cfg.classifier.text_mode == "independent")
        << "unexpected text_mode: " << cfg.classifier.text_mode;
}

TEST_P(E05ProfileAuditTest, TrainingParamsPositive)
{
    auto cfg = load(GetParam());
    EXPECT_GT(cfg.training.epochs, 0);
    // Check the lr the run will ACTUALLY use: resolved from the optimizer's reference
    // default when the profile omits it, so this holds for every profile either way.
    EXPECT_GT(cfg.training.effective_learning_rate(), 0.0f);

    // The resolved lr must be right FOR THIS PROFILE'S OPTIMIZER. When the profile declares
    // one, that value must survive verbatim (regression guard: making learning_rate optional
    // must not have changed any already-published run). When it omits one, it must inherit
    // its own optimizer's reference default -- never another optimizer's.
    if (cfg.training.learning_rate.has_value())
    {
        EXPECT_FLOAT_EQ(cfg.training.effective_learning_rate(), *cfg.training.learning_rate);
    }
    else
    {
        EXPECT_FLOAT_EQ(cfg.training.effective_learning_rate(),
            nn::optimizers::reference_learning_rate(cfg.training.optimizer_type));
    }
    EXPECT_GT(cfg.training.samples_per_batch, 0);
    EXPECT_GE(cfg.training.early_stop_patience, 0);
    EXPECT_GE(cfg.training.k_folds, 2);
}

// PyTorch/snnTorch is this project's correctness reference, so the defaults must BE the
// reference: exact activations on, gradient clipping off. A profile can trade either away,
// but only explicitly.
TEST(E05Fidelity, DefaultsMatchTheReferenceAndAreOverridable)
{
    E05Config c;
    c.experiment.run_tag = "t";
    c.dataset.root = "/tmp/x";
    c.classifier.enabled = false;

    // Exact sigmoid/tanh by default -> our LSTM equals torch.nn.LSTM. The fast softsign
    // gates are ~2x quicker but provably not torch (|tanh - tanh_fast| reaches 0.306).
    EXPECT_TRUE(c.numerics.exact_activations);

    // No clipping by default, like PyTorch. This is the ONLY clipping knob: MSELoss/MAELoss
    // used to clip themselves at norm 1.0 unconditionally, overriding this very field.
    EXPECT_FLOAT_EQ(c.training.gradient_clip_norm, 0.0f);
    EXPECT_NO_THROW(c.validate());

    // Both are overridable...
    c.numerics.exact_activations = false;
    c.training.gradient_clip_norm = 0.5f;
    EXPECT_NO_THROW(c.validate());

    // ...but a negative clip is meaningless and must be rejected rather than silently used.
    c.training.gradient_clip_norm = -1.0f;
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

// fixme.md D6: bark/mel are cochlear (hearing) scales with no physiological basis for EEG,
// and group_by_scale()'s Nyquist normalization made them provably degenerate to lfcc there
// (16 sub-bands -> 16 distinct bins -> one group each == lfcc). They are rejected for
// modality=eeg so the redundant axis cannot be reintroduced.
TEST(E05EegScaleAxis, BarkAndMelAreRejectedForEeg)
{
    auto make = [](const std::string& modality, const std::string& scale)
    {
        E05Config c;
        c.experiment.run_tag = "t";
        c.dataset.root = "/tmp/x";
        c.dataset.modality = modality;
        c.feature_extraction.strategy = "handcrafted";
        c.feature_extraction.handcrafted.scale = scale;
        c.classifier.enabled = false;
        return c;
    };

    // EEG: only lfcc (linear) is admissible.
    EXPECT_NO_THROW(make("eeg", "lfcc").validate());
    EXPECT_THROW(make("eeg", "bark").validate(), std::invalid_argument);
    EXPECT_THROW(make("eeg", "mel").validate(), std::invalid_argument);

    // Voice: all three remain valid — there the scales genuinely differ (bark merges the
    // 16 sub-bands into 9 groups, mel into 11), so the axis carries real information.
    for (const auto* s : {"lfcc", "bark", "mel"}) EXPECT_NO_THROW(make("voice", s).validate());

    // Fused is deliberately NOT restricted: its voice half legitimately uses bark/mel.
    for (const auto* s : {"lfcc", "bark", "mel"}) EXPECT_NO_THROW(make("fused", s).validate());
}

// Each optimizer must resolve to ITS OWN reference lr, not another's. Their usable rates
// differ by ~10x (Lion steps +/-lr on every coordinate), so a profile that names an
// optimizer but inherits a foreign lr would measure the learning rate, not the optimizer.
TEST(E05OptimizerLearningRate, EachOptimizerResolvesToItsOwnReferenceLr)
{
    E05Config cfg;
    const std::vector<std::pair<std::string, float>> expected = {
        {"adam", 1e-3f},
        {"sgd", 1e-2f},
        {"lion", 1e-4f},
        {"schedule-free-adamw", 2.5e-3f},
    };
    for (const auto& [opt, lr] : expected)
    {
        cfg.training.optimizer_type = opt;
        cfg.training.learning_rate = std::nullopt; // profile omits it
        EXPECT_FLOAT_EQ(cfg.training.effective_learning_rate(), lr) << "optimizer: " << opt;
    }

    // No two of them share a default -- otherwise the whole mechanism would be pointless.
    std::set<float> distinct;
    for (const auto& [opt, lr] : expected) distinct.insert(lr);
    EXPECT_EQ(distinct.size(), expected.size());
}

// An explicit profile lr still wins: sweeping lr per optimizer must remain possible.
TEST(E05OptimizerLearningRate, ExplicitProfileValueOverridesTheDefault)
{
    E05Config cfg;
    cfg.training.optimizer_type = "lion";
    EXPECT_FLOAT_EQ(cfg.training.effective_learning_rate(), 1e-4f); // inherited
    cfg.training.learning_rate = 5e-5f;
    EXPECT_FLOAT_EQ(cfg.training.effective_learning_rate(), 5e-5f); // explicit wins
}

// Every profile must name an optimizer the OptimizerFactory can actually build; a typo
// here would otherwise only surface as a runtime throw part-way into a long experiment.
TEST_P(E05ProfileAuditTest, OptimizerTypeIsSupported)
{
    auto cfg = load(GetParam());
    const auto& t = cfg.training.optimizer_type;
    const bool known = (t == "adam" || t == "sgd" || t == "lion" || t == "schedule-free-adamw");
    EXPECT_TRUE(known) << "unsupported training.optimizer_type: " << t;
    EXPECT_GE(cfg.training.optimizer_momentum, 0.0f);
    EXPECT_LT(cfg.training.optimizer_momentum, 1.0f);
}

TEST_P(E05ProfileAuditTest, RegularizationParamsValid)
{
    auto cfg = load(GetParam());
    EXPECT_GE(cfg.training.weight_decay, 0.0f);
    EXPECT_GE(cfg.training.firing_rate_reg_lambda, 0.0f);
    EXPECT_GE(cfg.training.firing_rate_min, 0.0f);
    EXPECT_LE(cfg.training.firing_rate_max, 1.0f);
    EXPECT_LE(cfg.training.firing_rate_min, cfg.training.firing_rate_max);
}

TEST_P(E05ProfileAuditTest, SeedDeterministicFalseForArticleProfiles)
{
    const std::string name = GetParam();
    if (name == "debug.json") return; // debug profile uses deterministic mode
    auto cfg = load(name);
    EXPECT_FALSE(cfg.experiment.seed_deterministic)
        << name << " should have seed_deterministic=false for reproducible sweeps";
}

// ─── classifier.enabled phase gate + handcrafted axis validation ─────────────
// These are non-parametric: they exercise E05Config::from_json/validate on
// synthetic configs rather than the shipped profiles.

namespace
{
// Minimal config that validates: handcrafted, phase-01 style (classifier runs).
nlohmann::json base_config()
{
    return nlohmann::json{
        {"experiment", {{"run_tag", "t"}, {"seed", 42}}},
        {"dataset", {{"root", "/x"}, {"modality", "voice"}}},
        {"feature_extraction",
            {{"strategy", "handcrafted"},
                {"handcrafted",
                    {{"scale", "lfcc"}, {"wavelet", "daub4"}, {"descriptors", {"energy"}}}}}},
        {"classifier",
            {{"type", "dsnn"},
                {"layer_spec", {"linear:64:relu", "residual:1", "linear:N_speakers:identity"}},
                {"text_mode", "independent"},
                {"enabled", true}}},
    };
}
} // namespace

TEST(E05ConfigGate, Phase00OmitsLayerSpec)
{
    // classifier.enabled=false (Phase 00): layer_spec not required.
    auto j = base_config();
    j["classifier"] = nlohmann::json{{"enabled", false}};
    auto cfg = E05Config::from_json(j);
    EXPECT_FALSE(cfg.classifier.enabled);
    EXPECT_NO_THROW(cfg.validate());
}

TEST(E05ConfigGate, Phase01RequiresLayerSpec)
{
    // classifier.enabled defaults true (Phase 01): missing layer_spec must throw.
    auto j = base_config();
    j["classifier"] = nlohmann::json{{"type", "dsnn"}, {"text_mode", "independent"}};
    auto cfg = E05Config::from_json(j);
    EXPECT_TRUE(cfg.classifier.enabled); // default
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(E05ConfigGate, FusionModeValidated)
{
    auto j = base_config();
    j["dataset"] = {{"root", "/x"}, {"modality", "fused"}, {"fusion_mode", "early"}};
    EXPECT_NO_THROW(E05Config::from_json(j).validate());
    j["dataset"]["fusion_mode"] = "bogus";
    EXPECT_THROW(E05Config::from_json(j).validate(), std::invalid_argument);
}

TEST(E05ConfigGate, WaveletValidated)
{
    auto j = base_config();
    j["feature_extraction"]["handcrafted"]["wavelet"] = "daub46";
    EXPECT_NO_THROW(E05Config::from_json(j).validate());
    j["feature_extraction"]["handcrafted"]["wavelet"] = "daub48"; // no traits
    EXPECT_THROW(E05Config::from_json(j).validate(), std::invalid_argument);
    j["feature_extraction"]["handcrafted"]["wavelet"] = "not-a-wavelet";
    EXPECT_THROW(E05Config::from_json(j).validate(), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(AllProfiles,
    E05ProfileAuditTest,
    ::testing::ValuesIn(all_profiles()),
    [](const ::testing::TestParamInfo<std::string>& param_info)
    {
        std::string name = param_info.param;
        // Replace non-alphanumeric chars with underscore for GTest naming.
        for (char& c : name)
            if (!std::isalnum(c)) c = '_';
        return name;
    });
