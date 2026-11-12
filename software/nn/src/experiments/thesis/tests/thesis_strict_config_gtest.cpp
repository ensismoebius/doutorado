// The profile contract of ThesisConfig: no key has a default. A key the run reads must be
// declared, a key it does not read must be absent, an unknown key is refused, and a value
// of the wrong JSON type is refused. Every refusal names the key.

#include <gtest/gtest.h>

#include <string>

#include "../lib/include/ThesisConfig.hpp"
#include "nlohmann/json.hpp"

using thesis::ThesisConfig;

namespace
{
// A complete Phase-00 handcrafted EEG profile: exactly the keys that run reads.
nlohmann::json handcrafted_eeg()
{
    return nlohmann::json::parse(R"({
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 1},
        "dataset": {"root": "/x", "results_dir": "results/thesis", "modality": "eeg", "eeg_condition": "pronounced",
                    "max_samples": 0},
        "feature_extraction": {
            "strategy": "handcrafted",
            "handcrafted": {"transform": "dtwpt", "scale": "lfcc", "descriptors": ["energy"],
                            "dtwpt_level": 4, "wavelet": "daub4", "cepstral": false}
        },
        "paraconsistent": {"enabled": true, "speaker_subsets": false},
        "classifier": {"enabled": false}
    })");
}

// A complete Phase-00 SNN-AE (poisson) EEG profile.
nlohmann::json snn_ae_eeg()
{
    return nlohmann::json::parse(R"({
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 1},
        "dataset": {"root": "/x", "results_dir": "results/thesis", "modality": "eeg", "eeg_condition": "pronounced",
                    "max_samples": 0},
        "feature_extraction": {
            "strategy": "autoencoder",
            "autoencoder": {
                "model": "snn-ae",
                "encoder_layer_spec": ["linear:64:leaky", "linear:16:identity"],
                "decoder_layer_spec": ["linear:64:leaky", "linear:output:identity"],
                "ae_loss_type": "spikecount",
                "encoding": "poisson", "time_steps": 16, "voltage_threshold": 0.2,
                "firing_rate_reg_lambda": 0.5, "firing_rate_min": 0.1, "firing_rate_max": 0.8
            }
        },
        "paraconsistent": {"enabled": true, "speaker_subsets": false},
        "classifier": {"enabled": false},
        "training": {"epochs": 5, "learning_rate": 0.001, "optimizer_type": "adam",
                     "gradient_clip_norm": 0.0, "samples_per_batch": 8}
    })");
}

// The message of the exception from_json + validate throw for `j` ("" if none).
std::string refusal(const nlohmann::json& j)
{
    try
    {
        ThesisConfig::from_json(j).validate();
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    return "";
}

void expect_refused(const nlohmann::json& j, const std::string& key, const std::string& why)
{
    const std::string msg = refusal(j);
    EXPECT_NE(msg.find(key), std::string::npos) << "refusal must name " << key << "; got: " << msg;
    EXPECT_NE(msg.find(why), std::string::npos) << "refusal must say '" << why << "'; got: " << msg;
}
} // namespace

TEST(ThesisStrictConfig, CompleteProfilesAreAccepted)
{
    EXPECT_EQ(refusal(handcrafted_eeg()), "");
    EXPECT_EQ(refusal(snn_ae_eeg()), "");
}

TEST(ThesisStrictConfig, MissingKeyIsRefusedNamingIt)
{
    auto j = handcrafted_eeg();
    j["dataset"].erase("max_samples");
    expect_refused(j, "dataset.max_samples", "does not declare");

    j = snn_ae_eeg();
    j["feature_extraction"]["autoencoder"].erase("model"); // used to become "lstm-ae"
    expect_refused(j, "feature_extraction.autoencoder.model", "does not declare");

    j = snn_ae_eeg();
    j["training"].erase("learning_rate"); // used to resolve to the optimizer's default
    expect_refused(j, "training.learning_rate", "does not declare");
}

TEST(ThesisStrictConfig, KeyTheRunDoesNotReadIsRefused)
{
    auto j = handcrafted_eeg();
    j["training"] = {{"epochs", 50}}; // a handcrafted Phase 00 run trains nothing
    expect_refused(j, "training.epochs", "never reads");

    j = handcrafted_eeg();
    j["dataset"]["fusion_mode"] = "late"; // fused only
    expect_refused(j, "dataset.fusion_mode", "never reads");

    j = snn_ae_eeg();
    j["training"]["optimizer_momentum"] = 0.9; // sgd only
    expect_refused(j, "training.optimizer_momentum", "never reads");

    j = snn_ae_eeg();
    j["experiment"]["seed_deterministic"] = false; // repeats > 1 only
    expect_refused(j, "experiment.seed_deterministic", "never reads");
}

TEST(ThesisStrictConfig, KeysFollowTheModelAndEncoding)
{
    // direct code: one analog frame, no T.
    auto j = snn_ae_eeg();
    auto& ae = j["feature_extraction"]["autoencoder"];
    ae["encoding"] = "direct";
    ae["ae_loss_type"] = "mse";
    expect_refused(j, "feature_extraction.autoencoder.time_steps", "never reads");
    ae.erase("time_steps");
    EXPECT_EQ(refusal(j), "");

    // regularizer off: no band.
    ae["firing_rate_reg_lambda"] = 0.0;
    expect_refused(j, "feature_extraction.autoencoder.firing_rate_min", "never reads");
    ae.erase("firing_rate_min");
    ae.erase("firing_rate_max");
    EXPECT_EQ(refusal(j), "");

    // ANN-AE: analog by construction, none of the SNN keys.
    ae["model"] = "ann-ae";
    expect_refused(j, "feature_extraction.autoencoder.encoding", "never reads");
}

TEST(ThesisStrictConfig, SgdNeedsMomentum)
{
    auto j = snn_ae_eeg();
    j["training"]["optimizer_type"] = "sgd";
    expect_refused(j, "training.optimizer_momentum", "does not declare");
    j["training"]["optimizer_momentum"] = 0.9;
    EXPECT_EQ(refusal(j), "");
}

TEST(ThesisStrictConfig, UnknownKeyIsRefused)
{
    auto j = handcrafted_eeg();
    j["numerics"] = {{"exact_activations", true}}; // removed: nothing read it
    expect_refused(j, "numerics", "unknown key");

    j = snn_ae_eeg();
    j["training"]["optimiser_type"] = "adam"; // typo
    expect_refused(j, "training.optimiser_type", "unknown key");
}

TEST(ThesisStrictConfig, WrongJsonTypeIsRefused)
{
    auto j = snn_ae_eeg();
    j["training"]["epochs"] = 5.5; // would have truncated to 5
    expect_refused(j, "training.epochs", "must be an integer");

    j = snn_ae_eeg();
    j["experiment"]["seed"] = -1; // would have wrapped to 4294967295
    expect_refused(j, "experiment.seed", "in range");

    j = handcrafted_eeg();
    j["classifier"]["enabled"] = 0;
    expect_refused(j, "classifier.enabled", "true or false");
}

TEST(ThesisStrictConfig, PcaProfile)
{
    auto j = handcrafted_eeg();
    j["feature_extraction"] = {{"strategy", "pca"}, {"pca", {{"n_components", 16}}}};
    EXPECT_EQ(refusal(j), "");

    j["feature_extraction"]["pca"]["n_components"] = 0;
    expect_refused(j, "n_components", "[1, 256]");

    j["feature_extraction"]["pca"]["n_components"] = 16;
    j["feature_extraction"]["handcrafted"] = handcrafted_eeg()["feature_extraction"]["handcrafted"];
    expect_refused(j, "feature_extraction.handcrafted", "never reads");
}

TEST(ThesisStrictConfig, DeclaredJsonIsTheProfile)
{
    // What the summary records is exactly what the profile declared: the same keys
    // (values compared by key; floats round-trip through float, 0.2 -> 0.2f).
    const auto j = snn_ae_eeg();
    const auto declared = ThesisConfig::from_json(j).declared_json().flatten();
    const auto profile = j.flatten();
    ASSERT_EQ(declared.size(), profile.size());
    for (const auto& [key, value] : profile.items())
    {
        ASSERT_TRUE(declared.contains(key)) << key;
        if (value.is_number_float())
            EXPECT_NEAR(declared[key].get<double>(), value.get<double>(), 1e-6) << key;
        else
            EXPECT_EQ(declared[key], value) << key;
    }
}
