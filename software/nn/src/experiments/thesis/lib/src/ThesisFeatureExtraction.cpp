#include "ThesisFeatureExtraction.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ThesisFeatureExtractionInternal.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "progress/ProgressManager.hpp"
#include "training/ProgressCallback.hpp"

using nn::models::autoencoder::ProtocolAutoencoder;
using nn::models::autoencoder::ProtocolSpikingAutoencoder;

namespace thesis
{

// Fail loudly when a spike loss contributed nothing. A run whose every gradient was
// zero must not be reported as a result — that is the silent-no-learning mode.
void assert_gradients_were_live(long backward_calls,
    long zero_grad_calls,
    const std::string& loss_token,
    const std::string& encoding,
    float firing_rate_reg_lambda)
{
    if (backward_calls == 0 || zero_grad_calls < backward_calls) return;

    throw std::runtime_error(
        "ThesisFeatureExtraction: ae_loss_type=" + loss_token + " (encoding=" + encoding +
        ") produced an ALL-ZERO gradient on every one of the " + std::to_string(backward_calls) +
        " training batches — the autoencoder trained on nothing and its features are "
        "meaningless. Cause: no output unit ever crossed the spike threshold, so the "
        "straight-through estimator had no spike time to attach a gradient to (the "
        "no-spike deadlock in .wiki/Concepts/Spike-Encoding.md). Fixes, in order of "
        "preference: (1) raise autoencoder.firing_rate_reg_lambda (currently " +
        std::to_string(firing_rate_reg_lambda) +
        ") so the encoder gets rate-derived gradient independent of spike position; "
        "(2) lower autoencoder.voltage_threshold so units fire at all; (3) enable tdBN "
        "upstream. Refusing to emit features from an untrained model.");
}

namespace
{

// DTWPT handcrafted-descriptor extraction, in parallel across samples.
FeatureSet extract_handcrafted_feature_set(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& cfg,
    const SignalGetter& get_signal,
    double sample_rate,
    const std::string& label_suffix)
{
    FeatureSet fs;
    fs.label = "handcrafted-" + cfg.handcrafted.scale + label_suffix;
    fs.vectors.reserve(view.samples.size());

    const auto n_samples = static_cast<long>(view.samples.size());
    uint32_t feat_bar = nn::progress::ProgressManager::instance().create_bar(
        "Feature extraction", static_cast<float>(n_samples));
    nn::progress::ProgressManager::instance().set_description(feat_bar,
        "DTWPT | scale=" + cfg.handcrafted.scale +
            "  descriptors=" + std::to_string(cfg.handcrafted.descriptors.size()));

    fs.vectors.resize(static_cast<size_t>(n_samples));
    long feat_done = 0;

#pragma omp parallel for schedule(dynamic, 4) shared(feat_done)
    for (long i = 0; i < n_samples; ++i)
    {
        const auto& sample = view.samples[static_cast<size_t>(i)];
        std::vector<double> sig = get_signal(sample);

        // Zero-pad to next power of two for DTWPT.
        size_t n = 1;
        while (n < sig.size()) n <<= 1;
        sig.resize(n, 0.0);

        fs.vectors[static_cast<size_t>(i)] = extract_handcrafted(sig, cfg.handcrafted, sample_rate);

        long done = 0;
#pragma omp atomic capture
        done = ++feat_done;
        nn::progress::ProgressManager::instance().update_bar(feat_bar, static_cast<float>(done));
    }
    nn::progress::ProgressManager::instance().complete_bar(feat_bar);
    return fs;
}

// Collects the raw per-sample signal (via get_signal) for every sample in `view`, and the
// longest one seen — used to size the autoencoder input and (for the LSTM path) the frame
// length. Throws if every signal came back empty.
std::pair<std::vector<std::vector<double>>, size_t> collect_raw_signals(
    const ThesisDatasetView& view, const SignalGetter& get_signal)
{
    std::vector<std::vector<double>> raw_signals;
    raw_signals.reserve(view.samples.size());
    size_t max_len = 0;
    for (const auto& sample : view.samples)
    {
        std::vector<double> sig = get_signal(sample);
        max_len = std::max(max_len, sig.size());
        raw_signals.push_back(std::move(sig));
    }
    if (max_len == 0)
        throw std::runtime_error("ThesisFeatureExtraction: no valid raw signals for autoencoder");

    return {std::move(raw_signals), max_len};
}

// Wraps `on_model_trained` (tagged with `part_tag`) into the plain state-dict callback that
// run_protocol_ae<T> expects, or returns nullptr when there is no callback to wrap.
std::function<void(const std::map<std::string, nn::Tensor>&)> make_ae_snapshot_callback(
    const std::string& part_tag, const ModelSnapshotFn& on_model_trained)
{
    if (!on_model_trained) return nullptr;
    return [&](const std::map<std::string, nn::Tensor>& sd) { on_model_trained(part_tag, sd); };
}

// Spiking AE trains batched: row-vector samples stack into a 2D (B, F) batch
// (Trainer::create_batch), and Lif/LifIntegrator resize their membrane state to the batch
// shape. Same batch size as ANN-AE.
FeatureSet extract_snn_ae_feature_set(const std::vector<std::vector<double>>& raw_signals,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& label_suffix,
    std::uint32_t seed,
    const std::string& part_tag,
    const ModelSnapshotFn& on_model_trained)
{
    FeatureSet fs;
    fs.label = "autoencoder-snn" + label_suffix;
    fs.vectors = run_protocol_ae<ProtocolSpikingAutoencoder>(raw_signals,
        cfg.autoencoder,
        training,
        label_suffix,
        "SNN-AE",
        training.samples_per_batch,
        cfg.autoencoder.encoding,
        cfg.autoencoder.time_steps,
        seed,
        cfg.autoencoder.voltage_threshold,
        cfg.autoencoder.ae_loss_type,
        /*spiking=*/true,
        make_ae_snapshot_callback(part_tag, on_model_trained));
    return fs;
}

// ANN-AE is non-spiking: always analog ("direct"), single frame.
FeatureSet extract_ann_ae_feature_set(const std::vector<std::vector<double>>& raw_signals,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& label_suffix,
    std::uint32_t seed,
    const std::string& part_tag,
    const ModelSnapshotFn& on_model_trained)
{
    FeatureSet fs;
    fs.label = "autoencoder-ann" + label_suffix;
    fs.vectors = run_protocol_ae<ProtocolAutoencoder>(raw_signals,
        cfg.autoencoder,
        training,
        label_suffix,
        "ANN-AE",
        training.samples_per_batch,
        "direct",
        1,
        seed,
        1.0f,
        cfg.autoencoder.ae_loss_type,
        /*spiking=*/false,
        make_ae_snapshot_callback(part_tag, on_model_trained));
    return fs;
}

// Sequence AE on windowed frames (Guayaquil-paper extractor).
FeatureSet extract_lstm_ae_feature_set(const std::vector<std::vector<double>>& raw_signals,
    size_t max_len,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& label_suffix)
{
    const int frame_len =
        std::max<int>(1, static_cast<int>((max_len + kAeMaxFrames - 1) / kAeMaxFrames));
    const int T_frames = kAeMaxFrames;
    const size_t padded = static_cast<size_t>(T_frames) * static_cast<size_t>(frame_len);

    std::vector<nn::Tensor> train_samples;
    train_samples.reserve(raw_signals.size());
    for (auto sig : raw_signals)
    {
        sig.resize(padded, 0.0);
        train_samples.push_back(vec_to_frame_tensor(sig, T_frames, frame_len));
    }

    nn::models::lstm::LSTMAutoencoderConfig ae_cfg;
    ae_cfg.input_size = frame_len;
    ae_cfg.seq_len = T_frames;
    ae_cfg.hidden_size = first_encoder_dim(cfg.autoencoder.encoder_layer_spec, 64);
    ae_cfg.latent_size = last_encoder_dim(cfg.autoencoder.encoder_layer_spec, 16);
    ae_cfg.num_layers =
        std::max<int>(1, static_cast<int>(cfg.autoencoder.encoder_layer_spec.size() / 2));

    nn::models::lstm::LSTMAutoencoder model(ae_cfg);

    nn::training::TrainerConfig trainer_cfg;
    trainer_cfg.epochs = training.epochs;
    trainer_cfg.learning_rate = training.effective_learning_rate();
    trainer_cfg.optimizer_type = training.optimizer_type;
    trainer_cfg.optimizer_momentum = training.optimizer_momentum;
    trainer_cfg.grad_clip_norm = training.gradient_clip_norm;
    trainer_cfg.batch_size = training.samples_per_batch;

    nn::training::Trainer<nn::models::lstm::LSTMAutoencoder> trainer(model, trainer_cfg);
    // Same enrichment as the SNN/ANN AE path: label the bar with the model + loss so
    // the metadata line matches the Guayaquil TUI look.
    auto ae_cb =
        std::make_shared<nn::training::ProgressCallback>("Autoencoder training" + label_suffix);
    ae_cb->set_metadata("LSTM-AE", 0, 1, "MSE");
    trainer.add_callback(ae_cb);
    (void) trainer.fit_autoencoder(train_samples);

    FeatureSet fs;
    fs.label = "autoencoder-lstm" + label_suffix;
    fs.vectors.reserve(train_samples.size());
    for (const auto& sample : train_samples)
        fs.vectors.push_back(tensor_to_vec(model.encode(sample, false)));
    return fs;
}

// Dispatches to the right autoencoder family by cfg.autoencoder.model.
FeatureSet extract_autoencoder_feature_set(const std::vector<std::vector<double>>& raw_signals,
    size_t max_len,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& label_suffix,
    std::uint32_t seed,
    const std::string& part_tag,
    const ModelSnapshotFn& on_model_trained)
{
    const std::string& ae_model = cfg.autoencoder.model;

    if (ae_model == "snn-ae")
        return extract_snn_ae_feature_set(
            raw_signals, cfg, training, label_suffix, seed, part_tag, on_model_trained);

    if (ae_model == "ann-ae")
        return extract_ann_ae_feature_set(
            raw_signals, cfg, training, label_suffix, seed, part_tag, on_model_trained);

    // "lstm-ae"
    return extract_lstm_ae_feature_set(raw_signals, max_len, cfg, training, label_suffix);
}

// Runs the configured strategy (handcrafted or autoencoder) against whatever
// single signal get_signal() returns per sample. Used directly for "voice"
// and "eeg" modality, and reused twice by late fusion (once per signal) and
// once by early fusion (on the pre-concatenated signal). label_suffix tags
// the resulting FeatureSet so voice-part/EEG-part/fused variants stay
// distinguishable in results output.
auto extract_features_core(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const SignalGetter& get_signal,
    double sample_rate,
    const std::string& label_suffix,
    std::uint32_t seed,
    const std::string& part_tag = "",
    const ModelSnapshotFn& on_model_trained = nullptr) -> std::vector<FeatureSet>
{
    std::vector<FeatureSet> result;

    if (cfg.strategy == "handcrafted")
    {
        result.push_back(
            extract_handcrafted_feature_set(view, cfg, get_signal, sample_rate, label_suffix));
    }
    else if (cfg.strategy == "autoencoder")
    {
        auto [raw_signals, max_len] = collect_raw_signals(view, get_signal);
        result.push_back(extract_autoencoder_feature_set(
            raw_signals, max_len, cfg, training, label_suffix, seed, part_tag, on_model_trained));
    }
    else
    {
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown strategy \"" + cfg.strategy + "\"");
    }

    return result;
}

} // namespace

auto extract_features(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& modality,
    const std::string& fusion_mode,
    std::uint32_t seed,
    const ModelSnapshotFn& on_model_trained) -> std::vector<FeatureSet>
{
    if (modality == "voice")
        return extract_features_core(
            view, cfg, training, voice_signal, kVoiceSampleRate, "", seed, "", on_model_trained);

    if (modality == "eeg")
        return extract_features_core(
            view, cfg, training, eeg_signal, kEegSampleRate, "", seed, "", on_model_trained);

    if (modality != "fused")
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown modality \"" + modality + "\"");

    if (fusion_mode == "early")
    {
        return extract_features_core(view,
            cfg,
            training,
            fused_early_signal,
            kVoiceSampleRate,
            "-fused-early",
            seed,
            "",
            on_model_trained);
    }

    if (fusion_mode != "late")
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown fusion_mode \"" + fusion_mode + "\"");

    // Late fusion: extract independently per signal, then concatenate the
    // resulting feature vectors sample-by-sample (audit C12). Two INDEPENDENT models
    // train here (voice AE + EEG AE) — tag each snapshot so a caller merging them
    // (e.g. paraconsistentGA's weight-snapshot capture) can tell them apart.
    auto voice_sets = extract_features_core(
        view, cfg, training, voice_signal, kVoiceSampleRate, "", seed, "voice", on_model_trained);
    auto eeg_sets = extract_features_core(
        view, cfg, training, eeg_signal, kEegSampleRate, "", seed, "eeg", on_model_trained);

    if (voice_sets.size() != eeg_sets.size())
        throw std::runtime_error(
            "ThesisFeatureExtraction: late fusion produced mismatched FeatureSet counts");

    std::vector<FeatureSet> result;
    result.reserve(voice_sets.size());
    for (size_t k = 0; k < voice_sets.size(); ++k)
    {
        auto& vfs = voice_sets[k];
        auto& efs = eeg_sets[k];
        if (vfs.vectors.size() != efs.vectors.size())
            throw std::runtime_error(
                "ThesisFeatureExtraction: late fusion sample-count mismatch between voice and EEG");

        FeatureSet fused;
        fused.label = vfs.label + "-fused-late";
        fused.vectors.resize(vfs.vectors.size());
        for (size_t i = 0; i < vfs.vectors.size(); ++i)
        {
            fused.vectors[i] = vfs.vectors[i];
            fused.vectors[i].insert(
                fused.vectors[i].end(), efs.vectors[i].begin(), efs.vectors[i].end());
        }
        result.push_back(std::move(fused));
    }
    return result;
}

} // namespace thesis
