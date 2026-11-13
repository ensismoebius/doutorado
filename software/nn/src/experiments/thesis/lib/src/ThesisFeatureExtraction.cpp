#include "ThesisFeatureExtraction.hpp"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "ThesisFeatureExtractionInternal.hpp"
#include "ThesisPca.hpp"
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
            "  descriptors=" + std::to_string(cfg.handcrafted.descriptors->size()));

    fs.vectors.resize(static_cast<size_t>(n_samples));
    long feat_done = 0;
    // An exception may not leave an OpenMP region (it would std::terminate the run);
    // the first one is kept, with the sample it came from, and rethrown after the loop.
    std::exception_ptr first_error;
    long first_error_sample = -1;

#pragma omp parallel for schedule(dynamic, 4) shared(feat_done)
    for (long i = 0; i < n_samples; ++i)
    {
        try
        {
            const auto& sample = view.samples[static_cast<size_t>(i)];
            std::vector<double> sig = get_signal(sample);
            if (sig.empty()) throw std::runtime_error("ThesisFeatureExtraction: empty signal");

            // Zero-pad to next power of two for DTWPT.
            size_t n = 1;
            while (n < sig.size()) n <<= 1;
            sig.resize(n, 0.0);

            fs.vectors[static_cast<size_t>(i)] =
                extract_handcrafted(sig, cfg.handcrafted, sample_rate);
        }
        catch (...)
        {
#pragma omp critical(thesis_handcrafted_error)
            if (!first_error || i < first_error_sample)
            {
                first_error = std::current_exception();
                first_error_sample = i;
            }
        }

        long done = 0;
#pragma omp atomic capture
        done = ++feat_done;
        nn::progress::ProgressManager::instance().update_bar(feat_bar, static_cast<float>(done));
    }
    if (first_error)
    {
        const auto& sample = view.samples[static_cast<size_t>(first_error_sample)];
        try
        {
            std::rethrow_exception(first_error);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("ThesisFeatureExtraction: " + fs.label + ", sample " +
                                     std::to_string(first_error_sample) + " (subject " +
                                     std::to_string(sample.subject_id) + ", stimulus " +
                                     std::to_string(sample.stimulus) + "): " + e.what());
        }
    }
    nn::progress::ProgressManager::instance().complete_bar(feat_bar);
    return fs;
}

// Collects the raw per-sample signal (via get_signal) for every sample in `view`. Throws on
// an empty signal or an empty view.
std::vector<std::vector<double>> collect_raw_signals(
    const ThesisDatasetView& view, const SignalGetter& get_signal)
{
    std::vector<std::vector<double>> raw_signals;
    raw_signals.reserve(view.samples.size());
    for (const auto& sample : view.samples)
    {
        std::vector<double> sig = get_signal(sample);
        // One empty signal used to pass as long as another was not: it then went through
        // the input transform and normalisation as if it held data.
        if (sig.empty())
            throw std::runtime_error("ThesisFeatureExtraction: subject " +
                                     std::to_string(sample.subject_id) + ", stimulus " +
                                     std::to_string(sample.stimulus) + " has an empty signal");
        raw_signals.push_back(std::move(sig));
    }
    if (raw_signals.empty())
        throw std::runtime_error("ThesisFeatureExtraction: no samples for the autoencoder");

    return raw_signals;
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
    // The direct code presents ONE analog frame: there is no T to read (the profile is
    // refused if it declares one), so the frame count is 1 by definition, not by default.
    const std::string& encoding = cfg.autoencoder.encoding;
    const int frames = encoding == "direct" ? 1 : cfg.autoencoder.time_steps.get();

    FeatureSet fs;
    fs.label = "autoencoder-snn" + label_suffix;
    fs.vectors = run_protocol_ae<ProtocolSpikingAutoencoder>(raw_signals,
        cfg.autoencoder,
        training,
        label_suffix,
        "SNN-AE",
        training.samples_per_batch,
        encoding,
        frames,
        seed,
        cfg.autoencoder.voltage_threshold,
        cfg.autoencoder.ae_loss_type,
        /*spiking=*/true,
        make_ae_snapshot_callback(part_tag, on_model_trained));
    return fs;
}

// ANN-AE is non-spiking: analog ("direct") by construction, one frame, no threshold. These
// constants define the model; they are not values a profile could choose (the ANN-AE
// profile is refused if it declares encoding/time_steps/voltage_threshold).
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

// Sequence AE (Meeting01-paper extractor) on the SAME input as the other autoencoders and
// PCA: the 256-value log power spectrum (normalize_signals), read as a sequence of
// kLstmSteps steps of kLstmStepWidth adjacent bins, low frequency first. Until 2026-11-12
// it read 64 frames of the raw waveform instead, so a profile choosing lstm-ae would have
// compared a different input, not a different extractor.
FeatureSet extract_lstm_ae_feature_set(const std::vector<std::vector<double>>& raw_signals,
    const ThesisConfig::FeatureExtraction& cfg,
    const ThesisConfig::Training& training,
    const std::string& label_suffix)
{
    const std::vector<std::vector<float>> normed = normalize_signals(raw_signals);
    std::vector<nn::Tensor> train_samples;
    train_samples.reserve(normed.size());
    for (const auto& row : normed)
        train_samples.push_back(spectrum_to_sequence(row, kLstmSteps, kLstmStepWidth));

    const std::vector<std::string>& spec = cfg.autoencoder.encoder_layer_spec;
    nn::models::lstm::LSTMAutoencoderConfig ae_cfg;
    ae_cfg.input_size = kLstmStepWidth;
    ae_cfg.seq_len = kLstmSteps;
    ae_cfg.hidden_size = first_encoder_dim(spec);
    ae_cfg.latent_size = last_encoder_dim(spec);
    // One LSTM layer per two encoder items; a 1-item spec used to be clamped to 1 layer.
    if (spec.size() < 2)
        throw std::invalid_argument(
            "ThesisFeatureExtraction: lstm-ae needs at least 2 encoder_layer_spec items (one "
            "LSTM layer per two), got " +
            std::to_string(spec.size()));
    ae_cfg.num_layers = static_cast<int>(spec.size() / 2);

    nn::models::lstm::LSTMAutoencoder model(ae_cfg);

    // validate() pins ae_loss_type to mse for lstm-ae: this Trainer's loss is MSE.
    const nn::training::TrainerConfig trainer_cfg =
        build_trainer_config(training, training.samples_per_batch, /*time_major=*/false);

    nn::training::Trainer<nn::models::lstm::LSTMAutoencoder> trainer(model, trainer_cfg);
    // Same enrichment as the SNN/ANN AE path: label the bar with the model + loss so
    // the metadata line matches the Meeting01 TUI look.
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

    if (ae_model == "lstm-ae")
        return extract_lstm_ae_feature_set(raw_signals, cfg, training, label_suffix);

    throw std::invalid_argument("ThesisFeatureExtraction: unknown autoencoder.model \"" + ae_model +
                                "\" (expected snn-ae, ann-ae or lstm-ae)");
}

// PCA on the autoencoders' exact input (normalize_signals: the 256-bin log power spectrum, min-max
// per sample), fitted on every sample — the same unsupervised, transductive protocol as the
// autoencoders, so the comparison differs only in the extractor.
FeatureSet extract_pca_feature_set(const std::vector<std::vector<double>>& raw_signals,
    const ThesisConfig::PcaConfig& cfg,
    const std::string& label_suffix)
{
    const std::vector<std::vector<float>> normed = normalize_signals(raw_signals);
    const PcaProjection pca = fit_pca(normed, cfg.n_components);

    FeatureSet fs;
    fs.label = "pca" + label_suffix;
    fs.vectors.reserve(normed.size());
    for (const auto& row : normed) fs.vectors.push_back(project_pca(pca, row));
    return fs;
}

// Runs the configured strategy (handcrafted, autoencoder or pca) against whatever
// single signal get_signal() returns per sample. Used directly for "voice"
// and "eeg" modality, and reused twice by late fusion (once per signal) and
// once by early fusion (on the pre-concatenated signal). label_suffix tags
// the resulting FeatureSet so voice-part/EEG-part/fused variants stay
// distinguishable in results output. part_tag names the model for the snapshot
// callback ("" = the only model; "voice"/"eeg" = late fusion's two models).
auto extract_features_core(const ThesisDatasetView& view,
    const ThesisConfig& cfg,
    const SignalGetter& get_signal,
    double sample_rate,
    const std::string& label_suffix,
    const std::string& part_tag,
    const ModelSnapshotFn& on_model_trained) -> std::vector<FeatureSet>
{
    const ThesisConfig::FeatureExtraction& fe = cfg.feature_extraction;
    std::vector<FeatureSet> result;

    if (fe.strategy == "handcrafted")
    {
        result.push_back(
            extract_handcrafted_feature_set(view, fe, get_signal, sample_rate, label_suffix));
    }
    else if (fe.strategy == "autoencoder")
    {
        const std::vector<std::vector<double>> raw_signals = collect_raw_signals(view, get_signal);
        result.push_back(extract_autoencoder_feature_set(raw_signals,
            fe,
            cfg.training,
            label_suffix,
            cfg.experiment.seed,
            part_tag,
            on_model_trained));
    }
    else if (fe.strategy == "pca")
    {
        result.push_back(
            extract_pca_feature_set(collect_raw_signals(view, get_signal), fe.pca, label_suffix));
    }
    else
    {
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown strategy \"" + fe.strategy + "\"");
    }

    return result;
}

} // namespace

auto extract_features(
    const ThesisDatasetView& view, const ThesisConfig& cfg, const ModelSnapshotFn& on_model_trained)
    -> std::vector<FeatureSet>
{
    const std::string& modality = cfg.dataset.modality;
    if (modality == "voice")
        return extract_features_core(
            view, cfg, voice_signal, kVoiceSampleRate, "", "", on_model_trained);

    if (modality == "eeg")
        return extract_features_core(
            view, cfg, eeg_signal, kEegSampleRate, "", "", on_model_trained);

    if (modality != "fused")
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown modality \"" + modality + "\"");

    const std::string& fusion_mode = cfg.dataset.fusion_mode;
    if (fusion_mode == "early")
    {
        return extract_features_core(
            view, cfg, fused_early_signal, kVoiceSampleRate, "-fused-early", "", on_model_trained);
    }

    if (fusion_mode != "late")
        throw std::invalid_argument(
            "ThesisFeatureExtraction: unknown fusion_mode \"" + fusion_mode + "\"");

    // Late fusion: extract independently per signal, then concatenate the
    // resulting feature vectors sample-by-sample (audit C12). Two INDEPENDENT models
    // train (or, for PCA, are fitted) here — tag each snapshot so a caller merging them
    // (e.g. paraconsistentGA's weight-snapshot capture) can tell them apart.
    auto voice_sets = extract_features_core(
        view, cfg, voice_signal, kVoiceSampleRate, "", "voice", on_model_trained);
    auto eeg_sets =
        extract_features_core(view, cfg, eeg_signal, kEegSampleRate, "", "eeg", on_model_trained);

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

auto spectral_input(const std::vector<double>& signal) -> std::vector<float>
{
    return normalize01(log_power_spectrum(signal));
}

} // namespace thesis
