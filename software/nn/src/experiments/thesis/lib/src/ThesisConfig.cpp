#include "ThesisConfig.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "StrictJson.hpp"

namespace thesis
{

namespace
{

using K = const ThesisConfig&;

// ─── Who reads what ──────────────────────────────────────────────────────────
//
// One row per profile key: the field, and the condition under which the run READS it, in
// words (quoted in error messages) and as code. This table is the single statement of the
// rules; the parser, check_declarations() and declared_json() all walk it.
//
// A condition may read other fields. Rows are ordered so that every field a condition
// reads comes earlier and has already been checked, and conditions short-circuit, so they
// never read a field that does not apply (`trains_ae(k) && model == ...` reads `model` only
// for an autoencoder run).
//
//   strategy      handcrafted       autoencoder               pca
//                 -----------       -----------               ---
//   reads         handcrafted.*     autoencoder.* (by model)  pca.n_components
//   trains        nothing           the AE (training: epochs, lr, optimizer, clip, batch)
//
//   classifier.enabled = true additionally reads classifier.* and the classifier's
//   training keys (folds, CV, standardization, weight decay; dsnn: firing rate, tdBN).

auto trains_ae(K k) -> bool
{
    return k.feature_extraction.strategy == "autoencoder";
}
auto is_snn_ae(K k) -> bool
{
    return trains_ae(k) && k.feature_extraction.autoencoder.model == "snn-ae";
}
auto trains_classifier(K k) -> bool
{
    return k.classifier.enabled;
}
/// A classifier that trains a network (dsnn, ann). The "cosine" control trains nothing.
auto trains_network_classifier(K k) -> bool
{
    return trains_classifier(k) && k.classifier.type != "cosine";
}
auto trains_anything(K k) -> bool
{
    return trains_ae(k) || trains_network_classifier(k);
}
auto reads_eeg(K k) -> bool
{
    return k.dataset.modality != "voice";
}
auto is_dsnn(K k) -> bool
{
    return trains_classifier(k) && k.classifier.type == "dsnn";
}

constexpr const char* kAlways = "always";
auto always(K /*unused*/) -> bool
{
    return true;
}

template <typename Cfg, typename Row>
void for_each_field(Cfg& c, Row&& row)
{
    auto& e = c.experiment;
    row(e.run_tag, kAlways, always);
    row(e.seed, kAlways, always);
    row(e.repeats, kAlways, always);
    row(e.seed_deterministic,
        "experiment.repeats > 1",
        [](K k) { return k.experiment.repeats > 1; });

    auto& d = c.dataset;
    row(d.source, kAlways, always);
    row(d.root, kAlways, always);
    row(d.results_dir, kAlways, always);
    row(d.modality, kAlways, always);
    row(d.max_samples, kAlways, always);
    row(d.fusion_mode,
        "dataset.modality is \"fused\"",
        [](K k) { return k.dataset.modality == "fused"; });
    row(d.eeg_condition, "dataset.modality is \"eeg\" or \"fused\"", reads_eeg);
    row(d.eeg_pairing_seed,
        "dataset.eeg_condition is \"imagined\"",
        [](K k) { return reads_eeg(k) && k.dataset.eeg_condition == "imagined"; });

    auto& fe = c.feature_extraction;
    row(fe.strategy, kAlways, always);

    constexpr const char* kHandcrafted = "feature_extraction.strategy is \"handcrafted\"";
    const auto handcrafted = [](K k) { return k.feature_extraction.strategy == "handcrafted"; };
    auto& hc = fe.handcrafted;
    row(hc.transform, kHandcrafted, handcrafted);
    row(hc.scale, kHandcrafted, handcrafted);
    row(hc.descriptors, kHandcrafted, handcrafted);
    row(hc.dtwpt_level, kHandcrafted, handcrafted);
    row(hc.wavelet, kHandcrafted, handcrafted);
    row(hc.cepstral, kHandcrafted, handcrafted);

    constexpr const char* kAe = "feature_extraction.strategy is \"autoencoder\"";
    constexpr const char* kSnn = "the autoencoder model is \"snn-ae\"";
    auto& ae = fe.autoencoder;
    row(ae.model, kAe, trains_ae);
    row(ae.encoder_layer_spec, kAe, trains_ae);
    row(ae.decoder_layer_spec,
        "the autoencoder model is \"snn-ae\" or \"ann-ae\"",
        [](K k) { return trains_ae(k) && k.feature_extraction.autoencoder.model != "lstm-ae"; });
    row(ae.ae_loss_type, kAe, trains_ae);
    row(ae.encoding, kSnn, is_snn_ae);
    row(ae.time_steps,
        "the autoencoder is \"snn-ae\" with a temporal encoding (poisson, latency)",
        [](K k) { return is_snn_ae(k) && k.feature_extraction.autoencoder.encoding != "direct"; });
    row(ae.voltage_threshold, kSnn, is_snn_ae);
    row(ae.firing_rate_reg_lambda, kSnn, is_snn_ae);
    constexpr const char* kAeBand = "the autoencoder is \"snn-ae\" with firing_rate_reg_lambda > 0";
    const auto ae_band = [](K k)
    { return is_snn_ae(k) && k.feature_extraction.autoencoder.firing_rate_reg_lambda > 0.0f; };
    row(ae.firing_rate_min, kAeBand, ae_band);
    row(ae.firing_rate_max, kAeBand, ae_band);

    row(fe.pca.n_components,
        "feature_extraction.strategy is \"pca\"",
        [](K k) { return k.feature_extraction.strategy == "pca"; });

    auto& pc = c.paraconsistent;
    row(pc.enabled, kAlways, always);
    row(pc.speaker_subsets,
        "paraconsistent.enabled is true",
        [](K k) { return k.paraconsistent.enabled; });
    constexpr const char* kSubsets = "paraconsistent.speaker_subsets is true";
    const auto subsets = [](K k)
    { return k.paraconsistent.enabled && k.paraconsistent.speaker_subsets; };
    row(pc.fold_k, kSubsets, subsets);
    row(pc.fold_seed, kSubsets, subsets);
    row(pc.fold_repeats, kSubsets, subsets);

    constexpr const char* kClf = "classifier.enabled is true";
    constexpr const char* kNetClf = "classifier.enabled with classifier.type \"dsnn\" or \"ann\"";
    auto& cl = c.classifier;
    row(cl.enabled, kAlways, always);
    row(cl.type, kClf, trains_classifier);
    row(cl.layer_spec, kNetClf, trains_network_classifier);
    row(cl.text_mode, kClf, trains_classifier);

    auto& dg = c.degradation;
    row(dg.enabled,
        "classifier.enabled with dataset.modality \"voice\" or \"fused\"",
        [](K k) { return trains_classifier(k) && k.dataset.modality != "eeg"; });
    constexpr const char* kDegraded = "degradation.enabled is true";
    const auto degraded = [](K k)
    { return trains_classifier(k) && k.dataset.modality != "eeg" && k.degradation.enabled; };
    row(dg.level, kDegraded, degraded);
    row(dg.scope, kDegraded, degraded);
    row(dg.seed, kDegraded, degraded);

    constexpr const char* kTrains =
        "a training runs (strategy \"autoencoder\", or a dsnn/ann classifier)";
    auto& t = c.training;
    row(t.epochs, kTrains, trains_anything);
    row(t.learning_rate, kTrains, trains_anything);
    row(t.optimizer_type, kTrains, trains_anything);
    row(t.optimizer_momentum,
        "a training runs with training.optimizer_type \"sgd\"",
        [](K k) { return trains_anything(k) && k.training.optimizer_type == "sgd"; });
    row(t.gradient_clip_norm, kTrains, trains_anything);
    row(t.samples_per_batch, kTrains, trains_anything);
    row(t.k_folds, kClf, trains_classifier);
    row(t.nested_cv, kClf, trains_classifier);
    row(t.early_stop_patience,
        "classifier.enabled and training.nested_cv are true",
        [](K k) { return trains_classifier(k) && k.training.nested_cv; });
    row(t.standardize_features, kClf, trains_classifier);
    row(t.weight_decay, kNetClf, trains_network_classifier);
    constexpr const char* kDsnn = "the classifier is enabled with classifier.type \"dsnn\"";
    row(t.firing_rate_reg_lambda, kDsnn, is_dsnn);
    constexpr const char* kDsnnBand =
        "the classifier is \"dsnn\" with training.firing_rate_reg_lambda > 0";
    const auto dsnn_band = [](K k)
    { return is_dsnn(k) && k.training.firing_rate_reg_lambda > 0.0f; };
    row(t.firing_rate_min, kDsnnBand, dsnn_band);
    row(t.firing_rate_max, kDsnnBand, dsnn_band);
    row(t.batch_normalization, kDsnn, is_dsnn);
    row(t.tdbn_alpha,
        "the classifier is \"dsnn\" with training.batch_normalization "
        "\"threshold-dependent\"",
        [](K k) { return is_dsnn(k) && k.training.batch_normalization == "threshold-dependent"; });
}

auto find_key(const std::vector<ThesisConfig::Embedding::Key>& keys, std::string_view path)
    -> const ThesisConfig::Embedding::Key*
{
    const auto it =
        std::find_if(keys.begin(), keys.end(), [&](const auto& key) { return key.path == path; });
    return it == keys.end() ? nullptr : &*it;
}

// ─── JSON -> fields ──────────────────────────────────────────────────────────

// The name every config error starts with (a function-local static: a namespace-scope
// std::string may throw during static initialisation, where nothing can catch it).
auto kOwner() -> const std::string&
{
    static const std::string owner = "ThesisConfig";
    return owner;
}

auto parse(const nlohmann::json& j, const ThesisConfig::Embedding* embedding) -> ThesisConfig
{
    ThesisConfig cfg;
    if (embedding != nullptr) cfg.embedded_in = *embedding;

    std::set<std::string> fields;
    for_each_field(cfg,
        [&](auto& field, const char* /*when*/, auto /*applies*/) { fields.insert(field.path()); });

    const std::string owner =
        embedding != nullptr ? kOwner() + " (embedded in " + embedding->program + ")" : kOwner();
    strict::refuse_unknown_keys(j,
        "",
        fields,
        strict::sections_of(fields),
        embedding != nullptr ? embedding->own_sections : std::vector<std::string>{},
        owner);

    if (embedding != nullptr)
    {
        const auto refuse = [&](const std::vector<ThesisConfig::Embedding::Key>& keys)
        {
            for (const auto& key : keys)
                if (strict::find_path(j, key.path) != nullptr)
                    throw std::invalid_argument("ThesisConfig: the profile declares \"" + key.path +
                                                "\", but " + embedding->program + " " + key.reason +
                                                ". Remove the key.");
        };
        refuse(embedding->set_by_program);
        refuse(embedding->not_read);
    }

    for_each_field(cfg,
        [&](auto& field, const char* /*when*/, auto /*applies*/)
        {
            using T = typename std::remove_cvref_t<decltype(field)>::value_type;
            if (const auto* v = strict::find_path(j, field.path()))
                field = strict::read_exact<T>(*v, kOwner(), field.path());
        });

    cfg.check_declarations();
    return cfg;
}

// ─── Value checks ────────────────────────────────────────────────────────────
//
// They run after check_declarations(), which has proven that a field is declared exactly
// when the run reads it. So below, `x.declared()` MEANS "x applies to this run" — it is
// not an "optional field" test. (The one exception is a field an embedding program sets
// itself, e.g. the GA's genome-owned autoencoder keys: undeclared in the GA's base config,
// declared in the configs it builds per genome, which are validated again.)

void validate_experiment(const ThesisConfig::Experiment& experiment)
{
    // run_tag and repeats are undeclared in a GA-embedded config (the GA does not read them).
    if (experiment.run_tag.declared() && experiment.run_tag->empty())
        throw std::invalid_argument("ThesisConfig: experiment.run_tag must not be empty");
    if (experiment.seed == 0u)
        throw std::invalid_argument("ThesisConfig: experiment.seed must be non-zero");
    if (experiment.repeats.declared() && experiment.repeats <= 0)
        throw std::invalid_argument("ThesisConfig: experiment.repeats must be > 0");
}

void validate_dataset(const ThesisConfig::Dataset& dataset)
{
    if (dataset.root->empty())
        throw std::invalid_argument("ThesisConfig: dataset.root must not be empty");
    if (dataset.results_dir.declared() && dataset.results_dir->empty())
        throw std::invalid_argument("ThesisConfig: dataset.results_dir must not be empty");

    const bool valid_modality =
        dataset.modality == "voice" || dataset.modality == "eeg" || dataset.modality == "fused";
    if (!valid_modality)
        throw std::invalid_argument("ThesisConfig: dataset.modality must be voice/eeg/fused");
    if (dataset.max_samples < 0)
        throw std::invalid_argument(
            "ThesisConfig: dataset.max_samples must be >= 0 (0 = every sample)");

    if (dataset.fusion_mode.declared() && dataset.fusion_mode != "early" &&
        dataset.fusion_mode != "late")
        throw std::invalid_argument("ThesisConfig: dataset.fusion_mode must be early/late");

    if (dataset.eeg_condition.declared() && dataset.eeg_condition != "pronounced" &&
        dataset.eeg_condition != "imagined")
        throw std::invalid_argument(
            "ThesisConfig: dataset.eeg_condition must be pronounced or imagined");
    if (dataset.eeg_pairing_seed.declared() && dataset.eeg_pairing_seed == 0u)
        throw std::invalid_argument("ThesisConfig: dataset.eeg_pairing_seed must be non-zero");

    if (dataset.source != "coretto2017" && dataset.source != "biomex2021")
        throw std::invalid_argument(
            "ThesisConfig: dataset.source must be coretto2017 or "
            "biomex2021, got \"" +
            *dataset.source + "\"");
    if (dataset.source == "biomex2021" && dataset.eeg_condition.declared() &&
        dataset.eeg_condition == "imagined")
        throw std::invalid_argument(
            "ThesisConfig: dataset.source biomex2021 has no imagined speech; its EEG is "
            "recorded while the digit is spoken (dataset.eeg_condition \"pronounced\")");
}

void validate_paraconsistent(const ThesisConfig::Paraconsistent& pc)
{
    if (!pc.fold_k.declared()) return; // no speaker subsets (check_declarations)
    if (pc.fold_k < 2)
        throw std::invalid_argument("ThesisConfig: paraconsistent.fold_k must be >= 2");
    if (pc.fold_seed == 0u)
        throw std::invalid_argument("ThesisConfig: paraconsistent.fold_seed must be non-zero");
    if (pc.fold_repeats < 1)
        throw std::invalid_argument("ThesisConfig: paraconsistent.fold_repeats must be >= 1");
}

void validate_degradation(const ThesisConfig& cfg)
{
    const auto& dg = cfg.degradation;
    if (!dg.level.declared()) return; // not degraded (check_declarations)
    if (!(dg.level > 0.0f && dg.level <= 1.0f))
        throw std::invalid_argument("ThesisConfig: degradation.level must be in (0, 1]");
    if (dg.scope != "probes" && dg.scope != "test")
        throw std::invalid_argument("ThesisConfig: degradation.scope must be probes or test");
    if (dg.seed == 0u)
        throw std::invalid_argument("ThesisConfig: degradation.seed must be non-zero");
    // The degraded features are extracted once, beside the clean ones. An autoencoder is
    // trained per repetition on the clean signals, so it has no single "degraded" twin.
    if (cfg.feature_extraction.strategy == "autoencoder")
        throw std::invalid_argument(
            "ThesisConfig: degradation needs a fixed extractor (handcrafted or pca); an "
            "autoencoder is retrained per repetition");
}

void validate_handcrafted(
    const ThesisConfig::HandcraftedConfig& handcrafted, const ThesisConfig::Dataset& dataset)
{
    if (handcrafted.transform != "dtwpt")
        throw std::invalid_argument("ThesisConfig: handcrafted.transform must be dtwpt");

    const bool valid_scale =
        handcrafted.scale == "bark" || handcrafted.scale == "mel" || handcrafted.scale == "lfcc";
    if (!valid_scale)
        throw std::invalid_argument("ThesisConfig: handcrafted.scale must be bark/mel/lfcc");

    // Bark and Mel are cochlear scales: they model the frequency resolution of human
    // HEARING. There is no physiological basis for applying them to EEG, which is not
    // sound. They are rejected for modality=eeg on that principle
    // (.wiki/Guides/Engineering-Fixes-Log.md D6).
    //
    // Empirically they were also inert here, which is what exposed the problem: for EEG
    // all three scales produced bit-identical d_truth in 46/46 wavelet x category groups.
    // The mechanism is group_by_scale()'s normalization by the signal's own Nyquist: for
    // EEG's 512 Hz Nyquist the Bark curve is stretched ~5x, the mapping becomes injective,
    // and the grouping degenerates to exactly lfcc's one-group-per-sub-band.
    //
    // The grid is therefore deliberately asymmetric: voice sweeps all three scales, EEG
    // uses lfcc only. modality=fused is intentionally NOT covered: its voice half
    // legitimately uses bark/mel.
    if (dataset.modality == "eeg" && handcrafted.scale != "lfcc")
        throw std::invalid_argument(
            "ThesisConfig: handcrafted.scale must be lfcc for modality=eeg — bark/mel are "
            "cochlear (hearing) scales with no physiological basis for EEG, and are "
            "provably degenerate to lfcc there (see .wiki/Guides/Engineering-Fixes-Log.md D6)");

    // Mother wavelets with coefficient traits in include/wavelet/Types.hpp.
    static const std::vector<std::string> valid_wavelets = {"haar",
        "daub4",
        "daub6",
        "daub8",
        "daub10",
        "daub12",
        "daub14",
        "daub16",
        "daub18",
        "daub20",
        "daub22",
        "daub24",
        "daub26",
        "daub28",
        "daub30",
        "daub32",
        "daub34",
        "daub36",
        "daub38",
        "daub40",
        "daub42",
        "daub44",
        "daub46"};
    if (std::find(valid_wavelets.begin(), valid_wavelets.end(), handcrafted.wavelet.get()) ==
        valid_wavelets.end())
        throw std::invalid_argument(
            "ThesisConfig: handcrafted.wavelet must be haar or daubN (even N in [4,46])");

    if (handcrafted.descriptors->empty())
        throw std::invalid_argument("ThesisConfig: handcrafted.descriptors must not be empty");
    // The extractor looks each known name up and computes it; a name it does not know, or
    // a repeated one, used to be passed over without a word. With cepstral = true the
    // band energies become the cepstral coefficients, so a declared "energy" was dropped.
    const std::vector<std::string> known = {
        "energy", "zcr", "entropy", "teager", "jitter", "shimmer"};
    std::vector<std::string> seen;
    for (const auto& d : *handcrafted.descriptors)
    {
        if (std::find(known.begin(), known.end(), d) == known.end())
            throw std::invalid_argument("ThesisConfig: handcrafted.descriptors has \"" + d +
                                        "\"; known: energy, zcr, entropy, teager, jitter, "
                                        "shimmer");
        if (std::find(seen.begin(), seen.end(), d) != seen.end())
            throw std::invalid_argument(
                "ThesisConfig: handcrafted.descriptors lists \"" + d + "\" twice");
        seen.push_back(d);
    }
    if (handcrafted.cepstral && std::find(seen.begin(), seen.end(), "energy") != seen.end())
        throw std::invalid_argument(
            "ThesisConfig: handcrafted.cepstral = true replaces the band energies with their "
            "cepstral coefficients; \"energy\" in descriptors would be ignored — remove it");

    if (handcrafted.dtwpt_level < 1)
        throw std::invalid_argument("ThesisConfig: handcrafted.dtwpt_level must be >= 1");
}

void validate_encoding_and_loss(const ThesisConfig::AutoencoderConfig& ae)
{
    const std::string& model = ae.model;
    const std::string& loss = ae.ae_loss_type;
    const bool spike_loss = (loss == "spikecount" || loss == "spiketime");

    // Spike losses need spikes: only the spiking model produces them.
    if (spike_loss && model != "snn-ae")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.ae_loss_type=" + loss +
            " requires model=snn-ae (ann-ae/lstm-ae emit continuous values, not spikes)");

    // The LSTM-AE path trains with MSE and nothing else; any other token would be ignored.
    if (model == "lstm-ae" && loss != "mse")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.ae_loss_type must be mse for model=lstm-ae — its "
            "training path is MSE only; got " +
            loss);

    if (!ae.encoding.declared()) return; // ann-ae / lstm-ae, or set later by a GA genome
    const std::string& encoding = ae.encoding;
    if (encoding != "poisson" && encoding != "latency" && encoding != "direct")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.encoding must be poisson, latency, or direct");

    if (spike_loss && encoding == "direct")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.ae_loss_type=" + loss +
            " is invalid for encoding=direct — direct is analog (no spikes); use mse or mae");

    // Encoding<->loss invariant (.wiki/Concepts/Spike-Encoding.md): the wrong spike loss
    // does not merely underperform, it destroys the gradient signal.
    //   SpikeTimeLoss sees only the FIRST spike -> blind to rate/count information.
    //   SpikeCountLoss treats absent spikes as count 0 -> wrong gradient for late spikes.
    if (loss == "spiketime" && encoding != "latency")
        throw std::invalid_argument(
            "ThesisConfig: ae_loss_type=spiketime requires encoding=latency "
            "(first-spike timing carries the information); got encoding=" +
            encoding);
    if (loss == "spikecount" && encoding != "poisson")
        throw std::invalid_argument(
            "ThesisConfig: ae_loss_type=spikecount requires encoding=poisson "
            "(rate/count carries the information); got encoding=" +
            encoding);
}

void validate_autoencoder(const ThesisConfig::AutoencoderConfig& ae)
{
    if (ae.model != "snn-ae" && ae.model != "ann-ae" && ae.model != "lstm-ae")
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.model must be snn-ae, ann-ae, or lstm-ae");
    if (ae.encoder_layer_spec.declared() && ae.encoder_layer_spec->empty())
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.encoder_layer_spec must not be empty");
    if (ae.decoder_layer_spec.declared() && ae.decoder_layer_spec->empty())
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.decoder_layer_spec must not be empty");

    const std::string& loss = ae.ae_loss_type;
    const bool known_loss =
        loss == "mse" || loss == "mae" || loss == "spikecount" || loss == "spiketime";
    if (!known_loss)
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.ae_loss_type must be mse, mae, spikecount, or "
            "spiketime");
    validate_encoding_and_loss(ae);

    if (ae.time_steps.declared() && ae.time_steps < 1)
        throw std::invalid_argument("ThesisConfig: autoencoder.time_steps must be >= 1");
    // Latency code puts each input's spike at a time step: one step has nowhere to put it.
    if (ae.encoding.declared() && ae.encoding.get() == "latency" && ae.time_steps.declared() &&
        ae.time_steps < 2)
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.encoding=latency needs time_steps >= 2");
    if (ae.voltage_threshold.declared() && ae.voltage_threshold <= 0.0f)
        throw std::invalid_argument("ThesisConfig: autoencoder.voltage_threshold must be > 0");
    if (ae.firing_rate_reg_lambda.declared() && ae.firing_rate_reg_lambda < 0.0f)
        throw std::invalid_argument(
            "ThesisConfig: autoencoder.firing_rate_reg_lambda must be >= 0 (0 = off)");
    if (ae.firing_rate_min.declared() && (ae.firing_rate_min < 0.0f || ae.firing_rate_max > 1.0f ||
                                             ae.firing_rate_min > ae.firing_rate_max))
        throw std::invalid_argument(
            "ThesisConfig: require 0 <= autoencoder.firing_rate_min <= "
            "autoencoder.firing_rate_max <= 1");
}

void validate_pca(const ThesisConfig::PcaConfig& pca)
{
    // 256 = the spectral input width shared with the autoencoders (kAeInputFeatures). The
    // other bound, n_samples - 1, depends on the data and is checked when PCA is fitted.
    constexpr int kInputWidth = 256;
    if (pca.n_components < 1 || pca.n_components > kInputWidth)
        throw std::invalid_argument(
            "ThesisConfig: feature_extraction.pca.n_components must be in [1, 256] (the "
            "spectral input has 256 values); got " +
            std::to_string(pca.n_components.get()));
}

void validate_classifier(const ThesisConfig::Classifier& classifier)
{
    if (!classifier.enabled) return; // Phase 00: check_declarations refused the other keys

    if (classifier.type != "ann" && classifier.type != "dsnn" && classifier.type != "cosine")
        throw std::invalid_argument("ThesisConfig: classifier.type must be dsnn, ann or cosine");

    const bool valid_text_mode =
        classifier.text_mode == "dependent" || classifier.text_mode == "independent";
    if (!valid_text_mode)
        throw std::invalid_argument(
            "ThesisConfig: classifier.text_mode must be dependent/independent");

    if (!classifier.layer_spec.declared()) return; // cosine: no network

    const std::vector<std::string>& spec = classifier.layer_spec;
    if (spec.size() < 3)
        throw std::invalid_argument(
            "ThesisConfig: classifier.layer_spec must have at least 3 items: "
            "linear:H:act, residual:D, linear:N_speakers:identity");

    const auto starts_with = [](const std::string& s, const std::string& p)
    { return s.rfind(p, 0) == 0; };

    if (!starts_with(spec.front(), "linear:"))
        throw std::invalid_argument(
            "ThesisConfig: first classifier.layer_spec item must start with linear:");

    const bool has_residual = std::any_of(spec.begin(),
        spec.end(),
        [&](const std::string& s) { return starts_with(s, "residual:"); });
    if (!has_residual)
        throw std::invalid_argument(
            "ThesisConfig: classifier.layer_spec must include residual:D item");

    if (!starts_with(spec.back(), "linear:N_speakers:"))
        throw std::invalid_argument(
            "ThesisConfig: last classifier.layer_spec item must start with linear:N_speakers:");
}

void validate_training(const ThesisConfig::Training& training)
{
    if (!training.epochs.declared()) return; // no training runs (check_declarations)

    if (training.epochs <= 0)
        throw std::invalid_argument("ThesisConfig: training.epochs must be > 0");
    if (training.learning_rate <= 0.0f)
        throw std::invalid_argument("ThesisConfig: training.learning_rate must be > 0");
    if (training.samples_per_batch <= 0)
        throw std::invalid_argument("ThesisConfig: training.samples_per_batch must be > 0");
    if (training.optimizer_type != "adam" && training.optimizer_type != "sgd" &&
        training.optimizer_type != "lion" && training.optimizer_type != "schedule-free-adamw")
        throw std::invalid_argument(
            "ThesisConfig: training.optimizer_type must be one of "
            "adam, sgd, lion, schedule-free-adamw");
    if (training.optimizer_momentum.declared() &&
        (training.optimizer_momentum < 0.0f || training.optimizer_momentum >= 1.0f))
        throw std::invalid_argument("ThesisConfig: require 0 <= training.optimizer_momentum < 1");
    if (training.gradient_clip_norm < 0.0f)
        throw std::invalid_argument(
            "ThesisConfig: training.gradient_clip_norm must be >= 0 (0 = off)");

    if (training.k_folds.declared() && training.k_folds < 2)
        throw std::invalid_argument("ThesisConfig: training.k_folds must be >= 2");
    if (training.early_stop_patience.declared() && training.early_stop_patience < 0)
        throw std::invalid_argument("ThesisConfig: training.early_stop_patience must be >= 0");
    if (training.weight_decay.declared() && training.weight_decay < 0.0f)
        throw std::invalid_argument("ThesisConfig: training.weight_decay must be >= 0");
    if (training.firing_rate_reg_lambda.declared() && training.firing_rate_reg_lambda < 0.0f)
        throw std::invalid_argument(
            "ThesisConfig: training.firing_rate_reg_lambda must be >= 0 (0 = off)");
    if (training.firing_rate_min.declared() &&
        (training.firing_rate_min < 0.0f || training.firing_rate_max > 1.0f ||
            training.firing_rate_min > training.firing_rate_max))
        throw std::invalid_argument(
            "ThesisConfig: require 0 <= training.firing_rate_min <= training.firing_rate_max "
            "<= 1");
    if (training.batch_normalization.declared() && training.batch_normalization != "none" &&
        training.batch_normalization != "threshold-dependent")
        throw std::invalid_argument(
            "ThesisConfig: training.batch_normalization must be none or threshold-dependent");
    if (training.tdbn_alpha.declared() && training.tdbn_alpha <= 0.0f)
        throw std::invalid_argument("ThesisConfig: training.tdbn_alpha must be > 0");
}

} // namespace

void ThesisConfig::check_declarations() const
{
    const Embedding* emb = embedded_in ? &*embedded_in : nullptr;
    for_each_field(*this,
        [&](const auto& field, const char* when, auto applies)
        {
            const std::string path = field.path();
            if (emb != nullptr && find_key(emb->set_by_program, path) != nullptr)
                return; // the program fills it in before use
            if (const auto* ignored = emb != nullptr ? find_key(emb->not_read, path) : nullptr)
            {
                if (field.declared())
                    throw std::invalid_argument(kOwner() + ": the profile declares \"" + path +
                                                "\", which this run never reads (" + emb->program +
                                                " " + ignored->reason + "). Remove it.");
                return;
            }
            strict::check_declared(field, applies(*this), when, kOwner());
        });
}

void ThesisConfig::validate() const
{
    check_declarations();

    validate_experiment(experiment);
    validate_dataset(dataset);

    const std::string& strategy = feature_extraction.strategy;
    if (strategy == "handcrafted")
        validate_handcrafted(feature_extraction.handcrafted, dataset);
    else if (strategy == "autoencoder")
        validate_autoencoder(feature_extraction.autoencoder);
    else if (strategy == "pca")
        validate_pca(feature_extraction.pca);
    else
        throw std::invalid_argument(
            "ThesisConfig: feature_extraction.strategy must be handcrafted, autoencoder, or pca; "
            "got \"" +
            strategy + "\"");

    validate_paraconsistent(paraconsistent);
    validate_classifier(classifier);
    validate_degradation(*this);
    validate_training(training);

    // The cosine control has no model to select, so a nested split would only repeat the
    // flat one under another name.
    if (classifier.enabled && classifier.type == "cosine" && training.nested_cv)
        throw std::invalid_argument(
            "ThesisConfig: classifier.type cosine trains nothing; training.nested_cv must be "
            "false");
}

ThesisConfig ThesisConfig::from_json(const nlohmann::json& j)
{
    return parse(j, nullptr);
}

ThesisConfig ThesisConfig::from_json(const nlohmann::json& j, const Embedding& embedding)
{
    return parse(j, &embedding);
}

ThesisConfig ThesisConfig::from_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("ThesisConfig: cannot open " + path);
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

auto ThesisConfig::declared_json() const -> nlohmann::json
{
    nlohmann::json j = nlohmann::json::object();
    for_each_field(*this,
        [&](const auto& field, const char* /*when*/, auto /*applies*/)
        {
            if (!field.declared()) return;
            std::string pointer = std::string("/") + field.path();
            std::replace(pointer.begin(), pointer.end(), '.', '/');
            j[nlohmann::json::json_pointer(pointer)] = field.get();
        });
    return j;
}

} // namespace thesis
