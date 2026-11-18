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
#include "VoiceDysphonia.hpp"

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

auto trains_classifier(K k) -> bool
{
    return k.classifier.enabled;
}
/// The per-fold protocol: the extractor and the classifier's hyperparameters are chosen in
/// each fold (selection.*), so the profile's own extractor and those keys do not exist.
auto selects(K k) -> bool
{
    return trains_classifier(k) && k.selection.enabled;
}
/// The profile names its extractor (every run except the per-fold protocol).
auto has_extractor(K k) -> bool
{
    return !selects(k);
}
auto trains_ae(K k) -> bool
{
    return has_extractor(k) && k.feature_extraction.strategy == "autoencoder";
}
auto is_snn_ae(K k) -> bool
{
    return trains_ae(k) && k.feature_extraction.autoencoder.model == "snn-ae";
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
/// A network classifier whose hyperparameters the profile states (not chosen per fold).
auto fixed_network_classifier(K k) -> bool
{
    return trains_network_classifier(k) && !selects(k);
}
/// The learning rate: the autoencoder's, or a fixed network classifier's.
auto reads_learning_rate(K k) -> bool
{
    return trains_ae(k) || fixed_network_classifier(k);
}
auto reads_eeg(K k) -> bool
{
    return k.dataset.modality != "voice";
}
auto is_dsnn(K k) -> bool
{
    return trains_classifier(k) && k.classifier.type == "dsnn";
}
/// A run that degrades the voice of its test speakers (voice or fused data, classifier on).
auto degraded(K k) -> bool
{
    return trains_classifier(k) && k.dataset.modality != "eeg" && k.degradation.enabled;
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

    // Read by the extractor rows' conditions below, so they come first.
    row(c.classifier.enabled, kAlways, always);
    row(c.selection.enabled, "classifier.enabled is true", trains_classifier);

    auto& fe = c.feature_extraction;
    row(fe.strategy,
        "selection.enabled is not true (the per-fold protocol chooses the extractor)",
        has_extractor);

    constexpr const char* kHandcrafted = "feature_extraction.strategy is \"handcrafted\"";
    const auto handcrafted = [](K k)
    { return has_extractor(k) && k.feature_extraction.strategy == "handcrafted"; };
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
        [](K k) { return has_extractor(k) && k.feature_extraction.strategy == "pca"; });

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
    constexpr const char* kFixedNet =
        "a dsnn/ann classifier runs with selection.enabled false (otherwise selection.* chooses "
        "it)";
    auto& cl = c.classifier;
    row(cl.type, kClf, trains_classifier);
    row(cl.layer_spec, kFixedNet, fixed_network_classifier);
    row(cl.text_mode, kClf, trains_classifier);

    auto& sel = c.selection;
    constexpr const char* kSel = "selection.enabled is true";
    row(sel.rank_modality, kSel, selects);
    constexpr const char* kSelNet = "selection.enabled with classifier.type \"dsnn\" or \"ann\"";
    const auto selects_network = [](K k) { return selects(k) && trains_network_classifier(k); };
    row(sel.inner_k, kSelNet, selects_network);
    row(sel.hidden_widths, kSelNet, selects_network);
    row(sel.residual_blocks, kSelNet, selects_network);
    row(sel.learning_rates, kSelNet, selects_network);
    row(sel.weight_decays, kSelNet, selects_network);

    auto& dg = c.degradation;
    row(dg.enabled,
        "classifier.enabled with dataset.modality \"voice\" or \"fused\"",
        [](K k) { return trains_classifier(k) && k.dataset.modality != "eeg"; });
    constexpr const char* kDegraded = "degradation.enabled is true";
    row(dg.profile, kDegraded, degraded);
    row(dg.scope, kDegraded, degraded);
    row(dg.seed, kDegraded, degraded);
    // The keys each profile reads. `profile` comes first, so these conditions may read it.
    constexpr const char* kWhisper = "degradation.profile is \"whisper_mix\"";
    const auto whisper = [](K k) { return degraded(k) && k.degradation.profile == "whisper_mix"; };
    row(dg.level, kWhisper, whisper);
    constexpr const char* kRough = "degradation.profile is \"rough\"";
    const auto rough = [](K k) { return degraded(k) && k.degradation.profile == "rough"; };
    row(dg.jitter_pct, kRough, rough);
    row(dg.shimmer_db, kRough, rough);
    constexpr const char* kTremor = "degradation.profile is \"tremor\"";
    const auto tremor = [](K k) { return degraded(k) && k.degradation.profile == "tremor"; };
    row(dg.f0_depth_pct, kTremor, tremor);
    row(dg.rate_hz, kTremor, tremor);
    constexpr const char* kBreaks = "degradation.profile is \"voice_breaks\"";
    const auto breaks = [](K k) { return degraded(k) && k.degradation.profile == "voice_breaks"; };
    row(dg.break_kind, kBreaks, breaks);
    row(dg.breaks_per_second, kBreaks, breaks);
    row(dg.break_ms, kBreaks, breaks);
    constexpr const char* kDiplo = "degradation.profile is \"diplophonia\"";
    const auto diplo = [](K k) { return degraded(k) && k.degradation.profile == "diplophonia"; };
    row(dg.period_alt_pct, kDiplo, diplo);
    row(dg.amp_alt_db, kDiplo, diplo);

    constexpr const char* kTrains =
        "a training runs (strategy \"autoencoder\", or a dsnn/ann classifier)";
    auto& t = c.training;
    row(t.epochs, kTrains, trains_anything);
    row(t.learning_rate,
        "the autoencoder trains, or a dsnn/ann classifier with selection.enabled false",
        reads_learning_rate);
    row(t.optimizer_type, kTrains, trains_anything);
    row(t.optimizer_momentum,
        "a training runs with training.optimizer_type \"sgd\"",
        [](K k) { return trains_anything(k) && k.training.optimizer_type == "sgd"; });
    row(t.gradient_clip_norm, kTrains, trains_anything);
    row(t.samples_per_batch, kTrains, trains_anything);
    row(t.k_folds, kClf, trains_classifier);
    row(t.nested_cv,
        "classifier.enabled with selection.enabled false (the per-fold protocol runs its own "
        "inner folds)",
        [](K k) { return trains_classifier(k) && !selects(k); });
    row(t.early_stop_patience,
        "classifier.enabled and training.nested_cv are true",
        [](K k) { return trains_classifier(k) && !selects(k) && k.training.nested_cv; });
    row(t.standardize_features, kClf, trains_classifier);
    row(t.weight_decay, kFixedNet, fixed_network_classifier);
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

constexpr std::string_view kCandidatesPath = "selection.extractor_candidates";

// One candidate object: only its two blocks are read here. Its keys are checked when the
// candidate is turned into a complete Phase 00 profile (candidate_config), where every rule of
// a Phase 00 profile applies to them, unknown keys included.
auto parse_candidate(const nlohmann::json& c, std::size_t i) -> ThesisConfig::ExtractorCandidate
{
    std::string where = kOwner();
    where += ": ";
    where += kCandidatesPath;
    where += "[" + std::to_string(i) + "]";
    if (!c.is_object())
        throw std::invalid_argument(where + " must be an object, got " + strict::describe(c));
    for (const auto& [key, value] : c.items())
        if (key != "feature_extraction" && key != "training")
        {
            std::string msg = where;
            msg += " has the key \"";
            msg += key;
            msg +=
                "\"; a candidate holds only feature_extraction and training (the run's own "
                "keys apply to everything else)";
            throw std::invalid_argument(msg);
        }
    if (!c.contains("feature_extraction"))
        throw std::invalid_argument(where + " has no feature_extraction");
    // Read through a whole profile so the field table types and refuses every key.
    nlohmann::json shell = {{"feature_extraction", c["feature_extraction"]}};
    if (c.contains("training")) shell["training"] = c["training"];
    ThesisConfig tmp;
    std::set<std::string> fields;
    for_each_field(tmp,
        [&](auto& field, const char* /*when*/, auto /*applies*/)
        {
            const std::string path = field.path();
            if (path.rfind("feature_extraction.", 0) == 0 || path.rfind("training.", 0) == 0)
                fields.insert(path);
        });
    strict::refuse_unknown_keys(shell, "", fields, strict::sections_of(fields), {}, where);
    for_each_field(tmp,
        [&](auto& field, const char* /*when*/, auto /*applies*/)
        {
            using T = typename std::remove_cvref_t<decltype(field)>::value_type;
            if (const auto* v = strict::find_path(shell, field.path()))
                field = strict::read_exact<T>(*v, where, field.path());
        });
    return {tmp.feature_extraction, tmp.training};
}

auto parse(const nlohmann::json& j, const ThesisConfig::Embedding* embedding) -> ThesisConfig
{
    ThesisConfig cfg;
    if (embedding != nullptr) cfg.embedded_in = *embedding;

    std::set<std::string> fields;
    for_each_field(cfg,
        [&](auto& field, const char* /*when*/, auto /*applies*/) { fields.insert(field.path()); });
    fields.insert(std::string(kCandidatesPath)); // an array of objects, parsed below

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

    if (const auto* cands = strict::find_path(j, kCandidatesPath))
    {
        if (!cands->is_array())
            throw std::invalid_argument(kOwner() + ": \"" + std::string(kCandidatesPath) +
                                        "\" must be an array, got " + strict::describe(*cands));
        for (std::size_t i = 0; i < cands->size(); ++i)
            cfg.selection.extractor_candidates.push_back(parse_candidate((*cands)[i], i));
    }

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
    if (!dg.profile.declared()) return; // not degraded (check_declarations)
    if (dg.scope != "probes" && dg.scope != "test")
        throw std::invalid_argument("ThesisConfig: degradation.scope must be probes or test");
    if (dg.seed == 0u)
        throw std::invalid_argument("ThesisConfig: degradation.seed must be non-zero");
    if (dg.profile == "whisper_mix")
    {
        if (!(dg.level > 0.0f && dg.level <= 1.0f))
            throw std::invalid_argument("ThesisConfig: degradation.level must be in (0, 1]");
    }
    else
    {
        // The F0 profiles: the limits of every value live in VoiceDysphonia (one place); an
        // unknown profile name is refused there too, naming all the valid ones.
        check_dysphonia_params(dysphonia_params(dg));
    }
    // The degraded features are extracted once, beside the clean ones. An autoencoder is
    // trained per repetition on the clean signals, so it has no single "degraded" twin.
    if (cfg.feature_extraction.strategy.declared() &&
        cfg.feature_extraction.strategy == "autoencoder")
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

void validate_selection(const ThesisConfig& cfg)
{
    const auto& s = cfg.selection;
    if (!s.rank_modality.declared()) return; // not the per-fold protocol (check_declarations)
    const std::string& modality = cfg.dataset.modality;
    const std::string expected = modality == "fused" ? "voice" : modality;
    if (s.rank_modality.get() != expected)
        throw std::invalid_argument(
            "ThesisConfig: selection.rank_modality must be \"" + expected + "\" for modality \"" +
            modality +
            "\" (a profile ranks its own signal; a fused one ranks the voice, whose winner feeds "
            "both parts)");
    // The whole-dataset ranking needs one extractor for every sample; here each fold has its
    // own, ranked inside the fold (FoldResult::selection carries every D_pen).
    if (cfg.paraconsistent.enabled)
        throw std::invalid_argument(
            "ThesisConfig: paraconsistent.enabled must be false with selection.enabled: the "
            "extractor is ranked inside each fold, there is no single feature set to rank");
    if (s.extractor_candidates.empty())
        throw std::invalid_argument(
            "ThesisConfig: selection.extractor_candidates must list at least one extractor");
    // Every candidate must be a valid Phase 00 profile of the ranked signal, and of the
    // profile's own modality (the winner extracts that one).
    for (std::size_t i = 0; i < s.extractor_candidates.size(); ++i)
    {
        (void) cfg.candidate_config(i, s.rank_modality.get());
        if (modality != s.rank_modality.get()) (void) cfg.candidate_config(i, modality);
    }
    if (!s.inner_k.declared()) return; // cosine: no hyperparameter search
    if (s.inner_k < 2) throw std::invalid_argument("ThesisConfig: selection.inner_k must be >= 2");
    const auto refuse_list = [](const char* name, bool bad, const std::string& why)
    {
        if (bad)
            throw std::invalid_argument(std::string("ThesisConfig: selection.") + name + " " + why);
    };
    const auto distinct = [](auto v)
    {
        std::sort(v.begin(), v.end());
        return std::adjacent_find(v.begin(), v.end()) == v.end();
    };
    const auto& w = *s.hidden_widths;
    const auto& b = *s.residual_blocks;
    const auto& lr = *s.learning_rates;
    const auto& wd = *s.weight_decays;
    refuse_list("hidden_widths", w.empty() || !distinct(w), "must be non-empty and distinct");
    refuse_list("residual_blocks", b.empty() || !distinct(b), "must be non-empty and distinct");
    refuse_list("learning_rates", lr.empty() || !distinct(lr), "must be non-empty and distinct");
    refuse_list("weight_decays", wd.empty() || !distinct(wd), "must be non-empty and distinct");
    refuse_list("hidden_widths",
        std::any_of(w.begin(), w.end(), [](int x) { return x < 1; }),
        "must be >= 1");
    refuse_list("residual_blocks",
        std::any_of(b.begin(), b.end(), [](int x) { return x < 0; }),
        "must be >= 0 (blocks after the input stage)");
    refuse_list("learning_rates",
        std::any_of(lr.begin(), lr.end(), [](float x) { return !(x > 0.0f); }),
        "must be > 0");
    refuse_list("weight_decays",
        std::any_of(wd.begin(), wd.end(), [](float x) { return !(x >= 0.0f); }),
        "must be >= 0");
}

void validate_training(const ThesisConfig::Training& training)
{
    if (!training.epochs.declared()) return; // no training runs (check_declarations)

    if (training.epochs <= 0)
        throw std::invalid_argument("ThesisConfig: training.epochs must be > 0");
    if (training.learning_rate.declared() && training.learning_rate <= 0.0f)
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
    if (selects(*this) != !selection.extractor_candidates.empty())
        throw std::invalid_argument(
            selects(*this)
                ? kOwner() +
                      ": the profile does not declare \"selection.extractor_candidates\", "
                      "which this run reads (selection.enabled is true). There is no "
                      "default: declare it."
                : kOwner() +
                      ": the profile declares \"selection.extractor_candidates\", which "
                      "this run never reads (it is read only when selection.enabled is "
                      "true). Remove it.");
}

void ThesisConfig::validate() const
{
    check_declarations();

    validate_experiment(experiment);
    validate_dataset(dataset);

    if (feature_extraction.strategy.declared()) // not the per-fold protocol
    {
        const std::string& strategy = feature_extraction.strategy;
        if (strategy == "handcrafted")
            validate_handcrafted(feature_extraction.handcrafted, dataset);
        else if (strategy == "autoencoder")
            validate_autoencoder(feature_extraction.autoencoder);
        else if (strategy == "pca")
            validate_pca(feature_extraction.pca);
        else
            throw std::invalid_argument(
                "ThesisConfig: feature_extraction.strategy must be handcrafted, autoencoder, or "
                "pca; got \"" +
                strategy + "\"");
    }
    validate_selection(*this);

    validate_paraconsistent(paraconsistent);
    validate_classifier(classifier);
    validate_degradation(*this);
    validate_training(training);

    // The cosine control has no model to select, so a nested split would only repeat the
    // flat one under another name.
    if (classifier.enabled && classifier.type == "cosine" && training.nested_cv.declared() &&
        training.nested_cv)
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
    if (!selection.extractor_candidates.empty())
    {
        nlohmann::json cands = nlohmann::json::array();
        for (const auto& cand : selection.extractor_candidates)
        {
            ThesisConfig tmp;
            tmp.feature_extraction = cand.feature_extraction;
            tmp.training = cand.training;
            nlohmann::json one = nlohmann::json::object();
            for_each_field(tmp,
                [&](const auto& field, const char* /*when*/, auto /*applies*/)
                {
                    if (!field.declared()) return;
                    std::string pointer = std::string("/") + field.path();
                    std::replace(pointer.begin(), pointer.end(), '.', '/');
                    one[nlohmann::json::json_pointer(pointer)] = field.get();
                });
            cands.push_back(std::move(one));
        }
        j["selection"]["extractor_candidates"] = std::move(cands);
    }
    return j;
}

auto ThesisConfig::candidate_config(std::size_t i, const std::string& modality) const
    -> ThesisConfig
{
    if (i >= selection.extractor_candidates.size())
        throw std::out_of_range("ThesisConfig: no extractor candidate " + std::to_string(i));
    if (modality != "voice" && modality != "eeg" && modality != "fused")
        throw std::invalid_argument("ThesisConfig: candidate modality must be voice/eeg/fused");
    const auto& cand = selection.extractor_candidates[i];
    ThesisConfig c;
    c.experiment.run_tag = experiment.run_tag.get() + "-candidate" + std::to_string(i);
    c.experiment.seed = experiment.seed.get();
    c.experiment.repeats = 1;
    c.dataset.source = dataset.source.get();
    c.dataset.root = dataset.root.get();
    c.dataset.results_dir = dataset.results_dir.get();
    c.dataset.modality = modality;
    c.dataset.max_samples = dataset.max_samples.get();
    if (modality == "fused") c.dataset.fusion_mode = dataset.fusion_mode.get();
    if (modality != "voice")
    {
        c.dataset.eeg_condition = dataset.eeg_condition.get();
        if (dataset.eeg_condition.get() == "imagined")
            c.dataset.eeg_pairing_seed = dataset.eeg_pairing_seed.get();
    }
    c.feature_extraction = cand.feature_extraction;
    c.training = cand.training;
    c.paraconsistent.enabled = false;
    c.classifier.enabled = false;
    try
    {
        c.validate();
    }
    catch (const std::exception& e)
    {
        throw std::invalid_argument("ThesisConfig: selection.extractor_candidates[" +
                                    std::to_string(i) + "] is not a valid " + modality +
                                    " extractor: " + e.what());
    }
    return c;
}

} // namespace thesis
