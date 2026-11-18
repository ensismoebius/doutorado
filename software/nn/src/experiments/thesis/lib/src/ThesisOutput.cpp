#include "ThesisOutput.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace thesis
{

void ensure_dir(const std::string& path)
{
    std::filesystem::create_directories(path);
}

void write_metrics_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_metrics.csv";
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("ThesisOutput: cannot write " + path);

    f << "feature_set,classifier,text_mode,fold,"
      << "accuracy,f1,precision,recall,specificity,eer,auc,train_ms,infer_ms,model_path\n";
    for (const auto& r : results)
    {
        for (const auto& fold : r.outer_folds)
        {
            f << r.feature_set_label << "," << r.classifier_type << "," << r.text_mode << ","
              << fold.fold << "," << std::fixed << std::setprecision(6) << fold.accuracy << ","
              << fold.f1 << "," << fold.precision << "," << fold.recall << "," << fold.specificity
              << "," << fold.eer << "," << fold.auc << "," << std::setprecision(3) << fold.train_ms
              << "," << fold.infer_ms << "," << fold.model_path << "\n";
        }
    }
}

void write_paraconsistent_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ParaconsistentScore>& scores)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_paraconsistent.csv";
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("ThesisOutput: cannot write " + path);

    f << "label,alpha,beta,g1,g2,d_truth,d_penalized\n";
    for (const auto& s : scores)
    {
        f << s.label << "," << std::fixed << std::setprecision(8) << s.alpha << "," << s.beta << ","
          << s.g1 << "," << s.g2 << "," << s.d_truth << "," << s.d_penalized << "\n";
    }
}

namespace
{

// Run identity plus the configuration the run actually used — recorded so a summary file
// is self-describing and reproducible from itself.
//
// `config` is every key the profile declared, which (ThesisConfig refuses the rest) is
// exactly every key the run read. It replaces a hand-picked subset that had to be kept in
// step with the config by hand and once printed a learning rate the profile never stated:
// the lr used to be optional and resolved from the optimizer, so the summary recorded the
// resolved value and a `learning_rate_source` flag (.wiki/Guides/Engineering-Fixes-Log.md
// D3). With every key required, the declared value IS the value that ran.
auto build_run_metadata_json(const ThesisConfig& cfg) -> nlohmann::json
{
    nlohmann::json j;
    j["run_tag"] = cfg.experiment.run_tag;
    j["seed"] = cfg.experiment.seed;
    j["modality"] = cfg.dataset.modality;
    // The per-fold protocol names no extractor of its own (each fold chooses one, recorded in
    // its fold entry), so it has no "strategy": a reader that expects one fails loudly instead
    // of filing a per-fold run under some extractor.
    const bool per_fold = cfg.classifier.enabled && cfg.selection.enabled;
    j["protocol"] = per_fold ? "per-fold" : "fixed";
    if (!per_fold) j["strategy"] = cfg.feature_extraction.strategy;
    if (cfg.classifier.enabled)
    {
        j["classifier"] = cfg.classifier.type;
        j["text_mode"] = cfg.classifier.text_mode;
    }
    j["config"] = cfg.declared_json();
    return j;
}

// Inference energy per test sample (nJ), FoldResult.hpp. Absent for the cosine control.
auto energy_json(const InferenceEnergy& e) -> nlohmann::json
{
    return {{"nj", e.nj},
        {"ops_nj", e.ops_nj},
        {"mem_nj", e.mem_nj},
        {"nj_as_executed", e.nj_as_executed},
        {"nj_sram8k", e.nj_sram8k},
        {"nj_sram32k", e.nj_sram32k},
        {"nj_sram1m", e.nj_sram1m},
        {"fp_macs", e.fp_macs},
        {"fp_adds", e.fp_adds},
        {"accesses", e.accesses},
        {"spikes", e.spikes}};
}

// What the per-fold protocol chose in one fold, with everything it chose among. The cosine
// control tunes nothing: its entry has no grid keys at all (not an empty grid).
auto selection_json(const FoldSelection& s) -> nlohmann::json
{
    nlohmann::json ex = nlohmann::json::array();
    for (const auto& e : s.extractors)
        ex.push_back({{"label", e.label}, {"d_penalized", e.d_penalized}});
    nlohmann::json j{{"extractors", ex},
        {"extractor_index", s.extractor_index},
        {"extractor_label", s.extractors.at(s.extractor_index).label},
        {"param_count", s.param_count},
        {"fold_ms", s.fold_ms}};
    if (s.grid.empty()) return j;
    nlohmann::json grid = nlohmann::json::array();
    for (const auto& p : s.grid)
        grid.push_back({{"hidden_width", p.hidden_width},
            {"residual_blocks", p.residual_blocks},
            {"learning_rate", p.learning_rate},
            {"weight_decay", p.weight_decay},
            {"param_count", p.param_count},
            {"inner_eer", p.inner_eer},
            {"mean_inner_eer", p.mean_inner_eer}});
    j["grid"] = grid;
    j["grid_index"] = s.grid_index;
    return j;
}

auto build_fold_entry_json(const FoldResult& fold) -> nlohmann::json
{
    nlohmann::json fj;
    fj["fold"] = fold.fold;
    fj["model_path"] = fold.model_path;
    fj["train_ms"] = fold.train_ms;
    fj["infer_ms"] = fold.infer_ms;
    fj["epochs_run"] = fold.history.size();
    if (!fold.history.empty())
    {
        const auto& last = fold.history.back();
        fj["final_train_loss"] = last.train_loss;
        if (!std::isnan(last.val_loss)) fj["final_val_loss"] = last.val_loss;
    }
    if (fold.energy) fj["energy"] = energy_json(*fold.energy);
    if (fold.selection) fj["selection"] = selection_json(*fold.selection);
    return fj;
}

auto build_result_entry_json(const ClassificationResult& r) -> nlohmann::json
{
    nlohmann::json rj;
    rj["feature_set"] = r.feature_set_label;
    rj["mean_accuracy"] = r.mean_accuracy;
    rj["std_accuracy"] = r.std_accuracy;
    rj["ci95_accuracy"] = r.ci95_accuracy;
    rj["mean_f1"] = r.mean_f1;
    rj["std_f1"] = r.std_f1;
    rj["mean_precision"] = r.mean_precision;
    rj["mean_recall"] = r.mean_recall;
    rj["mean_specificity"] = r.mean_specificity;
    rj["std_specificity"] = r.std_specificity;
    rj["mean_eer"] = r.mean_eer;
    rj["std_eer"] = r.std_eer;
    rj["ci95_eer"] = r.ci95_eer;
    rj["mean_auc"] = r.mean_auc;
    rj["std_auc"] = r.std_auc;

    // Run cost / complexity (Meeting01-style): model size + mean per-fold wall-clock.
    if (r.param_count) rj["param_count"] = *r.param_count;
    rj["mean_train_ms"] = r.mean_train_ms;
    rj["mean_infer_ms"] = r.mean_infer_ms;
    if (!std::isnan(r.mean_spike_rate)) // SNN classifiers only
    {
        rj["mean_spike_rate"] = r.mean_spike_rate;
        rj["final_sops"] = r.final_sops;
    }

    nlohmann::json folds_arr = nlohmann::json::array();
    for (const auto& fold : r.outer_folds)
    {
        folds_arr.push_back(build_fold_entry_json(fold));
    }
    rj["fold_models"] = folds_arr;
    if (r.mean_energy) rj["energy"] = energy_json(*r.mean_energy);
    return rj;
}

} // namespace

void write_summary_json(const std::string& results_dir,
    const std::string& run_tag,
    const ThesisConfig& cfg,
    const std::vector<ClassificationResult>& results,
    const std::vector<ParaconsistentScore>& scores,
    int n_subjects,
    int n_stimuli,
    size_t n_samples,
    std::size_t config_hash)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_summary.json";

    nlohmann::json j = build_run_metadata_json(cfg);
    j["config_hash"] = config_hash; // provenance/determinism (Meeting01 parity)

    // Dataset composition actually fed to this run — after load_dataset drops
    // trials missing either audio or EEG (paired-samples guarantee), so the
    // count here can be lower than the raw .mat trial count.
    j["dataset"] = {
        {"n_subjects", n_subjects},
        {"n_stimuli", n_stimuli},
        {"n_samples", n_samples},
    };

    nlohmann::json results_arr = nlohmann::json::array();
    for (const auto& r : results)
    {
        results_arr.push_back(build_result_entry_json(r));
    }
    j["results"] = results_arr;

    if (!scores.empty())
    {
        j["best_feature_set"] = scores[0].label;
        j["best_d_truth"] = scores[0].d_truth;
        j["best_d_penalized"] = scores[0].d_penalized;
        j["best_alpha"] = scores[0].alpha;
        j["best_beta"] = scores[0].beta;
    }

    // NOTE: no in-run significance test (Meeting01 records SNN-vs-LSTM t-test/Wilcoxon/Cohen's d
    // because it trains both families in one process). An Thesis run scores exactly one feature
    // set, so the analogous comparison is cross-PROFILE and belongs in a post-hoc aggregation
    // step over these summary files, not here.

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("ThesisOutput: cannot write " + path);
    f << j.dump(2) << "\n";
}

namespace
{
// true when every fold of every result carries a selection, false when none does; a mix is a
// bug of the caller (one run is one protocol).
auto has_selection(const std::vector<ClassificationResult>& results) -> bool
{
    std::size_t with = 0, total = 0;
    for (const auto& r : results)
        for (const auto& fold : r.outer_folds)
        {
            ++total;
            if (fold.selection) ++with;
        }
    if (with != 0 && with != total)
        throw std::logic_error("ThesisOutput: " + std::to_string(with) + " of " +
                               std::to_string(total) + " folds carry a per-fold selection");
    return with != 0;
}
} // namespace

void write_selection_csvs(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results)
{
    if (!has_selection(results))
        throw std::logic_error("ThesisOutput: write_selection_csvs on a fixed-configuration run");
    ensure_dir(results_dir);
    const std::string ex_path = results_dir + "/e05_" + run_tag + "_selection_extractors.csv";
    std::ofstream ex(ex_path);
    if (!ex.is_open()) throw std::runtime_error("ThesisOutput: cannot write " + ex_path);
    const std::string grid_path = results_dir + "/e05_" + run_tag + "_selection_grid.csv";
    std::ofstream grid(grid_path);
    if (!grid.is_open()) throw std::runtime_error("ThesisOutput: cannot write " + grid_path);

    ex << "feature_set,fold,candidate,label,d_penalized,chosen\n";
    grid << "feature_set,fold,point,hidden_width,residual_blocks,learning_rate,weight_decay,"
            "param_count,inner_fold,inner_eer,mean_inner_eer,chosen\n";
    for (const auto& r : results)
        for (const auto& fold : r.outer_folds)
        {
            if (!fold.selection) throw std::logic_error("ThesisOutput: a fold lost its selection");
            const FoldSelection& s = *fold.selection;
            for (std::size_t i = 0; i < s.extractors.size(); ++i)
                ex << r.feature_set_label << "," << fold.fold << "," << i << ","
                   << s.extractors[i].label << "," << std::setprecision(10)
                   << s.extractors[i].d_penalized << "," << (i == s.extractor_index ? 1 : 0)
                   << "\n";
            for (std::size_t g = 0; g < s.grid.size(); ++g)
            {
                const GridPoint& p = s.grid[g];
                for (std::size_t k = 0; k < p.inner_eer.size(); ++k)
                    grid << r.feature_set_label << "," << fold.fold << "," << g << ","
                         << p.hidden_width << "," << p.residual_blocks << ","
                         << std::setprecision(8) << p.learning_rate << "," << p.weight_decay << ","
                         << p.param_count << "," << k << "," << std::setprecision(10)
                         << p.inner_eer[k] << "," << p.mean_inner_eer << ","
                         << (g == s.grid_index ? 1 : 0) << "\n";
            }
        }
}

void write_comparison_dat(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_comparison.dat";
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("ThesisOutput: cannot write " + path);

    f << "x label accuracy std_accuracy ci95_accuracy"
      << " f1 std_f1 precision recall specificity std_specificity"
      << " eer std_eer ci95_eer auc std_auc\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        f << i << " " << r.feature_set_label << " " << std::fixed << std::setprecision(6)
          << r.mean_accuracy << " " << r.std_accuracy << " " << r.ci95_accuracy << " " << r.mean_f1
          << " " << r.std_f1 << " " << r.mean_precision << " " << r.mean_recall << " "
          << r.mean_specificity << " " << r.std_specificity << " " << r.mean_eer << " " << r.std_eer
          << " " << r.ci95_eer << " " << r.mean_auc << " " << r.std_auc << "\n";
    }
}

void write_learning_curves_dat(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results)
{
    // Nothing trained (e.g. all cache hits, or a phase00 ranking-only run) → no file.
    bool any = false;
    for (const auto& r : results)
        for (const auto& fold : r.outer_folds)
            if (!fold.history.empty()) any = true;
    if (!any) return;

    ensure_dir(results_dir);
    const std::string path = results_dir + "/e05_" + run_tag + "_learning_curves.dat";
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("ThesisOutput: cannot write " + path);

    // spike_rate/sops are NaN/0 for non-spiking classifiers; the columns are kept
    // so every run has the same schema (pgfplots-friendly).
    f << "feature_set fold epoch train_loss val_loss epoch_ms spike_rate sops\n";
    for (const auto& r : results)
    {
        for (const auto& fold : r.outer_folds)
        {
            for (const auto& e : fold.history)
            {
                f << r.feature_set_label << " " << fold.fold << " " << e.epoch << " " << std::fixed
                  << std::setprecision(6) << e.train_loss << " " << e.val_loss << " "
                  << std::setprecision(3) << e.epoch_ms << " " << e.mean_spike_rate << " " << e.sops
                  << "\n";
            }
        }
    }
}

} // namespace thesis
