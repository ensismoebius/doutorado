#include "E05Output.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace e05
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
    if (!f.is_open()) throw std::runtime_error("E05Output: cannot write " + path);

    f << "feature_set,classifier,text_mode,fold,"
      << "accuracy,f1,precision,recall,specificity,eer,auc,model_path\n";
    for (const auto& r : results)
    {
        for (const auto& fold : r.outer_folds)
        {
            f << r.feature_set_label << "," << r.classifier_type << "," << r.text_mode << ","
              << fold.fold << "," << std::fixed << std::setprecision(6) << fold.accuracy << ","
              << fold.f1 << "," << fold.precision << "," << fold.recall << "," << fold.specificity
              << "," << fold.eer << "," << fold.auc << "," << fold.model_path << "\n";
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
    if (!f.is_open()) throw std::runtime_error("E05Output: cannot write " + path);

    f << "label,alpha,beta,g1,g2,d_truth,d_penalized\n";
    for (const auto& s : scores)
    {
        f << s.label << "," << std::fixed << std::setprecision(8) << s.alpha << "," << s.beta << ","
          << s.g1 << "," << s.g2 << "," << s.d_truth << "," << s.d_penalized << "\n";
    }
}

void write_summary_json(const std::string& results_dir,
    const std::string& run_tag,
    const E05Config& cfg,
    const std::vector<ClassificationResult>& results,
    const std::vector<ParaconsistentScore>& scores,
    int n_subjects,
    int n_stimuli,
    size_t n_samples)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_summary.json";

    nlohmann::json j;
    j["run_tag"] = cfg.experiment.run_tag;
    j["seed"] = cfg.experiment.seed;
    j["modality"] = cfg.dataset.modality;
    j["strategy"] = cfg.feature_extraction.strategy;
    j["classifier"] = cfg.classifier.type;
    j["text_mode"] = cfg.classifier.text_mode;

    // Dataset composition actually fed to this run — after load_dataset drops
    // trials missing either audio or EEG (paired-samples guarantee), so the
    // count here can be lower than the raw .mat trial count.
    j["dataset"] = {
        {"n_subjects", n_subjects},
        {"n_stimuli", n_stimuli},
        {"n_samples", n_samples},
    };

    // Training config actually used, recorded so a result file is self-describing and
    // reproducible from itself. `learning_rate` is the RESOLVED value
    // (Training::effective_learning_rate()): profiles may omit the field and inherit the
    // optimizer's reference default, which differs ~10x between optimizers, so the declared
    // profile is not always enough to know what ran. `learning_rate_source` says which it
    // was. This exists because of fixme.md D3, where the published numbers had been produced
    // at an effective lr 10x below the one the profiles declared, with nothing on disk
    // recording the discrepancy.
    j["training"] = {
        {"optimizer_type", cfg.training.optimizer_type},
        {"learning_rate", cfg.training.effective_learning_rate()},
        {"learning_rate_source",
            cfg.training.learning_rate.has_value() ? "profile" : "optimizer_default"},
        {"epochs", cfg.training.epochs},
        {"samples_per_batch", cfg.training.samples_per_batch},
        {"weight_decay", cfg.training.weight_decay},
    };
    if (cfg.training.optimizer_type == "sgd")
    {
        j["training"]["optimizer_momentum"] = cfg.training.optimizer_momentum;
    }

    // Feature-extraction config actually used to produce this run's vectors —
    // recorded so a result file is self-describing without cross-referencing
    // the source profile (e.g. distinguishing the 18 SNN-AE poisson/latency/
    // direct × tiny/small/base variants, which otherwise share the same
    // "autoencoder-snn" FeatureSet label in the paraconsistent CSV).
    if (cfg.feature_extraction.strategy == "handcrafted")
    {
        const auto& hc = cfg.feature_extraction.handcrafted;
        j["handcrafted"] = {
            {"wavelet", hc.wavelet},
            {"scale", hc.scale},
            {"cepstral", hc.cepstral},
            {"dtwpt_level", hc.dtwpt_level},
            {"descriptors", hc.descriptors},
        };
    }
    else if (cfg.feature_extraction.strategy == "autoencoder")
    {
        const auto& ae = cfg.feature_extraction.autoencoder;
        j["autoencoder"] = {
            {"model", ae.model},
            {"encoder_layer_spec", ae.encoder_layer_spec},
            {"decoder_layer_spec", ae.decoder_layer_spec},
        };
        if (ae.model == "snn-ae")
        {
            j["autoencoder"]["encoding"] = ae.encoding;
            j["autoencoder"]["time_steps"] = ae.encoding == "direct" ? 1 : ae.time_steps;
            j["autoencoder"]["voltage_threshold"] = ae.voltage_threshold;
        }
    }

    nlohmann::json results_arr = nlohmann::json::array();
    for (const auto& r : results)
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

        nlohmann::json folds_arr = nlohmann::json::array();
        for (const auto& fold : r.outer_folds)
        {
            nlohmann::json fj;
            fj["fold"] = fold.fold;
            fj["model_path"] = fold.model_path;
            folds_arr.push_back(fj);
        }
        rj["fold_models"] = folds_arr;
        results_arr.push_back(rj);
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

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("E05Output: cannot write " + path);
    f << j.dump(2) << "\n";
}

void write_comparison_dat(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_comparison.dat";
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("E05Output: cannot write " + path);

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

} // namespace e05
