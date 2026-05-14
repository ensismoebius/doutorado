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
    if (!f.is_open())
        throw std::runtime_error("E05Output: cannot write " + path);

    f << "feature_set,classifier,text_mode,fold,accuracy,f1,precision,recall,eer,auc\n";
    for (const auto& r : results)
    {
        for (const auto& fold : r.outer_folds)
        {
            f << r.feature_set_label << ","
              << r.classifier_type << ","
              << r.text_mode << ","
              << fold.fold << ","
              << std::fixed << std::setprecision(6)
              << fold.accuracy  << ","
              << fold.f1        << ","
              << fold.precision << ","
              << fold.recall    << ","
              << fold.eer       << ","
              << fold.auc       << "\n";
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
    if (!f.is_open())
        throw std::runtime_error("E05Output: cannot write " + path);

    f << "label,alpha,beta,g1,g2,d_truth\n";
    for (const auto& s : scores)
    {
        f << s.label << ","
          << std::fixed << std::setprecision(8)
          << s.alpha << "," << s.beta << ","
          << s.g1    << "," << s.g2    << ","
          << s.d_truth << "\n";
    }
}

void write_summary_json(const std::string& results_dir,
    const std::string& run_tag,
    const E05Config& cfg,
    const std::vector<ClassificationResult>& results,
    const std::vector<ParaconsistentScore>& scores)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_summary.json";

    nlohmann::json j;
    j["run_tag"]   = cfg.experiment.run_tag;
    j["seed"]      = cfg.experiment.seed;
    j["modality"]  = cfg.dataset.modality;
    j["strategy"]  = cfg.feature_extraction.strategy;
    j["classifier"] = cfg.classifier.type;
    j["text_mode"] = cfg.classifier.text_mode;

    nlohmann::json results_arr = nlohmann::json::array();
    for (const auto& r : results)
    {
        nlohmann::json rj;
        rj["feature_set"]     = r.feature_set_label;
        rj["mean_accuracy"]   = r.mean_accuracy;
        rj["std_accuracy"]    = r.std_accuracy;
        rj["ci95_accuracy"]   = r.ci95_accuracy;
        rj["mean_f1"]         = r.mean_f1;
        rj["std_f1"]          = r.std_f1;
        rj["mean_precision"]  = r.mean_precision;
        rj["mean_recall"]     = r.mean_recall;
        rj["mean_eer"]        = r.mean_eer;
        rj["std_eer"]         = r.std_eer;
        rj["ci95_eer"]        = r.ci95_eer;
        rj["mean_auc"]        = r.mean_auc;
        rj["std_auc"]         = r.std_auc;
        results_arr.push_back(rj);
    }
    j["results"] = results_arr;

    if (!scores.empty())
    {
        j["best_feature_set"]  = scores[0].label;
        j["best_d_truth"]      = scores[0].d_truth;
        j["best_alpha"]        = scores[0].alpha;
        j["best_beta"]         = scores[0].beta;
    }

    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("E05Output: cannot write " + path);
    f << j.dump(2) << "\n";
}

void write_comparison_dat(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results)
{
    ensure_dir(results_dir);
    std::string path = results_dir + "/e05_" + run_tag + "_comparison.dat";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("E05Output: cannot write " + path);

    // pgfplots DAT: whitespace-delimited, header row starts with x (label index)
    f << "x label accuracy std_accuracy ci95_accuracy f1 std_f1 precision recall eer std_eer ci95_eer auc std_auc\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        f << i << " "
          << r.feature_set_label << " "
          << std::fixed << std::setprecision(6)
          << r.mean_accuracy  << " "
          << r.std_accuracy   << " "
          << r.ci95_accuracy  << " "
          << r.mean_f1        << " "
          << r.std_f1         << " "
          << r.mean_precision << " "
          << r.mean_recall    << " "
          << r.mean_eer       << " "
          << r.std_eer        << " "
          << r.ci95_eer       << " "
          << r.mean_auc       << " "
          << r.std_auc        << "\n";
    }
}

} // namespace e05
