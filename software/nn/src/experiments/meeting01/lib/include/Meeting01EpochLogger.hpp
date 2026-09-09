#pragma once
// Meeting01EpochLogger.hpp — training callback that emits one plain log line per
// epoch for the individual model being trained.
//
// The ProgressManager's live bars use absolute-cursor ANSI control and collapse
// to a single rewriting line when stdout is not a terminal (nohup, `> file`,
// `tail -f`). This callback writes newline-terminated progress to the logger
// (stderr) instead, so a nested-LOSO run's log keeps a scrollable per-epoch
// history of every training: which model / encoding / config, which epoch, and
// the train / validation loss.

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "core/training/EpochResult.hpp"
#include "logging/Logger.hpp"
#include "training/ITrainingCallback.hpp"

namespace meeting01
{

// "<dataset> fold<f>  run <n>/<total>  seed=<s>" — the coordinates of one
// individual training within a nested-LOSO fold process.
inline auto progress_context(
    const Meeting01Config& cfg, std::size_t run_id, std::size_t total_runs, std::uint32_t seed)
    -> std::string
{
    const std::string ds =
        cfg.evaluation.datasets.empty() ? std::string("data") : cfg.evaluation.datasets.front();
    std::ostringstream ss;
    ss << ds;
    if (cfg.dataset.cv_fold >= 0) ss << " fold" << cfg.dataset.cv_fold;
    ss << "  run " << (run_id + 1) << "/" << total_runs << "  seed=" << seed;
    return ss.str();
}

class Meeting01EpochLogger : public nn::training::ITrainingCallback
{
   public:
    // `label` identifies the individual training (e.g. "SNN-recurrent: encoding=poisson
    // v=1.50 a=0.99"); `context` prefixes it with the run coordinates
    // (e.g. "mitbih fold2 run 137/450").
    Meeting01EpochLogger(std::string context, std::string label)
        : context_(std::move(context)), label_(std::move(label))
    {
    }

    void on_train_begin(int total_epochs) override
    {
        total_epochs_ = total_epochs;
        NN_LOG_INFO("[loso] " + context_ + "  START  " + label_ +
                    "  (<= " + std::to_string(total_epochs) + " epochs)");
    }

    void on_epoch_end(
        const nn::training::TrainingState& /*state*/, const nn::training::EpochResult& r) override
    {
        std::ostringstream ss;
        ss << "[loso] " << context_ << "  epoch " << r.epoch << "/" << total_epochs_
           << "  train=" << fmt(r.train_loss);
        if (!std::isnan(r.val_loss)) ss << "  val=" << fmt(r.val_loss);
        NN_LOG_INFO(ss.str());
    }

    void on_train_end(const std::vector<nn::training::EpochResult>& history) override
    {
        std::ostringstream ss;
        ss << "[loso] " << context_ << "  DONE   " << label_ << "  " << history.size() << " epochs";
        if (!history.empty())
        {
            ss << "  final train=" << fmt(history.back().train_loss);
            if (!std::isnan(history.back().val_loss))
                ss << "  val=" << fmt(history.back().val_loss);
        }
        NN_LOG_INFO(ss.str());
    }

   private:
    static auto fmt(float v) -> std::string
    {
        std::ostringstream o;
        o << std::fixed << std::setprecision(6) << v;
        return o.str();
    }

    std::string context_;
    std::string label_;
    int total_epochs_ = 0;
};

} // namespace meeting01
