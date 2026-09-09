#pragma once
// GuayaquilEventCallback.hpp — training callback that turns Trainer lifecycle
// hooks into structured JSONL events (config_begin / epoch_progress / epoch /
// train_end) on the process-wide ExperimentEvents sink. Sibling of
// GuayaquilEpochLogger (which writes the human-readable [loso] stderr lines).
//
// epoch_progress is a *throttled* intra-epoch heartbeat: at most one line every
// kProgressEverySeconds. A fast epoch (< that) emits none; a slow one emits a
// batch-fraction update so the monitor can show a within-epoch bar + ETA.
//
// Identity + static config come from ExperimentEvents::pending_context(), set by
// the driver immediately before each training call — so this callback takes no
// constructor arguments and no experiment signature has to grow an event param.
//
// OBSERVABILITY ONLY: never influences training (no should_stop override, no
// mutation of state).

#include <chrono>
#include <limits>
#include <vector>

#include "GuayaquilEvents.hpp"
#include "core/training/EpochResult.hpp"
#include "training/ITrainingCallback.hpp"
#include "training/TrainingState.hpp"

namespace guayaquil
{

class GuayaquilEventCallback : public nn::training::ITrainingCallback
{
   public:
    GuayaquilEventCallback() : ctx_(ExperimentEvents::instance().pending_context()) {}

    void on_train_begin(int total_epochs) override
    {
        if (total_epochs > 0) ctx_.max_epochs = total_epochs;
        ExperimentEvents::instance().emit("config_begin",
            {{"config_id", ctx_.config_id},
                {"model", ctx_.model},
                {"encoding", ctx_.encoding},
                {"role", ctx_.role},
                {"hyperparams", ctx_.hyperparams},
                {"run_id", ctx_.run_id},
                {"seed", ctx_.seed},
                {"max_epochs", ctx_.max_epochs},
                {"lr", jnum(ctx_.lr)},
                {"lr_biophysical", jnum(ctx_.lr_biophysical)},
                {"early_stop_patience", ctx_.early_stop_patience},
                {"param_count", ctx_.param_count},
                {"macs", ctx_.macs}});
    }

    void on_epoch_begin(const nn::training::TrainingState& s) override
    {
        cur_epoch_ = s.epoch;
        epoch_start_ = std::chrono::steady_clock::now();
        last_progress_ = epoch_start_;
    }

    // Throttled intra-epoch heartbeat: fires at most once per kProgressEverySeconds,
    // so a fast epoch produces nothing and a slow one a handful of lines.
    void on_batch_end(const nn::training::TrainingState& s) override
    {
        if (s.total_batches <= 1) return;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_progress_ < std::chrono::seconds(kProgressEverySeconds)) return;
        last_progress_ = now;

        const double elapsed = std::chrono::duration<double>(now - epoch_start_).count();
        const double frac =
            s.total_batches > 0 ? static_cast<double>(s.batch) / s.total_batches : 0.0;
        const double eta = (frac > 1e-6) ? elapsed * (1.0 - frac) / frac : -1.0;

        ExperimentEvents::instance().emit("epoch_progress",
            {{"config_id", ctx_.config_id},
                {"epoch", cur_epoch_},
                {"max_epochs", ctx_.max_epochs},
                {"batch", s.batch},
                {"total_batches", s.total_batches},
                {"frac", frac},
                {"batch_loss", jnum(s.batch_loss)},
                {"epoch_elapsed_s", elapsed},
                {"epoch_eta_s", jnum(eta)}});
    }

    void on_epoch_end(
        const nn::training::TrainingState& /*state*/, const nn::training::EpochResult& r) override
    {
        ExperimentEvents::instance().emit("epoch",
            {{"config_id", ctx_.config_id},
                {"epoch", r.epoch},
                {"max_epochs", ctx_.max_epochs},
                {"train_loss", jnum(r.train_loss)},
                {"val_loss", jnum(r.val_loss)},
                {"epoch_ms", jnum(r.epoch_ms)},
                {"mean_spike_rate", jnum(r.mean_spike_rate)},
                {"sops", r.sops}});
    }

    void on_train_end(const std::vector<nn::training::EpochResult>& history) override
    {
        int best_epoch = 0;
        float best = std::numeric_limits<float>::infinity();
        for (const auto& e : history)
        {
            const float v = std::isnan(e.val_loss) ? e.train_loss : e.val_loss;
            if (v < best)
            {
                best = v;
                best_epoch = e.epoch;
            }
        }
        const bool early = static_cast<int>(history.size()) < ctx_.max_epochs;
        ExperimentEvents::instance().emit("train_end",
            {{"config_id", ctx_.config_id},
                {"epochs_run", static_cast<int>(history.size())},
                {"stop_reason", early ? "early_stop" : "max_epochs"},
                {"best_val_loss", jnum(best)},
                {"best_val_epoch", best_epoch}});
    }

   private:
    static constexpr int kProgressEverySeconds = 5;

    EventContext ctx_;
    int cur_epoch_ = 0;
    std::chrono::steady_clock::time_point epoch_start_{};
    std::chrono::steady_clock::time_point last_progress_{};
};

} // namespace guayaquil
