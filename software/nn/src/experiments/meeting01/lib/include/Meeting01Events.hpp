#pragma once
// Meeting01Events.hpp — process-wide append-only JSONL event sink for one
// (dataset, outer fold) run. Consumed live by
// scripts/pipeline/meeting01/monitor.py.
//
// OBSERVABILITY ONLY. Emitting an event never gates, blocks, or alters training,
// the split, the metrics, the seeds, or any persisted CSV/JSON output — it is
// extra side output. Raw values are written at full precision; every bit of
// rounding / formatting lives in the Python renderer.
//
// One process owns one file (truncate-on-open), so a re-run of a single fold
// replaces its own events cleanly. Schema version: 1.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "nlohmann/json.hpp"

namespace meeting01
{

// NaN / Inf → JSON null; otherwise the finite value. Keeps unmeasured metrics
// (e.g. EpochResult::mean_spike_rate for ANN models) explicit rather than
// leaning on the serializer's NaN handling.
[[nodiscard]] inline auto jnum(double v) -> nlohmann::json
{
    if (!std::isfinite(v)) return nullptr;
    return v;
}

// Stable, human-readable identity for one trained model instance. Matches between
// the driver's `config_end` and the callback's `config_begin`/`epoch`/`train_end`.
[[nodiscard]] inline auto make_config_id(const std::string& model,
    const std::string& encoding,
    const std::string& role,
    float v_th,
    float alpha,
    std::uint32_t seed,
    int run_id) -> std::string
{
    char buf[192];
    if (model == "snn-ae")
        std::snprintf(buf,
            sizeof(buf),
            "%s_%s_%s_v%.2f_a%.2f_seed%u_run%d",
            model.c_str(),
            encoding.c_str(),
            role.c_str(),
            static_cast<double>(v_th),
            static_cast<double>(alpha),
            seed,
            run_id);
    else
        std::snprintf(
            buf, sizeof(buf), "%s_%s_seed%u_run%d", model.c_str(), encoding.c_str(), seed, run_id);
    return buf;
}

// Identity + static configuration of one training, carried from the driver (which
// knows model/role/hyperparameters) to the in-training callback via the singleton's
// pending slot. No signature churn through train_ae / train_with_early_stopping_snn.
//
// ONE CONTEXT, ONE TRAINING. The callback takes the slot (take_pending_context empties it),
// and with the sink open a training whose slot is empty throws. Before, the slot was only
// read, and only the final fits set it, so every GA evaluation after a final inherited that
// final's identity: in job 5315335, 11 of the Transformer-AE's FSDD search trainings were
// logged as "transformer-ae_direct_seed42_run1" with its hyperparameters. Silent: nothing
// downstream of the events log reads those fields to compute a result, so no number moved.
struct EventContext
{
    std::string config_id;
    std::string model;    // "lstm-ae" | "gru-ae" | "transformer-ae" | "snn-ae" | "ann-ae"
    std::string encoding; // "direct" | "poisson" | "latency"
    std::string role;     // "ga_eval" | "<arch>_final" | "snn_final"
    nlohmann::json hyperparams = nlohmann::json::object();
    int run_id = 0;
    std::uint32_t seed = 0u;
    int max_epochs = 0;
    float lr = 0.0f;
    float lr_biophysical = 0.0f;
    int early_stop_patience = -1;
    std::size_t param_count = 0u;
    std::size_t macs = 0u;
};

// The identity of one GA evaluation: "<model>_ga_<genome key>_seed<seed>", role "ga_eval".
// The genome key is the search cache's own (genome_key), so a config_id names exactly one
// cached evaluation.
[[nodiscard]] inline auto ga_eval_context(const std::string& model,
    const std::string& encoding,
    const std::string& genome_key,
    nlohmann::json hyperparams,
    std::uint32_t seed,
    int max_epochs,
    float lr,
    float lr_biophysical,
    int early_stop_patience) -> EventContext
{
    if (model.empty() || genome_key.empty())
        throw std::invalid_argument("ga_eval_context: empty " +
                                    std::string(model.empty() ? "model" : "genome key") +
                                    "; a GA evaluation needs both to be identified in the log");
    EventContext c;
    c.config_id = model + "_ga_" + genome_key + "_seed" + std::to_string(seed);
    c.model = model;
    c.encoding = encoding;
    c.role = "ga_eval";
    c.hyperparams = std::move(hyperparams);
    c.seed = seed;
    c.max_epochs = max_epochs;
    c.lr = lr;
    c.lr_biophysical = lr_biophysical;
    c.early_stop_patience = early_stop_patience;
    return c;
}

class ExperimentEvents
{
   public:
    static auto instance() -> ExperimentEvents&;

    ExperimentEvents(const ExperimentEvents&) = delete;
    auto operator=(const ExperimentEvents&) -> ExperimentEvents& = delete;

    // Truncate + open the events file. An empty path leaves the sink disabled
    // (legacy pooled runs / results_dir unset) — every emit() is then a no-op.
    void open(const std::string& path);
    [[nodiscard]] auto is_open() const -> bool;

    // Close and re-open the same path in APPEND mode. Call before the terminal
    // events (fold_end / session_end / session_error): a long-lived ofstream
    // keeps writing to its original inode, so if an external process replaced
    // the file on disk mid-run (a `git checkout` / `git stash` / `git clean` on
    // a tracked events file — see .gitignore) every later write silently went
    // to the now-unlinked inode and the on-disk log froze. Re-opening by path
    // re-attaches to whatever inode currently lives there so at least the run's
    // terminal state is recorded. No-op when the sink is disabled.
    void reopen_append();

    // Fields merged into every subsequent line: {v, run_tag, dataset, fold}.
    void set_common(nlohmann::json common);

    // Slot taken by the Meeting01EventCallback constructor. Set right before each training
    // call (every GA evaluation and every final fit); taking it leaves it empty, so a
    // context serves exactly one training.
    void set_pending_context(EventContext ctx);
    [[nodiscard]] auto take_pending_context() -> EventContext;

    // Emit one line: {<common>, ts_unix, type, <fields>}. Thread-safe; flushes.
    void emit(const std::string& type, const nlohmann::json& fields = nlohmann::json::object());

    void close();

   private:
    ExperimentEvents() = default;
    ~ExperimentEvents();

    mutable std::mutex mu_;
    std::ofstream out_;
    std::string path_;
    bool warned_closed_ = false;
    nlohmann::json common_ = nlohmann::json::object();
    EventContext pending_;
};

} // namespace meeting01
