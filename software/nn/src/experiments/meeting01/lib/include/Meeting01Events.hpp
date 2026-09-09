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
#include <string>

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
struct EventContext
{
    std::string config_id;
    std::string model;    // "lstm-ae" | "gru-ae" | "transformer-ae" | "snn-ae"
    std::string encoding; // "direct" | "poisson" | "latency"
    std::string role;     // "baseline" | "snn_sweep" | "snn_final"
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

    // Slot read by Meeting01EventCallback::on_train_begin. Set by the driver right
    // before each training call.
    void set_pending_context(EventContext ctx);
    [[nodiscard]] auto pending_context() const -> EventContext;

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
