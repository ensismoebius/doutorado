#include "../include/Meeting01RecurrentGaFitness.hpp"

#include "../include/Meeting01Events.hpp"
#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Training.hpp"
#include "models/gru/GRUAutoencoder.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"

namespace meeting01::ga
{

namespace
{
auto recurrent_eval_context(const char* model,
    const RecurrentGenome& g,
    const meeting01::Meeting01Config& cfg,
    std::uint32_t seed) -> meeting01::EventContext
{
    return meeting01::ga_eval_context(model,
        g.encoding,
        genome_key(g),
        {{"hidden_size", g.hidden_size}, {"num_layers", g.num_layers}},
        seed,
        cfg.training.epochs,
        cfg.training.learning_rate,
        cfg.training.learning_rate_biophysical,
        cfg.training.early_stop_patience);
}
} // namespace

void evaluate_individual(LstmGaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total)
{
    const RecurrentGenome& g = ind.genome;
    const auto lstm_cfg = to_lstm_cfg(g, cfg, seed);

    nn::models::lstm::LSTMAutoencoder model(lstm_cfg);

    meeting01::ExperimentEvents::instance().set_pending_context(
        recurrent_eval_context("lstm-ae", g, cfg, seed));

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const meeting01::TrainResult train_result = meeting01::train_with_early_stopping_lstm(model,
        cfg,
        split.train_samples,
        split.val_samples,
        split.train_meta,
        split.val_meta,
        g.encoding,
        seed,
        progress_index,
        progress_total,
        train_ms,
        infer_ms);

    ind.val_mse = train_result.metrics.mse;
    ind.param_count = meeting01::parameter_count(model.params());
    ind.inference_cost = meeting01::estimate_lstm_macs(lstm_cfg);

    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {static_cast<double>(ind.val_mse), static_cast<double>(ind.inference_cost)};
}

void evaluate_individual(GruGaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total)
{
    const RecurrentGenome& g = ind.genome;
    const auto gru_cfg = to_gru_cfg(g, cfg, seed);

    nn::models::gru::GRUAutoencoder model(gru_cfg);

    meeting01::ExperimentEvents::instance().set_pending_context(
        recurrent_eval_context("gru-ae", g, cfg, seed));

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const meeting01::TrainResult train_result = meeting01::train_with_early_stopping_gru(model,
        cfg,
        split.train_samples,
        split.val_samples,
        split.train_meta,
        split.val_meta,
        g.encoding,
        seed,
        progress_index,
        progress_total,
        train_ms,
        infer_ms);

    ind.val_mse = train_result.metrics.mse;
    ind.param_count = meeting01::parameter_count(model.params());
    ind.inference_cost = meeting01::estimate_gru_macs(gru_cfg);

    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {static_cast<double>(ind.val_mse), static_cast<double>(ind.inference_cost)};
}

} // namespace meeting01::ga
