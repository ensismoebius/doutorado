#include "../include/Meeting01RecurrentGaFitness.hpp"

#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Training.hpp"
#include "models/gru/GRUAutoencoder.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"

namespace meeting01::ga
{

void evaluate_individual(LstmGaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total)
{
    const RecurrentGenome& g = ind.genome;
    const auto lstm_cfg = to_lstm_cfg(g, cfg);

    nn::models::lstm::LSTMAutoencoder model(lstm_cfg);

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const meeting01::TrainResult train_result = meeting01::train_with_early_stopping_lstm(model,
        cfg,
        split.train_samples,
        split.val_samples,
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
    const auto gru_cfg = to_gru_cfg(g, cfg);

    nn::models::gru::GRUAutoencoder model(gru_cfg);

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const meeting01::TrainResult train_result = meeting01::train_with_early_stopping_gru(model,
        cfg,
        split.train_samples,
        split.val_samples,
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
