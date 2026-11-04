#include "../include/Meeting01GaFitness.hpp"

#include <algorithm>

#include "../include/Meeting01Events.hpp"
#include "../include/Meeting01Metrics.hpp"
#include "../include/Meeting01Training.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"

namespace meeting01::ga
{

void evaluate_individual(Meeting01GaIndividual& ind,
    const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    std::uint32_t seed,
    std::size_t progress_index,
    std::size_t progress_total)
{
    using nn::models::autoencoder::AutoencoderConfig;
    using nn::models::autoencoder::ProtocolSpikingAutoencoder;

    const Genome& g = ind.genome;
    // The genome's own T: the training, the encodings and the cost below all read it.
    const meeting01::Meeting01Config cfg_t = meeting01::with_snn_time_steps(cfg, g.time_steps);

    AutoencoderConfig snn_config = to_ae_config(g, cfg);
    snn_config.initializer_seed = seed;
    snn_config.initializer_sampler_type = "ga|" + genome_key(g);

    ProtocolSpikingAutoencoder model(snn_config);

    meeting01::ExperimentEvents::instance().set_pending_context(meeting01::ga_eval_context("snn-ae",
        g.encoding,
        genome_key(g),
        {{"encoder_widths", g.encoder_widths},
            {"architecture", g.architecture},
            {"v_th", g.voltage_threshold},
            {"alpha", g.alpha},
            {"time_steps", g.time_steps}},
        seed,
        cfg.training.epochs,
        cfg.training.learning_rate,
        cfg.training.learning_rate_biophysical,
        cfg.training.early_stop_patience));

    float train_ms = 0.0f;
    float infer_ms = 0.0f;
    const meeting01::TrainResult train_result = meeting01::train_with_early_stopping_snn(model,
        cfg_t,
        split.train_samples,
        split.val_samples,
        split.train_meta,
        split.val_meta,
        split.val_labels,
        g.encoding,
        g.architecture,
        g.alpha,
        g.voltage_threshold,
        seed,
        progress_index,
        progress_total,
        train_ms,
        infer_ms);

    ind.val_mse = train_result.metrics.mse;
    ind.param_count = meeting01::parameter_count(model.params());
    // Spike-blind and analytic (it ranks genomes, so it must not depend on how one training
    // run happened to fire); the decoder runs once when the latent is averaged over time.
    ind.inference_cost =
        meeting01::estimate_snn_macs(static_cast<std::size_t>(cfg.dataset.window_size),
            g.encoder_widths,
            g.time_steps,
            cfg.model.snn_time_pool == "latent" ? 1 : g.time_steps);

    ind.feasible = true;
    ind.constraint_violation = 0.0;
    ind.objectives = {static_cast<double>(ind.val_mse), static_cast<double>(ind.inference_cost)};
}

} // namespace meeting01::ga
