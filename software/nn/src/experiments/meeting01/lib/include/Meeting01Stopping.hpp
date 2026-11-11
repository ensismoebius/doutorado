#pragma once
// Meeting01Stopping.hpp -- when a meeting01 training stops, and which epoch's network it keeps
// (results_format 7). Every training uses it: each GA evaluation, each winner's final fit,
// every family (SNN-AE, ANN-AE, and the LSTM/GRU/Transformer-AE of the pre-validation).
//
// THE PROBLEM. An autoencoder of z-scored windows often starts on a plateau: its output
// ignores the input, its loss sits at the error of the mean target (~1.0 here), and a few
// epochs later it starts to learn. Patience 5 cannot tell that plateau from convergence. In
// the pre-validation (GridUnesp job 5314513) it stopped 8 of the 11 LSTM-AE search
// evaluations on EEG at epochs 8-16 with validation MSE 0.98-1.00 (the 3 that left the
// plateau in epoch 1 reached 0.58); the LSTM-AE's final refits on EEG stopped at epochs 8 and
// 13 and scored test MSE 1.000; even the ANN-AE control stopped once at epoch 6 at 0.991. A
// stop there reports "cannot learn" for a network that had not started.
//
// THE RULE, in two parts:
//
//   1. Patience is armed by skill, not by epoch. L0 is the monitor loss of the best predictor
//      that ignores its input (the mean training target; input_independent_monitor_loss).
//      Patience counts only once the monitor loss is at most (1 - s) x L0, with
//      s = training.early_stop_min_skill: the network must first remove that share of the
//      error the mean leaves. A run that never does trains every epoch (it is not stopped,
//      so it got the whole budget and still failed -- that IS the result).
//
//          monitor loss
//          L0  ─┬───────╮___________                 plateau: not armed, no bad epochs
//               │                   ╲
//      (1-s)L0 ─┼────────────────────●  armed here (epoch a): patience counts from now
//               │                     ╲___╱‾╲___ ... 5 epochs without a new best -> stop
//
//   2. The network kept is the one of the best monitor epoch (nn::training::
//      BestWeightsCallback), not the last one, which is `patience` epochs past it.
//
// Why s = 0.10 in the production profiles. Measured on the corrected windows (results_format 10,
// 06_meeting01_stopping_calibration.py): 84 trainings of all five families on fold 0 of FSDD and
// EEGMMIDB, monitor losses only. 29 ended on the plateau, and none of them passed a skill of
// 0.0505, so 0.10 is about twice what plateau noise reaches.
//
// The price of s is compute, not bias: a run below s never arms, so it trains all its epochs and
// keeps its best one. 14 of the 54 runs that learned are such runs (all on FSDD, LSTM/GRU/
// Transformer-AE, peaks 0.052-0.095), and every one ended within 0.01 of its peak, so patience
// had little to cut. Replaying the recorded curves, s = 0.05 saves 13 of 2349 epochs, lowers two
// runs' results, and changes no SNN-AE or ANN-AE training. The script's automatic rule refused
// 0.10 because of those 14; keeping it was the author's decision (2026-11-05), on these numbers.
//
// So "armed" does not mean "learned": on FSDD a run can learn a little without ever arming.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "Meeting01Config.hpp"
#include "tensor/Tensor.hpp"

namespace meeting01
{

// The monitor loss L0 of the input-independent predictor: the elementwise masked mean of
// the `fit` targets, scored on `monitor` exactly as Trainer::fit_supervised_masked scores
// its validation set -- masked MSE over each chunk of `batch_size` windows (sum of
// mask * residual^2 over sum of mask), chunks weighted by their window count. A position no
// fit window covers (mask 0 in all of them) is predicted as 0, the z-scored windows' mean;
// it only counts where a monitor window covers it. Triple is (input, target, mask), the
// Trainer's SampleTriple. Throws on an empty set or a target/mask whose size differs.
template <typename Triple>
auto input_independent_monitor_loss(
    const std::vector<Triple>& fit, const std::vector<Triple>& monitor, int batch_size) -> double
{
    if (fit.empty() || monitor.empty())
        throw std::invalid_argument(
            "input_independent_monitor_loss: " +
            std::string(fit.empty() ? "no fit windows" : "no monitor windows") +
            "; the stopping rule needs both to price the mean predictor");
    if (batch_size < 1)
        throw std::invalid_argument(
            "input_independent_monitor_loss: batch_size must be >= 1 (got " +
            std::to_string(batch_size) + ")");

    const auto n = static_cast<std::size_t>(std::get<1>(fit.front()).size());
    const auto check = [n](const auto& target, const auto& mask, const char* set, std::size_t i)
    {
        if (static_cast<std::size_t>(target.size()) != n ||
            static_cast<std::size_t>(mask.size()) != n)
            throw std::invalid_argument(
                "input_independent_monitor_loss: " + std::string(set) + " window " +
                std::to_string(i) + " has " + std::to_string(target.size()) + " target / " +
                std::to_string(mask.size()) + " mask values, the first " + "fit window " +
                std::to_string(n) + "; every window must have one shape");
    };

    std::vector<double> num(n, 0.0);
    std::vector<double> den(n, 0.0);
    for (std::size_t i = 0; i < fit.size(); ++i)
    {
        const auto& target = std::get<1>(fit[i]);
        const auto& mask = std::get<2>(fit[i]);
        check(target, mask, "fit", i);
        for (std::size_t k = 0; k < n; ++k)
        {
            const double m = mask.at(static_cast<nn::Index>(k));
            num[k] += m * target.at(static_cast<nn::Index>(k));
            den[k] += m;
        }
    }
    std::vector<double> mean(n, 0.0);
    for (std::size_t k = 0; k < n; ++k)
        if (den[k] > 0.0) mean[k] = num[k] / den[k];

    double loss_sum = 0.0;
    std::size_t scored = 0;
    const auto chunk = static_cast<std::size_t>(batch_size);
    for (std::size_t start = 0; start < monitor.size(); start += chunk)
    {
        const std::size_t end = std::min(start + chunk, monitor.size());
        double sq = 0.0;
        double mask_sum = 0.0;
        for (std::size_t i = start; i < end; ++i)
        {
            const auto& target = std::get<1>(monitor[i]);
            const auto& mask = std::get<2>(monitor[i]);
            check(target, mask, "monitor", i);
            for (std::size_t k = 0; k < n; ++k)
            {
                const double m = mask.at(static_cast<nn::Index>(k));
                const double d = mean[k] - target.at(static_cast<nn::Index>(k));
                sq += m * d * d;
                mask_sum += m;
            }
        }
        // MSELossImpl's masked loss of a chunk with no valid value is 0; same here.
        loss_sum += (mask_sum > 0.0 ? sq / mask_sum : 0.0) * static_cast<double>(end - start);
        scored += end - start;
    }
    return loss_sum / static_cast<double>(scored);
}

// L0 and the level that arms patience, (1 - early_stop_min_skill) x L0.
struct StoppingRule
{
    float trivial_loss = 0.0f;
    float arm_below = 0.0f;
};

// Throws when early_stop_min_skill is outside [0, 1) -- a config built in code, not from a
// validated profile, must set it -- or when L0 is not a positive finite number (a monitor set
// whose targets the mean reproduces exactly leaves nothing to learn, so "skill" is undefined).
inline auto make_stopping_rule(const Meeting01Config& cfg, double trivial_loss) -> StoppingRule
{
    const float s = cfg.training.early_stop_min_skill;
    if (!(s >= 0.0f && s < 1.0f))
        throw std::invalid_argument(
            "make_stopping_rule: training.early_stop_min_skill is " + std::to_string(s) +
            ", it must be in [0, 1). Set it in the profile's training block "
            "(Meeting01Config::validate checks it); a Meeting01Config built in code must set it "
            "too.");
    if (!(trivial_loss > 0.0) || !std::isfinite(trivial_loss))
        throw std::logic_error("make_stopping_rule: the mean predictor's monitor loss is " +
                               std::to_string(trivial_loss) +
                               "; it must be a positive finite number. Check the monitor windows "
                               "(empty masks, constant targets, NaN in the data).");
    return {static_cast<float>(trivial_loss), static_cast<float>((1.0 - s) * trivial_loss)};
}

} // namespace meeting01
