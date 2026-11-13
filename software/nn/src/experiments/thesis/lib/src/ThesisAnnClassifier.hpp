// The non-spiking twin of ThesisDsnnClassifier: classifier.type "ann".
//
// WHY A TWIN. A control answers "what does spiking buy?" only if it differs from the DSNN in
// spiking alone. The control used before (SimpleResNet) did not: for layer_spec residual:2 it
// built two Linear->ReLU->Linear blocks (four hidden 128x128 matrices) where the DSNN builds
// one Linear->LIF block (one), so a DSNN x control difference mixed spiking with three times
// the hidden capacity. This class copies the DSNN's graph and replaces what is spiking:
//
//                   ThesisDsnnClassifier                    ThesisAnnClassifier
//   input           (B, D), tiled over T = 16 steps         (B, D), once
//   fc_in           Linear D->H                             Linear D->H      (same init)
//   nonlinearity    LIF (spikes)                            ReLU
//   block i         Linear H->H -> LIF, + skip              Linear H->H -> ReLU, + skip
//   (depth - 1 of them)
//   fc_out          Linear H->C, mean over the T steps      Linear H->C      (same init)
//
// The Linear layers are Kaiming-initialized with the DSNN's seeds and tag, so for one seed
// the two networks start from the same weights (verified in thesis_protocol_gtest).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "Backend.hpp"
#include "Meeting01Energy.hpp"
#include "initializers/kaiming_snn.hpp"
#include "layers/activations/ReLU.hpp"
#include "layers/dense/Linear.hpp"
#include "tensor/Tensor.hpp"

namespace thesis
{

class ThesisAnnClassifier : public Module<nn::Backend>
{
   public:
    using Tensor = nn::Tensor;

    ThesisAnnClassifier(
        int input_dim, int hidden_dim, int output_dim, int depth, std::uint32_t seed)
        : fc_in_(std::make_shared<LinearImpl<nn::Backend>>(input_dim, hidden_dim)),
          act_in_(std::make_shared<ReLUImpl<nn::Backend>>()),
          fc_out_(std::make_shared<LinearImpl<nn::Backend>>(hidden_dim, output_dim)),
          in_dim_(input_dim),
          hidden_dim_(hidden_dim),
          out_dim_(output_dim)
    {
        if (depth < 1) throw std::invalid_argument("ThesisAnnClassifier: depth must be >= 1");
        const int n_blocks = depth - 1; // as ThesisDsnnClassifier
        for (int i = 0; i < n_blocks; ++i)
        {
            hidden_fc_.push_back(std::make_shared<LinearImpl<nn::Backend>>(hidden_dim, hidden_dim));
            hidden_act_.push_back(std::make_shared<ReLUImpl<nn::Backend>>());
        }
        kaimingSNNInitializer(fc_in_, seed + 1U, "e05_dsnn");
        kaimingSNNInitializer(fc_out_, seed + 2U, "e05_dsnn");
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
            kaimingSNNInitializer(
                hidden_fc_[i], seed + 100U + static_cast<unsigned int>(i), "e05_dsnn");
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        last_batch_ = static_cast<double>(input.rows());
        Tensor h = act_in_->forward(fc_in_->forward(input, requires_grad), requires_grad);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            const Tensor skip = h;
            Tensor b =
                hidden_act_[i]->forward(hidden_fc_[i]->forward(h, requires_grad), requires_grad);
            h = b.add(skip);
        }
        return fc_out_->forward(h, requires_grad);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        Tensor g = fc_out_->backward(grad_output);
        for (size_t i = hidden_fc_.size(); i-- > 0;)
        {
            const Tensor skip_grad = g;
            g = hidden_fc_[i]->backward(hidden_act_[i]->backward(g));
            g.add_inplace(skip_grad);
        }
        return fc_in_->backward(act_in_->backward(g));
    }

    [[nodiscard]] auto params() -> std::span<Tensor*> override
    {
        param_ptrs_.clear();
        auto add = [this](const std::span<Tensor*>& p)
        { param_ptrs_.insert(param_ptrs_.end(), p.begin(), p.end()); };
        add(fc_in_->params());
        for (auto& l : hidden_fc_) add(l->params());
        add(fc_out_->params());
        return {param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        std::map<std::string, Tensor> out;
        auto merge = [&out](const std::string& prefix, const std::map<std::string, Tensor>& sd)
        {
            for (const auto& kv : sd) out[prefix + kv.first] = kv.second;
        };
        merge("fc_in.", fc_in_->state_dict());
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
            merge("hidden_fc." + std::to_string(i) + ".", hidden_fc_[i]->state_dict());
        merge("fc_out.", fc_out_->state_dict());
        return out;
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        auto load_prefix = [&sd](const std::string& prefix, auto& layer)
        {
            std::map<std::string, Tensor> sub;
            for (const auto& kv : sd)
                if (kv.first.rfind(prefix, 0) == 0) sub[kv.first.substr(prefix.size())] = kv.second;
            layer->load_state_dict(sub);
        };
        load_prefix("fc_in.", fc_in_);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
            load_prefix("hidden_fc." + std::to_string(i) + ".", hidden_fc_[i]);
        load_prefix("fc_out.", fc_out_);
    }

    void train(bool on) override
    {
        fc_in_->train(on);
        act_in_->train(on);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            hidden_fc_[i]->train(on);
            hidden_act_[i]->train(on);
        }
        fc_out_->train(on);
    }

    /// The inference energy of ONE sample (Meeting01Energy's ledger: Lemaire et al. 2022 at
    /// Horowitz 2014 prices). A dense network's cost does not depend on the data, but the
    /// ledger is read after a forward like the DSNN's, so both are measured the same way.
    [[nodiscard]] auto inference_ledger() const -> meeting01::EnergyLedger
    {
        if (last_batch_ <= 0.0)
            throw std::logic_error("ThesisAnnClassifier: inference_ledger() before any forward");
        using meeting01::add_dense_fc;
        using meeting01::add_nonlinearity;
        using meeting01::Nonlinearity;
        const double D = in_dim_, H = hidden_dim_, C = out_dim_;
        meeting01::EnergyLedger e;
        add_dense_fc(e, 1.0, D, H, true);
        add_nonlinearity(e, H, Nonlinearity::Comparison);
        for (size_t i = 0; i < hidden_fc_.size(); ++i)
        {
            add_dense_fc(e, 1.0, H, H, true);
            add_nonlinearity(e, H, Nonlinearity::Comparison);
            e.fp_adds += H; // the skip add: two operands read, one sum written
            e.access(e.rd_in, 2.0 * H, H);
            e.access(e.wr_out, H, H);
        }
        add_dense_fc(e, 1.0, H, C, true);
        return e;
    }

   private:
    std::shared_ptr<LinearImpl<nn::Backend>> fc_in_;
    std::shared_ptr<ReLUImpl<nn::Backend>> act_in_;
    std::vector<std::shared_ptr<LinearImpl<nn::Backend>>> hidden_fc_;
    std::vector<std::shared_ptr<ReLUImpl<nn::Backend>>> hidden_act_;
    std::shared_ptr<LinearImpl<nn::Backend>> fc_out_;
    std::vector<Tensor*> param_ptrs_;
    int in_dim_ = 0;
    int hidden_dim_ = 0;
    int out_dim_ = 0;
    double last_batch_ = 0.0;
};

} // namespace thesis
