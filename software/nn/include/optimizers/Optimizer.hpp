#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <map>
#include <span>
#include <stdexcept>
#include <vector>

#include "Backend.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file Optimizer.hpp
 * @brief Minimal optimizer interface for updating trainable parameters in-place.
 *
 * How it fits:
 * - A model exposes its trainable tensors via `Module::params()`.
 * - An `Optimizer` receives those pointers and updates `param->data` using `param->grad()`.
 *
 * Design choices:
 * - Parameters are passed as `std::span<Tensor*>` to avoid unnecessary allocations.
 * - `attach()` lets an optimizer allocate per-parameter state (e.g., Adam moments) once.
 */

struct Optimizer
{
    using Tensor = nn::TensorImpl<nn::Backend>;

    /**
     * @brief Default contructor of the Optimizer object
     *
     */
    Optimizer() = default;

    /**
     * @brief Copy contructor of an Optimizer object (enabled)
     * Used whe copying an object from an variable to another
     * @param otherObjectReference
     */
    Optimizer(const Optimizer& otherObjectReference) = default;

    /**
     * @brief The '=' operator copies an object from an variable to another one
     *
     * @param otherObjectReference
     * @return Optimizer&
     */
    auto operator=(const Optimizer& otherObjectReference) -> Optimizer& = default;

    /**
     * @brief Move constructor of an Optimizer object (disabled)
     * Used when moving an object from an variable to another
     * @param otherObjectReference
     */
    Optimizer(Optimizer&& otherObjectReference) = delete;

    /**
     * @brief Disabled move operation from an objecto to another
     *
     * @param otherObjectReference
     * @return Optimizer&
     */
    auto operator=(Optimizer&& otherObjectReference) -> Optimizer& = delete;

    /**
     * @brief Apply one parameter update step.
     *
     * Precondition: each tensor in `params` has a valid gradient (usually set by backward()).
     */
    virtual auto step(std::span<Tensor*> params) -> void = 0;

    /**
     * Convenience no-arg overload: step using attached parameters (if any).
     * Calls the `step(std::span<...>)` virtual and dispatches to concrete
     * implementations.
     */
    virtual auto step() -> void
    {
        if (!attached_params_.empty())
        {
            step(std::span<Tensor*>{attached_params_.data(), attached_params_.size()});
        }
        else
        {
            throw std::runtime_error("Optimizer::step() called with no attached parameters");
        }
    }

    /**
     * @brief Set all parameter gradients to zero before the next backward pass.
     */
    virtual auto zero_grad(std::span<Tensor*> params) -> void = 0;

    /**
     * Convenience no-arg overload for zeroing attached parameters' gradients.
     */
    virtual auto zero_grad() -> void
    {
        if (!attached_params_.empty())
        {
            zero_grad(std::span<Tensor*>{attached_params_.data(), attached_params_.size()});
        }
        else
        {
            throw std::runtime_error("Optimizer::zero_grad() called with no attached parameters");
        }
    }

    /**
     * @brief Switch between the training and evaluation iterate, for optimizers that
     * distinguish them.
     *
     * Default: no-op — Adam, SGD and Lion evaluate at the same point they train at,
     * so the concept does not apply. Schedule-Free methods do: they train at an
     * extrapolated iterate `y` but their convergence guarantee (and their benefit) applies
     * to the averaged iterate `x`, so validation/inference must be done with `x` written
     * back into the parameters. Callers that run a validation pass should bracket it with
     * `train_mode(false)` / `train_mode(true)`; `Trainer` does this automatically.
     *
     * @param on true = training iterate, false = evaluation iterate.
     */
    virtual auto train_mode(bool on) -> void {}

    /**
     * @brief Optional hook for optimizers that need per-parameter state.
     *
     * Example: Adam stores first/second moments m/v with the same shape as each parameter.
     * Call this after building the model and before training. Stores `params` for the
     * no-arg step()/zero_grad() convenience overloads and resets `lr_scales_` to global
     * lr (attach_with_scales() re-assigns it after calling this). Concrete optimizers
     * that override this to allocate their own per-parameter state (Adam's moments,
     * SGD's velocity) must call `Optimizer::attach(params)` first to preserve this.
     */
    virtual auto attach(std::span<Tensor*> params) -> void
    {
        attached_params_.assign(params.begin(), params.end());
        lr_scales_.clear();
    }

    /**
     * @brief Attach parameters with per-parameter learning-rate scales.
     *
     * Enables per-parameter-group learning rates — critical for SNN training
     * where biophysical parameters (R, C, V_th) require a much smaller lr than
     * weight matrices (see .wiki/Guides/Engineering-Fixes-Log.md D3/D5, and TrainerConfig's
     * snn_lr_scale). Default implementation calls attach(params) then stores `lr_scales_`; concrete
     * step() implementations read `lr_scales_[i]` (default 1.0 past its size) to compute each
     * parameter's effective lr. Any Optimizer gets per-group scales for free unless it needs custom
     * behavior.
     *
     * @param params     Parameter tensors to optimise.
     * @param lr_scales  Per-parameter lr multiplier (same size as params).
     *                   Effective lr for param i = learning_rate * lr_scales[i].
     */
    virtual auto attach_with_scales(std::span<Tensor*> params, std::span<const float> lr_scales)
        -> void
    {
        if (params.size() != lr_scales.size())
        {
            throw std::invalid_argument("attach_with_scales: params and lr_scales must match");
        }
        attach(params);
        lr_scales_.assign(lr_scales.begin(), lr_scales.end());
    }

    // Stored copy of the last attached parameters. Concrete optimizers may still
    // override `attach()` but should call `Optimizer::attach(params)` first to
    // preserve this storage for the no-arg convenience methods.
    std::vector<Tensor*> attached_params_;

    // Per-parameter lr multipliers (1.0 = global lr), aligned with attached_params_.
    // Populated by attach_with_scales(); reset to empty by attach().
    std::vector<float> lr_scales_;

    /// Decoupled L2 weight decay. 0 = disabled. Decoupled (applied to the parameter
    /// directly, not folded into the gradient) so an adaptive optimizer's per-parameter
    /// scaling does not distort the penalty. Concrete step() implementations apply it
    /// only to 2-D weight matrices (rows>1 && cols>1); biases (Nx1) and SNN biophysical
    /// scalars (1x1: R, C, V_th) are excluded, so tau=R*C and the firing threshold are
    /// never pulled toward zero. Lives on the base so callers holding an `Optimizer&`
    /// can configure it without knowing the concrete type.
    /// Reference: Loshchilov & Hutter, ICLR 2019 (Decoupled Weight Decay Regularization),
    /// which defines both the AdamW and SGDW variants.
    float weight_decay = 0.0F;

    virtual ~Optimizer() = default;
    /**
     * @brief Return optimizer internal state as a map of name->Tensor.
     * Default: empty. Concrete optimizers may override to expose moments or counters.
     */
    virtual auto state_dict() const -> std::map<std::string, Tensor>
    {
        return {};
    }

    /**
     * @brief Load optimizer internal state from a map produced by `state_dict()`.
     */
    virtual void load_state_dict(const std::map<std::string, Tensor>&) {}
};

/**
 * @brief RAII guard that puts an optimizer in evaluation mode for the current scope.
 *
 * Exposes the evaluation iterate on construction and restores the training iterate on
 * destruction, so a validation/inference pass measures the right point. A no-op for every
 * optimizer that does not distinguish the two (see `Optimizer::train_mode`), which makes it
 * safe to wrap unconditionally around any evaluation block.
 */
class OptimizerEvalScope
{
   public:
    explicit OptimizerEvalScope(Optimizer& opt) : opt_(opt)
    {
        opt_.train_mode(false);
    }

    ~OptimizerEvalScope()
    {
        opt_.train_mode(true);
    }

    OptimizerEvalScope(const OptimizerEvalScope&) = delete;
    auto operator=(const OptimizerEvalScope&) -> OptimizerEvalScope& = delete;
    OptimizerEvalScope(OptimizerEvalScope&&) = delete;
    auto operator=(OptimizerEvalScope&&) -> OptimizerEvalScope& = delete;

   private:
    Optimizer& opt_;
};

#endif // OPTIMIZER_HPP
