#ifndef OPTIMIZER_EVAL_SCOPE_HPP
#define OPTIMIZER_EVAL_SCOPE_HPP

#include "optimizers/Optimizer.hpp"

/**
 * @file OptimizerEvalScope.hpp
 * @brief RAII eval-mode guard for Optimizer (extracted from Optimizer.hpp).
 */

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

#endif // OPTIMIZER_EVAL_SCOPE_HPP
