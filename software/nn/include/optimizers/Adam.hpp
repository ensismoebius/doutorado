#ifndef ADAM_HPP
#define ADAM_HPP

#include <concepts>
#include <stdexcept>

#include "optimizers/Optimizer.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file Adam.hpp
 * @brief Adam optimizer implementation for this project.
 *
 * How it fits:
 * - Training code typically does:
 *   1) `auto params = model.params();`
 *   2) `optimizer.attach(params);`
 *   3) Loop: `zero_grad(params)` → forward/loss/backward → `step(params)`.
 *
 * Important invariant:
 * - `attach()` allocates per-parameter state (m, v) sized to match the parameters.
 *   If you change the model architecture (different params list), you must call `attach()` again.
 */
/**
 * Adam (Adaptive Moment Estimation) combina as ideias do Momentum
 * (acumular média móvel dos gradientes) e do RMSProp (acumular
 * média móvel dos quadrados dos gradientes), ajustando dinamicamente
 * a taxa de aprendizado para cada parâmetro.
 *
 * Estrutura do otimizador Adam.
 * Para cada parâmetro, Adam mantém duas "memórias":
 *   - m: média móvel dos gradientes (primeiro momento, similar ao momentum)
 *   - v: média móvel dos quadrados dos gradientes (segundo momento, similar ao RMSProp)
 */
struct Adam : public Optimizer
{
    using Tensor = Optimizer::Tensor;

    float learning_rate;      // Taxa de aprendizado (learning rate)
    float decay_rate_moment1; // Hiperparâmetro de decaimento do primeiro momento (tipicamente 0.9)
    float decay_rate_moment2; // Hiperparâmetro de decaimento do segundo momento (tipicamente 0.999)
    float epsilon;            // Termo pequeno para evitar divisão por zero
    int time_step;            // Contador de iterações

    // `weight_decay` (decoupled L2 → AdamW) and `lr_scales_` (per-parameter lr
    // multipliers) are inherited from Optimizer — see Optimizer.hpp. Both are read
    // by step() below.

    std::vector<Tensor> moment1; // Vetor de médias móveis dos gradientes
    std::vector<Tensor> moment2; // Vetor de médias móveis dos quadrados dos gradientes

    // Inicializa o Adam com hiperparâmetros padrão recomendados na literatura.
    // learning_rate: taxa de aprendizado, decay_rate_moment1: decaimento do primeiro momento,
    // decay_rate_moment2: decaimento do segundo momento, epsilon: termo de estabilidade.
    explicit Adam(float learning_rate_ = 0.001F,
        float decay_rate_moment1_ = 0.9F,
        float decay_rate_moment2_ = 0.999F,
        float epsilon_ = 1e-8F)
        : learning_rate(learning_rate_),
          decay_rate_moment1(decay_rate_moment1_),
          decay_rate_moment2(decay_rate_moment2_),
          epsilon(epsilon_),
          time_step(0)
    {
        if (learning_rate_ <= 0.0F)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (learning_rate_ > 1e8F)
        {
            throw std::invalid_argument("Learning rate is unreasonably large");
        }
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        std::map<std::string, Tensor> out;
        // Save time_step as a 1x1 tensor
        Tensor t_time(1, 1);
        t_time.at(0, 0) = static_cast<float>(time_step);
        out["time_step"] = t_time;
        // Save moment1 and moment2 as indexed entries
        for (size_t i = 0; i < moment1.size(); ++i)
        {
            out["moment1." + std::to_string(i)] = moment1[i];
        }
        for (size_t i = 0; i < moment2.size(); ++i)
        {
            out["moment2." + std::to_string(i)] = moment2[i];
        }
        return out;
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        // time_step
        auto it = sd.find("time_step");
        if (it != sd.end())
        {
            const Tensor& tt = it->second;
            if (tt.rows() > 0 && tt.cols() > 0)
            {
                time_step = static_cast<int>(tt.at(0, 0));
            }
        }
        // moments: look for keys moment1.N and moment2.N
        for (const auto& kv : sd)
        {
            const std::string& k = kv.first;
            if (k.rfind("moment1.", 0) == 0)
            {
                auto idx = static_cast<size_t>(std::stoul(k.substr(8)));
                if (idx < moment1.size()) moment1[idx] = kv.second;
            }
            else if (k.rfind("moment2.", 0) == 0)
            {
                auto idx = static_cast<size_t>(std::stoul(k.substr(8)));
                if (idx < moment2.size()) moment2[idx] = kv.second;
            }
        }
    }

    // attach_with_scales() is inherited from Optimizer: its base implementation calls
    // this attach() (virtual, so Adam's moments are allocated) and then stores the
    // scales, which step() reads below. No Adam-specific override is needed.

    // Inicializa os vetores moment1 e moment2 para cada parâmetro, com zeros do mesmo shape dos
    // gradientes. Deve ser chamado sempre que os parâmetros mudarem.
    void attach(std::span<Tensor*> params) override
    {
        // Stores attached_params_ for the no-arg step()/zero_grad(), and resets
        // lr_scales_ to global lr (audit a-1) — attach_with_scales() re-assigns
        // them after calling this.
        Optimizer::attach(params);

        // Why attach(): Adam needs one moment1/moment2 tensor per parameter.
        // Pitfall: if `params` changes size/order, moment1/moment2 will no longer align.
        moment1.clear();
        moment2.clear();
        for (auto* param : params)
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Cannot attach null parameter to optimizer");
            }
            // Initialize moment1 and moment2 tensors with zeros matching parameter shape
            moment1.emplace_back(param->rows(), param->cols());
            moment2.emplace_back(param->rows(), param->cols());
            // Initialize to zero
            for (size_t i = 0; i < moment1.back().rows(); ++i)
            {
                for (size_t j = 0; j < moment1.back().cols(); ++j)
                {
                    moment1.back().at(i, j) = 0.0f;
                    moment2.back().at(i, j) = 0.0f;
                }
            }
        }
    }

    // Realiza um passo de atualização dos parâmetros segundo o algoritmo Adam.
    // Fórmulas:
    //   m_t = β1 * m_{t-1} + (1 - β1) * g_t
    //   v_t = β2 * v_{t-1} + (1 - β2) * (g_t)^2
    //   m̂_t = m_t / (1 - β1^t)
    //   v̂_t = v_t / (1 - β2^t)
    //   θ = θ - learning_rate * m̂_t / (sqrt(v̂_t) + ε)
    // Dispatches to the backend's fused Adam kernel when one exists (must be a
    // template: `if constexpr` only discards the untaken branch during template
    // instantiation, so a plain member function would fail to compile for
    // backends without adam_step_inplace).
    template <typename B>
    auto try_fused_step(
        nn::TensorImpl<B>& param, nn::TensorImpl<B>& m1, nn::TensorImpl<B>& m2, float lr_i) -> bool
    {
        if constexpr (requires(B& p, B& a, B& b, const B& g) {
                          {
                              p.adam_step_inplace(a, b, g, 0.F, 0.F, 0.F, 0.F, 1.F, 1.F)
                          } -> std::convertible_to<bool>;
                      })
        {
            const float bc1 =
                static_cast<float>(1.0 - std::pow(static_cast<double>(decay_rate_moment1),
                                             static_cast<double>(time_step)));
            const float bc2 =
                static_cast<float>(1.0 - std::pow(static_cast<double>(decay_rate_moment2),
                                             static_cast<double>(time_step)));
            return param.get_backend().adam_step_inplace(m1.get_backend(),
                m2.get_backend(),
                param.get_backend().grad_ref(),
                lr_i,
                decay_rate_moment1,
                decay_rate_moment2,
                epsilon,
                bc1,
                bc2);
        }
        else
        {
            return false;
        }
    }

    auto step(std::span<Tensor*> paramsList) -> void override
    {
        // Numerical note: bias correction uses pow(beta, t). For large t, beta^t → 0.
        time_step += 1;
        for (size_t i = 0; i < paramsList.size(); ++i) [[likely]]
        {
            if (paramsList[i] == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            auto& param = *paramsList[i];
            // Per-parameter lr: use scale if provided, otherwise 1.0 (global lr).
            const float lr_i = learning_rate * (i < lr_scales_.size() ? lr_scales_[i] : 1.0f);

            // Decoupled weight decay (AdamW): θ ← θ(1 - lr_i*wd), applied BEFORE the
            // gradient step. Loshchilov & Hutter (ICLR 2019) define the decay against
            // θ_{t-1}, i.e. the pre-update parameter, and torch.optim.AdamW implements it
            // in exactly this order. Applying it after the update instead decays θ_t,
            // leaving a systematic lr²·wd·u error — small, but a real deviation from the
            // algorithm; ground truth (optimizer_parity_gtest vs torch.optim.AdamW) is
            // what surfaced it. Restricted to 2-D weight matrices: biases (N×1) and SNN
            // biophysical scalars (1×1: R, C, V_th) are skipped, so tau=R·C and the firing
            // threshold are never decayed.
            if (weight_decay > 0.0f && param.rows() > 1 && param.cols() > 1)
            {
                // nn::Tensor is a value type: assigning to `param` replaces its storage and
                // DROPS the gradient buffer (see SGDMinimal.hpp). Save the gradient across
                // the decay and restore it, or every read below — including the fused
                // kernel's grad_ref() — would see an empty gradient and skip the step.
                const Tensor saved_grad = param.grad();
                // Multiplicative form p*(1 - lr*wd) mirrors torch's p.mul_(1 - lr*wd):
                // algebraically identical to p + p*(-lr*wd), but a single rounding.
                param = param.multiply_scalar(1.0F - (lr_i * weight_decay));
                param.set_grad(saved_grad);
            }

            // Fused single-kernel update when the backend provides it (OpenCL:
            // adam_step_kernel) — same math as the generic path below, without
            // ~15 kernel launches and intermediate tensors per parameter.
            if (try_fused_step(param, moment1[i], moment2[i], lr_i))
            {
                continue;
            }

            // Atualiza as médias móveis dos gradientes e dos quadrados dos gradientes
            // moment1[i] = decay_rate_moment1 * moment1[i] + (1 - decay_rate_moment1) * grad
            // Precondition: `param.grad()` must be meaningful. If the model did not run
            // backward(), gradients may be zero or uninitialized depending on backend behavior.
            Tensor grad = param.grad(); // Get gradient from parameter
            Tensor grad_contrib = grad.multiply_scalar(1.0f - decay_rate_moment1);
            moment1[i] = moment1[i].multiply_scalar(decay_rate_moment1).add(grad_contrib);

            // moment2[i] = decay_rate_moment2 * moment2[i] + (1 - decay_rate_moment2) * grad^2
            Tensor grad_squared = grad.multiply(grad); // Square gradient, not weights!
            Tensor grad_squared_contrib = grad_squared.multiply_scalar(1.0f - decay_rate_moment2);
            moment2[i] = moment2[i].multiply_scalar(decay_rate_moment2).add(grad_squared_contrib);

            // Corrige o viés das médias móveis
            float bias_correction1 =
                static_cast<float>(1.0 - std::pow(static_cast<double>(decay_rate_moment1),
                                             static_cast<double>(time_step)));
            float bias_correction2 =
                static_cast<float>(1.0 - std::pow(static_cast<double>(decay_rate_moment2),
                                             static_cast<double>(time_step)));
            auto m_hat = moment1[i] / bias_correction1;
            auto v_hat = moment2[i] / bias_correction2;

            // Atualiza o parâmetro usando as médias móveis corrigidas
            // param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            Tensor v_hat_sqrt = v_hat.sqrt();
            Tensor v_hat_sqrt_eps = v_hat_sqrt.add_scalar(epsilon);
            Tensor m_hat_scaled = m_hat.multiply_scalar(lr_i);
            Tensor update_step = m_hat_scaled.divide(v_hat_sqrt_eps);

            param = param.add(update_step.multiply_scalar(-1.0f));

            // Weight decay is NOT applied here: it is decoupled and must act on the
            // pre-update parameter, so it happens at the top of this loop iteration
            // (see the comment there). Decoupled = not folded into the gradient, so
            // Adam's adaptive 1/sqrt(v̂) scaling never distorts the penalty.
        }
    }

    // Zera os gradientes de todos os parâmetros.
    auto zero_grad(std::span<Tensor*> paramsList) -> void override
    {
        for (auto* param : paramsList) [[likely]]
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            param->zero_grad();
        }
    }

    // Exemplo Numérico (com vetores):
    // Suponha um parâmetro θ = [1.0, 2.0], gradiente g_t = [0.1, -0.2], m e v inicializados em
    // zero. Passo 1 (t=1):
    //   m = 0.9 * [0, 0] + 0.1 * [0.1, -0.2] = [0.01, -0.02]
    //   v = 0.999 * [0, 0] + 0.001 * [0.01, 0.04] = [0.00001, 0.00004]
    //   m̂ = m / (1 - 0.9^1) = [0.01, -0.02] / 0.1 = [0.1, -0.2]
    //   v̂ = v / (1 - 0.999^1) = [0.00001, 0.00004] / 0.001 = [0.01, 0.04]
    //   Atualização:
    //     θ = θ - lr * m̂ / (sqrt(v̂) + ε)
    //     Para lr=0.001, ε=1e-8:
    //     sqrt(v̂) = [0.1, 0.2]
    //     θ = [1.0, 2.0] - 0.001 * [0.1, -0.2] / ([0.1, 0.2] + 1e-8)
    //       = [1.0, 2.0] - 0.001 * [1, -1]
    //       = [0.999, 2.001]

    // O Adam adapta automaticamente o tamanho do passo para cada parâmetro, acelerando a
    // convergência e tornando o treinamento mais robusto a diferentes escalas de gradiente.

    // Referências:
    // - Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.
    // https://arxiv.org/abs/1412.6980
    // - https://ruder.io/optimizing-gradient-descent/index.html#adam
};
#endif // ADAM_HPP