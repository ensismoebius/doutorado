#include <Eigen/src/Core/Matrix.h>

#include "Optimizer.hpp"
#include "../tensor/Tensor.hpp"

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
    float lr;    // Taxa de aprendizado (learning rate)
    float beta1; // Hiperparâmetro de decaimento do primeiro momento (tipicamente 0.9)
    float beta2; // Hiperparâmetro de decaimento do segundo momento (tipicamente 0.999)
    float eps;   // Termo pequeno para evitar divisão por zero
    int t;       // Contador de iterações

    std::vector<Eigen::MatrixXf> m; // Vetor de médias móveis dos gradientes
    std::vector<Eigen::MatrixXf> v; // Vetor de médias móveis dos quadrados dos gradientes

    // Inicializa o Adam com hiperparâmetros padrão recomendados na literatura.
    // lr: taxa de aprendizado, beta1: decaimento do primeiro momento, beta2: decaimento do segundo
    // momento, eps: termo de estabilidade.
    explicit Adam(float learning_rate = 0.001F, float beta1 = 0.9F, float beta2 = 0.999F,
                  float eps = 1e-8F)
        : lr(learning_rate), beta1(beta1), beta2(beta2), eps(eps), t(0)
    {
    }

    // Inicializa os vetores m e v para cada parâmetro, com zeros do mesmo shape dos gradientes.
    // Deve ser chamado sempre que os parâmetros mudarem.
    auto attach(std::vector<Tensor*>& paramsList) -> void
    {
        m.clear();
        v.clear();
        for (auto* param : paramsList)
        {
            m.emplace_back(Eigen::MatrixXf::Zero(param->get_grad_ref().rows(), param->get_grad_ref().cols()));
            v.emplace_back(Eigen::MatrixXf::Zero(param->get_grad_ref().rows(), param->get_grad_ref().cols()));
        }
    }

    // Realiza um passo de atualização dos parâmetros segundo o algoritmo Adam.
    // Fórmulas:
    //   m_t = β1 * m_{t-1} + (1 - β1) * g_t
    //   v_t = β2 * v_{t-1} + (1 - β2) * (g_t)^2
    //   m̂_t = m_t / (1 - β1^t)
    //   v̂_t = v_t / (1 - β2^t)
    //   θ = θ - lr * m̂_t / (sqrt(v̂_t) + ε)
    auto step(std::vector<Tensor*>& paramsList) -> void override
    {
        t += 1;
        for (size_t i = 0; i < paramsList.size(); ++i)
        {
            auto& param = *paramsList[i];
            // Atualiza as médias móveis dos gradientes e dos quadrados dos gradientes
            m[i] = (beta1 * m[i].array() + (1 - beta1) * param.get_grad_ref().array()).matrix();
            v[i] = (beta2 * v[i].array() + (1 - beta2) * param.get_grad_ref().array().square()).matrix();

            // Corrige o viés das médias móveis
            Eigen::MatrixXf m_hat = m[i] / (1 - std::pow(beta1, t));
            Eigen::MatrixXf v_hat = v[i] / (1 - std::pow(beta2, t));

            // Atualiza o parâmetro usando as médias móveis corrigidas
            param.set_data((param.get_data_ref().array() - lr * m_hat.array() / (v_hat.array().sqrt() + eps)).matrix());
        }
    }

    // Zera os gradientes de todos os parâmetros.
    auto zero_grad(std::vector<Tensor*>& paramsList) -> void override
    {
        for (auto* param : paramsList)
        {
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
