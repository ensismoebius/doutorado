"""Demonstração — como comparar autoencoders de forma justa (SNN x LSTM x
GRU x Transformer), no desenho do experimento Meeting01 do software/nn.

Every number on screen is either computed here (a real PCA fitted on
synthetic z-scored windows, the mean-frame reference, one sample's squared
error) or quoted from software/nn (latent 16 audio / 64 EEG, Meeting01's B2
trivial-model MSEs, the measured per-epoch costs). The four families are
NOT trained here and get no invented scores: the last step leaves their
bars empty on purpose -- see comparison/autoencoder_synthetic.py.

The two PCA sweeps (k = 1 -> k, then k -> 4k) are built frame by frame
from the exact reconstruction at each k, not tweened between two
endpoints, so the curve and the error shading on screen are the real ones.
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.comparison.autoencoder_synthetic import (
    B2_FREE_WIN,
    EPOCH_SECONDS,
    LATENCY_TRIVIAL_MSE_T16,
    LATENT_AUDIO,
    LATENT_EEG,
    OLD_LATENCY_TIME_AXIS,
    TRIVIAL_MSE_BY_TARGET,
    WINDOW,
    fit_pca,
    mse,
    pca_reconstruct,
    synthetic_window,
    window_set,
)
from efficient_nn_lab.core.demo import DemoModule, Frame, slider, transition
from efficient_nn_lab.core.math_utils import SEED

_N_TRAIN = 400
_TEST_SEED = SEED + 7  # the test window is drawn apart from every training window
_K_MAX = 128
_STEPS = 10
_PCA_EQUATION = "\\hat{x} = mu + (x - mu) V_k V_k^T"
_MAX_SWEEP_FRAMES = 24


def _sweep(first: int, stop: int) -> list[int]:
    """Integer k values from ``first`` up to (not including) ``stop``, at most
    _MAX_SWEEP_FRAMES of them -- every frame is an exact PCA at an integer k,
    so a wide range is subsampled rather than interpolated."""
    if stop <= first:
        return []
    count = min(stop - first, _MAX_SWEEP_FRAMES)
    return sorted({int(round(v)) for v in np.linspace(first, stop - 1, count)})


class AutoencoderComparisonDemo(DemoModule):
    title = "Comparação -> Autoencoders (SNN x LSTM x GRU x Transformer)"
    slug = "comparison.autoencoders"
    description = (
        "Como o experimento Meeting01 compara quatro famílias de autoencoder sem trapacear: "
        "mesmo gargalo, mesmo alvo, referências lineares (média e PCA) e avaliação num locutor "
        "nunca visto. PCA de verdade sobre janelas sintéticas; as quatro famílias não são "
        "treinadas aqui e não recebem números inventados."
    )

    def __init__(self) -> None:
        self.latent = LATENT_AUDIO
        super().__init__()

    def parameters(self) -> dict[str, dict[str, object]]:
        return {"latent": slider("tamanho do latente k", 4, 64, 4, self.latent)}

    def _build_frames(self) -> list[Frame]:
        k = int(self.latent)
        k_wide = min(4 * k, _K_MAX)
        pca = fit_pca(window_set(_N_TRAIN))
        window = synthetic_window(np.random.RandomState(_TEST_SEED))
        k_all = np.arange(1, _K_MAX + 1)
        mse_all = np.array([mse(window, pca_reconstruct(pca, window, int(j))) for j in k_all])
        mean_mse = mse(window, pca.mean)
        sample = int(np.argmax(np.abs(window)))  # the most visible single-sample error

        def recon(j: int) -> np.ndarray:
            return pca_reconstruct(pca, window, j)

        base: dict[str, object] = {
            "kind": "autoencoder_comparison_pipeline",
            "window": window, "recon": pca.mean, "recon_reveal": 0.0, "recon_label": "média do treino",
            "error_reveal": 0.0, "sample": sample, "sample_reveal": 0.0,
            "k_now": float(k), "k_target": k, "k_all": k_all, "mse_all": mse_all, "mean_mse": mean_mse,
            "curve_upto": 0.0, "domain_reveal": 0.0, "bottom": "curve", "bottom_reveal": 0.0,
            "trivial": TRIVIAL_MSE_BY_TARGET, "epoch_seconds": EPOCH_SECONDS,
        }

        def frame(label: str, explanation: str, equation: str, checkpoint: bool = True, **values: object) -> Frame:
            return Frame(label, {**base, **values}, explanation, equation, is_checkpoint=checkpoint)

        def pca_values(j: int) -> dict[str, object]:
            return {
                "recon": recon(j), "recon_reveal": 1.0, "recon_label": f"PCA, k = {j}",
                "error_reveal": 1.0, "k_now": float(j), "curve_upto": float(j), "bottom_reveal": 1.0,
            }

        intro = frame(
            f"O problema: 256 amostras em {k} números",
            f"Uma janela de áudio tem {WINDOW} amostras. Um autoencoder espreme essas {WINDOW} em só "
            f"{k} números — o latente — e tenta reconstruir a janela a partir deles. Se der certo, os "
            f"{k} números guardam o essencial do sinal (é por isso que a tese usa o latente como "
            "features). É a mesma tarefa para as quatro famílias do Meeting01: SNN, LSTM, GRU e "
            "Transformer. A janela é sintética e normalizada como no software/nn: média 0, desvio 1.",
            f"x \\in R^{{{WINDOW}}} \\to z \\in R^{{{k}}} \\to \\hat{{x}} \\in R^{{{WINDOW}}}",
        )

        x, guess = float(window[sample]), float(pca.mean[sample])
        mse_step = frame(
            "Erro de reconstrução: o MSE",
            "Como medir 'deu certo'? Amostra a amostra: erro = original − reconstrução, ao quadrado; "
            f"o MSE é a média nas {WINDOW}. O palpite mais preguiçoso devolve a janela média do treino "
            f"(≈ 0 em toda amostra). Na amostra {sample} o original vale {x:.2f} e o palpite "
            f"{guess:.2f}: erro² = {(x - guess) ** 2:.2f}. Na média das {WINDOW}, MSE = {mean_mse:.2f}. "
            "Como a janela tem desvio 1, MSE ≈ 1 quer dizer 'não aprendeu nada' — a linha tracejada "
            "lá embaixo.",
            "MSE = \\dfrac{1}{256} sum_{i} (x_i - \\hat{x}_i)^2",
            recon_reveal=1.0, error_reveal=1.0, sample_reveal=1.0, bottom_reveal=1.0,
        )

        examples = "; ".join(f"com {j}, {mse_all[j - 1]:.3f}" for j in sorted({1, 4, k}) if j <= k)
        pca_text = (
            f"A PCA, ajustada só nas {_N_TRAIN} janelas de treino, acha as k direções que mais explicam "
            f"o sinal e guarda a coordenada da janela em cada uma. MSE {examples}. Em média no treino, "
            f"nenhum compressor LINEAR com {k} números erra menos (Eckart–Young). O software/nn roda "
            "essa mesma PCA em cada dobra como régua — com o k do latente, contra a janela original e "
            "ajustada em treino + validação, os mesmos dados do ajuste final das famílias: uma família "
            "que não fica abaixo dela não comprou nada com a não-linearidade."
        )
        pca_label = f"PCA: o melhor compressor linear com {k} números"
        pca_step = frame(pca_label, pca_text, _PCA_EQUATION, sample_reveal=1.0, **pca_values(k))
        pca_sweep = [
            frame(pca_label, pca_text, _PCA_EQUATION, checkpoint=False, sample_reveal=1.0, **pca_values(j))
            for j in _sweep(1, k)
        ]

        ratio = mse_all[k - 1] / mse_all[k_wide - 1]
        wide_text = (
            f"Agora k vai de {k} a {k_wide}: o MSE cai de {mse_all[k - 1]:.3f} para "
            f"{mse_all[k_wide - 1]:.3f} — {ratio:.1f}× menor, sem inteligência nenhuma a mais, só um "
            "gargalo mais largo. Por isso o software/nn fixa o latente por dataset e nunca o deixa "
            f"evoluir: áudio {LATENT_AUDIO} ({WINDOW // LATENT_AUDIO}:1), EEG {LATENT_EEG} "
            f"({WINDOW // LATENT_EEG}:1). Dar mais números a uma família que a outra decidiria o "
            "vencedor antes do treino — e as duas tabelas pareceriam normais. Falha silenciosa."
        )
        wide_label = "O tamanho do latente decide o placar"
        wide_step = frame(wide_label, wide_text, _PCA_EQUATION, domain_reveal=1.0, **pca_values(k_wide))
        wide_sweep = [
            frame(wide_label, wide_text, _PCA_EQUATION, checkpoint=False, domain_reveal=1.0, **pca_values(j))
            for j in _sweep(k + 1, k_wide)
        ]

        settled = pca_values(k)
        target_step = frame(
            "Mesmo alvo, ou o MSE mente",
            "Um MSE só compara modelos que reconstroem o MESMO alvo. No Meeting01, o alvo da SNN já foi "
            "o próprio código de pulsos — e um modelo que não aprende nada marcava "
            f"{TRIVIAL_MSE_BY_TARGET['direta']:.3f} com codificação direta, "
            f"{TRIVIAL_MSE_BY_TARGET['Poisson']:.3f} com Poisson e {TRIVIAL_MSE_BY_TARGET['latência']:.4f} "
            "com latência, porque um alvo quase todo zero tem pouca variância. A busca 'descobriu' que "
            f"latência era {B2_FREE_WIN}× melhor, de graça. Esses números são do codificador da época, que "
            f"usava as {OLD_LATENCY_TIME_AXIS} amostras da janela como eixo de tempo (latência: cerca de 1 "
            f"disparo a cada {OLD_LATENCY_TIME_AXIS} posições, 99,6% de zeros). Com o de hoje, T = 16, seria 1 disparo em 16 "
            f"— variância ≈ {LATENCY_TRIVIAL_MSE_T16:.3f}, ainda ~{1 / LATENCY_TRIVIAL_MSE_T16:.0f}× abaixo da "
            "direta: a armadilha não depende do T. A correção: toda família reconstrói a janela original.",
            "MSE_{trivial} = Var(\\text{alvo})",
            bottom="targets", **settled,
        )

        cost = EPOCH_SECONDS
        families_step = frame(
            "Quatro famílias, quatro jeitos de ler a janela",
            "SNN: a janela é apresentada ao longo de T = 16 passos — como corrente analógica (codificação "
            "direta, sem pulsos na entrada) ou como pulsos (Poisson ou latência); a codificação também é "
            "buscada. LSTM e GRU: leem a janela em quadros, um após o outro, "
            "levando memória num estado oculto (a GRU com menos portas). Transformer: vê todos os "
            "quadros de uma vez e pesa cada par por atenção. Custo medido por época neste framework "
            f"(CPU, lote 1, redes típicas): SNN {cost['SNN-AE']:g} s; Transformer "
            f"{cost['Transformer-AE']:.0f} s; LSTM {cost['LSTM-AE']:.0f} s; GRU {cost['GRU-AE']:.0f} s.",
            "h_t = f(x_t, h_{t-1})\\ \\text{(LSTM, GRU)};\\quad softmax(Q K^T) V\\ \\text{(Transformer)}",
            bottom="cost", **settled,
        )

        verdict_step = frame(
            "Quem vence? Só o experimento diz",
            "No Meeting01 cada família busca a própria arquitetura (NSGA-II) com o mesmo latente; a "
            "vencedora é avaliada uma única vez num locutor ou sujeito nunca visto (6 dobras, um grupo "
            "de fora por vez), e a estatística é pareada por gravação — janelas da mesma gravação não "
            "são independentes. Para valer, a família precisa ficar abaixo da PCA. Este app não treina "
            "as quatro: as barras ficam vazias de propósito — um número inventado aqui viraria "
            "'resultado' numa aula.",
            "d_r = MSE_{familia}(r) - MSE_{PCA}(r)",
            bottom="verdict", **settled,
        )

        frames = [intro]
        frames += transition(intro, mse_step, _STEPS)
        frames.append(mse_step)
        frames += pca_sweep  # one exact frame per k, no interpolation
        frames.append(pca_step)
        frames += wide_sweep
        frames.append(wide_step)
        # Different chart below from here on: jump, then settle back at k.
        frames.append(target_step)
        frames.append(families_step)
        frames.append(verdict_step)
        return frames
