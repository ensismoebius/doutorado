"""
Gera ``simulatedDysphonia.pdf``: uma locucao real da base de Coretto degradada pela disfonia
simulada da tese, nos niveis 0 (voz limpa), 0,5 e 1 (sussurro), e o espectro de um quadro
vozeado limpo e sussurrado.

A degradacao NAO e reimplementada aqui: o script compila o proprio
``software/nn/src/experiments/thesis/lib/src/VoiceDegradation.cpp`` (o mesmo arquivo do
experimento) numa biblioteca compartilhada e chama ``thesis::degrade_voice`` por ctypes.
O que a figura mostra e, portanto, o que o experimento fez com essa locucao, a menos da
semente: no experimento ela e a do perfil misturada ao indice da amostra
(``degrade_voices`` em thesis.cpp); aqui e fixa.

Uso:
    python3 generate_dysphonia_figure.py --db ~/database.sqlite [--out-dir images]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sqlite3
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import signal  # noqa: E402

# Virgula decimal, como no texto da tese.
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "software", "nn", "src",
                                    "experiments", "thesis", "lib"))
SUBJECT, STIMULUS = "S01", "Arriba"   # uma palavra, com vogais e consoantes
LEVELS = (0.0, 0.5, 1.0)
SEED = 7                              # DEGRADATION_SEED dos perfis
FMAX = 5000.0                         # Hz dos espectrogramas: os primeiros formantes
# Paleta Okabe-Ito, a mesma de chapters/data/visualStyle.tex (cbBlue, cbVermillion, ...).
CB_BLUE = (0 / 255, 114 / 255, 178 / 255)
CB_SKYBLUE = (86 / 255, 180 / 255, 233 / 255)
CB_VERMILLION = (213 / 255, 94 / 255, 0 / 255)
CB_GREY = (166 / 255, 166 / 255, 166 / 255)
# Espectrograma: branco -> azul-celeste -> azul -> azul-escuro (claridade monotonica).
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "okabe_blues", [(1, 1, 1), CB_SKYBLUE, CB_BLUE, (0, 57 / 255, 89 / 255)])
FMAX_FRAME = 2500.0                   # Hz do espectro do quadro: onde os harmonicos se separam

WRAPPER = r"""
#include "VoiceDegradation.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>
extern "C" int degrade(const float* x, std::size_t n, double fs, float level,
                       std::uint32_t seed, float* out)
{
    try {
        const std::vector<float> in(x, x + n);
        const std::vector<float> y = thesis::degrade_voice(in, fs, level, seed);
        for (std::size_t i = 0; i < n; ++i) out[i] = y[i];
        return 0;
    } catch (...) { return 1; }
}
"""


def build(tmp):
    src = os.path.join(tmp, "wrap.cpp")
    with open(src, "w") as f:
        f.write(WRAPPER)
    so = os.path.join(tmp, "libdegrade.so")
    subprocess.run(["g++", "-std=c++20", "-O2", "-shared", "-fPIC",
                    "-I", os.path.join(LIB, "include"), src,
                    os.path.join(LIB, "src", "VoiceDegradation.cpp"), "-o", so], check=True)
    lib = ctypes.CDLL(so)
    lib.degrade.restype = ctypes.c_int
    lib.degrade.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_double,
                            ctypes.c_float, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
    return lib


def degrade(lib, x, fs, level):
    if level == 0.0:
        return x.copy()
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    fp = ctypes.POINTER(ctypes.c_float)
    if lib.degrade(x.ctypes.data_as(fp), x.size, fs, level, SEED, out.ctypes.data_as(fp)):
        raise SystemExit("thesis::degrade_voice refused the signal")
    return out


def utterance(db):
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT a.samples FROM trial t JOIN subject s ON s.id = t.subject_id "
        "JOIN stimulus st ON st.id = t.stimulus_id JOIN modalities m ON m.id = t.modality_id "
        "JOIN audio_samples a ON a.trial_id = t.id "
        "WHERE s.subject_name = ? AND st.name = ? AND m.name = 'Pronounced' "
        "ORDER BY t.original_row LIMIT 1", (SUBJECT, STIMULUS)).fetchall()
    con.close()
    if not rows:
        raise SystemExit(f"{db}: no pronounced {STIMULUS} of {SUBJECT}")
    return np.frombuffer(rows[0][0], dtype=np.float32).copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="a base de Coretto (database.sqlite)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    fs = 44100.0
    x = utterance(os.path.expanduser(args.db))
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(tmp)
        ys = [degrade(lib, x, fs, lv) for lv in LEVELS]
    for lv, y in zip(LEVELS, ys):   # a mistura preserva a potencia (secao da tese)
        print(f"level {lv}: power ratio {np.mean(y.astype(float)**2) / np.mean(x.astype(float)**2):.3f}")

    # Trecho falado: onde a energia de curto prazo passa de 1 % do maximo, com margem.
    frame = int(0.025 * fs)
    env = np.convolve(x.astype(float) ** 2, np.ones(frame) / frame, mode="same")
    on = np.flatnonzero(env > 0.01 * env.max())
    a, b = max(on[0] - int(0.1 * fs), 0), min(on[-1] + int(0.1 * fs), x.size)
    # Quadro vozeado de 25 ms: o mais periodico (maior pico da autocorrelacao normalizada
    # entre 70 e 400 Hz) entre os que tem ao menos 10 % da energia maxima. O de maior
    # energia caia num ataque consonantal, sem harmonicos.
    best, c = -1.0, None
    lo, hi = int(fs / 400), int(fs / 70)
    for centre in range(a + frame, b - frame, int(0.01 * fs)):
        if env[centre] < 0.1 * env.max():
            continue
        seg = x[centre - frame // 2:centre + frame // 2].astype(float)
        seg = seg - seg.mean()
        r = np.correlate(seg, seg, mode="full")[frame - 1:]
        score = r[lo:hi].max() / r[0]
        if score > best:
            best, c = score, centre
    q = slice(c - frame // 2, c + frame // 2)
    print(f"voiced frame at {c / fs:.3f} s, periodicity {best:.2f}")

    fig = plt.figure(figsize=(6.3, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.55, wspace=0.12)
    titles = ["(a) nível 0: voz limpa", "(b) nível 0,5", "(c) nível 1: sussurro"]
    vmax = None
    for i, (y, title) in enumerate(zip(ys, titles)):
        ax = fig.add_subplot(gs[0, i])
        f, t, S = signal.spectrogram(y[a:b].astype(float), fs=fs, window="hann",
                                     nperseg=1024, noverlap=768)
        keep = f <= FMAX
        db = 10 * np.log10(S[keep] + 1e-12)
        vmax = db.max() if vmax is None else vmax
        ax.pcolormesh(t, f[keep] / 1000, db, shading="auto", cmap=BLUES,
                      vmin=vmax - 55, vmax=vmax, rasterized=True)
        ax.axvline((c - a) / fs, color=CB_VERMILLION, linewidth=1.0)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("tempo (s)", fontsize=7.5)
        if i == 0:
            ax.set_ylabel("frequência (kHz)", fontsize=7.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(COMMA)
        if i == 0:
            ax.yaxis.set_major_formatter(COMMA)

    ax = fig.add_subplot(gs[1, :])
    win = np.hanning(frame)
    for y, lab, sty in ((ys[0], "voz limpa (nível 0)", dict(color=CB_BLUE, linewidth=1.0)),
                        (ys[2], "sussurro (nível 1)", dict(color=CB_VERMILLION, linewidth=1.0))):
        spec = np.abs(np.fft.rfft(y[q].astype(float) * win)) ** 2
        freqs = np.fft.rfftfreq(frame, 1 / fs)
        keep = freqs <= FMAX_FRAME
        ax.plot(freqs[keep] / 1000, 10 * np.log10(spec[keep] + 1e-12), label=lab, **sty)
    ax.set_xlim(0, FMAX_FRAME / 1000)
    ax.set_xlabel("frequência (kHz)", fontsize=7.5)
    ax.set_ylabel("potência (dB)", fontsize=7.5)
    ax.set_title("(d) espectro do quadro de 25 ms marcado em vermelho-alaranjado", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(COMMA)
    ax.yaxis.set_major_formatter(COMMA)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "simulatedDysphonia.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=200,
                metadata={"CreationDate": None, "ModDate": None})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
