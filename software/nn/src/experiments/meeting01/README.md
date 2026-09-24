Instalação/deploy do meeting01 no GridUnesp — passo a passo:

0. Conta (humano). Coordenador (afiliado UNESP, ex: orientador) registra projeto em unesp.br/portal#!/gridunesp/submissao-de-projetos/ ou grid@ncc.unesp.br (form src/experiments/meeting01/docs/formularioGridUnesp.odt já pronto). Depois você se registra em ncc.unesp.br/registration/. ~1-2 dias úteis.

1. Deploy (um comando).


cd software/nn
./scripts/pipeline/meeting01/gridunesp_deploy.sh
Faz tudo sozinho: pede credencial uma vez (cacheia em .env, git-ignored), rsync do checkout pro cluster, bootstrap de env conda meeting01-build (supre OpenBLAS/ninja/GCC13 que os modules do grid não têm), baixa os 4 datasets (ensure_datasets.sh, ~24GB), cmake --preset=max-performance, build cruzado num compute node (srun, pra -march=native bater com o Xeon E5-2680 v4 real — build NÃO pode rodar no login node). Termina com smoke test meeting01 --help e imprime o comando sbatch exato. Não submete o job longo sozinho.

2. Validar toolchain local antes (opcional, recomendado).


./scripts/pipeline/meeting01/run_gridunesp_docker_sim.sh
Container AlmaLinux+conda simulando o gap de toolchain, sem gastar fila real.

3. Submeter o job real.


sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch
squeue -u $USER
Fila long (30 dias, hard cap), 28 cores, job-nanny envolvendo o 01_meeting01_run_loso.sh sem modificação. RESUME=1 sbatch ... retoma run interrompido.

4. Monitorar.


./scripts/pipeline/meeting01/remote_monitor.sh   # dashboard live via SSH
# ou
./scripts/pipeline/meeting01/pull_progress.sh    # rsync resultados pra local
5. Trazer resultados de volta.


rsync -avz user@access.grid.unesp.br:software/nn/results/meeting01/ results/meeting01/
EXPERIMENT_CONFIRMED=1 RESUME=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
Fold já completo → pula treino, roda pós-processamento (03_/02_/04_) local.

Fonte canônica, ler antes da submissão real: .wiki/Guides/GridUnesp-Deployment.md. Dois pontos em aberto lá: acesso à internet dos compute nodes (não confirmado) e semântica das vars INPUT/OUTPUT do job-nanny — vale testar com job curto (meeting01 --help na fila short) antes de comprometer 30 dias de fila.

Nota: EXPERIMENT_CONFIRMED=1 obrigatório (guard do hook) — já embutido no sbatch script.


# meeting01 — Nested-LOSO comparative autoencoder study

Profile-driven comparative experiment: SNN, LSTM, GRU, and Transformer autoencoders, each
with its own genetic-algorithm architecture search, compared on FSDD / AudioMNIST /
eegmmidb / Siena Scalp EEG under nested leave-one-speaker/recording-out cross-validation.

Full documentation lives in the wiki — this file is just a build/run pointer:
[.wiki/Experiments/Meeting01.md](../../../.wiki/Experiments/Meeting01.md).

> The original design (four fixed-architecture `article-*.json` profiles, one LSTM baseline
> vs. three SNN variants, split by pooling+shuffling every window) was **deleted
> 2026-09-23**: it never set `dataset.cv_fold`, so the same speaker/recording could land in
> both train and validation — the leakage defect a reviewer flagged as strong-reject on
> submission 71. `dataset.cv_fold` is now required everywhere; there is no non-LOSO
> fallback left in the code.

## Build

```bash
cd software/nn
cmake --preset=max-performance
cmake --build --preset=max-performance --target meeting01 -j"$(nproc)"
```

## Run

```bash
# Single (dataset, fold) slice
./out/build/max-performance/src/experiments/meeting01/meeting01 \
  --comparative-config src/experiments/meeting01/profiles/meeting01-loso.json \
  --dataset fsdd --cv-fold 0

# Full nested-LOSO grid + paper post-processing (weeks-scale — see Re-run Runbook)
EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
```

See [Re-run Runbook](../../../.wiki/Guides/Re-run-Runbook.md) for `RESUME=1`,
`SKIP_BUILD=1`, `SKIP_POSTPROCESS=1`, and the full post-processing chain (03_/02_/04_).

## Running on GridUnesp (UNESP's HPC cluster)

```bash
# One-shot: sync checkout, bootstrap env, fetch datasets, configure, build.
# Idempotent, safe to re-run. Stops before submitting -- prints the sbatch command.
./scripts/pipeline/meeting01/gridunesp_deploy.sh

# Monitor a running job without an interactive login shell:
./scripts/pipeline/meeting01/remote_monitor.sh   # live, --plain, one SSH session
./scripts/pipeline/meeting01/pull_progress.sh    # sync results/ down, use local monitor.py
```

All three prompt interactively for your GridUnesp username and, unless an SSH key
is already set up, your password (at `ssh`'s own prompt — never captured or stored
by these scripts). Set `GRIDUNESP_USER=<user>` beforehand to skip the prompt.

Full runbook, cluster facts, and open questions:
[GridUnesp Deployment](../../../.wiki/Guides/GridUnesp-Deployment.md).

## Profiles (`profiles/`)

`meeting01-loso.json` is the only production profile. Everything else in `profiles/` is a
dev/smoke fixture (small caps, `time_steps=4`) used by tests and local iteration — see the
directory audit in `tests/profile_audit_gtest.cpp` for what each one is for.

## Profile audit tests

```bash
cmake --build out/build/max-performance --target profile_audit_gtest -j"$(nproc)"
ctest --test-dir out/build/max-performance -R profile_audit --output-on-failure
```

Every profile in `profiles/` is checked directory-wide (parses + `validate()` doesn't
throw); `meeting01-loso.json` additionally gets a fuller, hand-maintained set of checks.

## Key source files

Current module layout (`lib/include/`, `lib/src/`) — see the wiki page's
[Implementation](../../../.wiki/Experiments/Meeting01.md#implementation) section and
[Multi-family architecture search](../../../.wiki/Experiments/Meeting01.md#multi-family-architecture-search-added-2026-09-22-same-day-later-scope-change)
for what each does; the short version:

| File | Role |
|---|---|
| `meeting01.cpp` | Thin CLI entry point |
| `lib/include/Meeting01Cli*.hpp`, `lib/src/Meeting01Cli.cpp` | CLI parsing |
| `lib/include/Meeting01Config.hpp`, `lib/src/Meeting01Config.cpp` | Profile JSON schema + validation |
| `lib/include/Meeting01Dataset*.hpp`, `lib/src/Meeting01Dataset.cpp` | Dataset loading, nested-LOSO split |
| `lib/src/Meeting01Eeg.cpp`, `lib/src/Meeting01MitBih.cpp` | Per-dataset loaders (eegmmidb/siena, mitbih) |
| `lib/include/Meeting01Encoding.hpp`, `lib/src/Meeting01Encoding.cpp` | Spike encoding (direct/poisson/latency), LSTM framing |
| `lib/include/Meeting01GaGenome.hpp` + family variants, `lib/src/Meeting01Ga*.cpp` | Per-family genetic architecture search (SNN, recurrent LSTM/GRU, Transformer) |
| `lib/include/Meeting01Training.hpp`, `lib/src/Meeting01Training.cpp` | Training loop |
| `lib/src/Meeting01Experiment.cpp` | Per-fold orchestration (search → retrain winners → manifest) |
| `lib/include/Meeting01Metrics.hpp`, `lib/src/Meeting01Metrics.cpp` | Reconstruction metrics, cost proxies |
| `lib/include/Meeting01Output.hpp`, `lib/src/Meeting01Output.cpp` | CSV/manifest writers |

## SNN architecture note

`snn_architectures: ["dense", "conv1d", "recurrent"]` in a profile selects the **input
transform**, not a fixed network topology — the SNN's actual layer widths are chosen by
the genetic search. See the wiki's NSGA-II / multi-family search sections for details.
