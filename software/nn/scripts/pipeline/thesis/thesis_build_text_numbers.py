#!/usr/bin/env python3
"""thesis_build_text_numbers.py — every number the thesis PROSE quotes, as LaTeX data.

The problem. The tables of the results chapter are typeset from CSVs, so they follow a
re-run. The sentences around them did not: 58 numbers were typed by hand into chapter 9
and 11 into the conclusions, and after two re-runs most of them described experiments that
no longer existed (an EER of 0.4534 from July sat in the abstract next to tables that said
0.22). A hand-typed number fails silently: the PDF compiles, and nothing says it is stale.

What this writes. One file, `tables/thesis_numbers.tex`, that defines a value per KEY:

    \\tn{p01.grid.lateimag.dep.flat.std.eer}     ->  0{,}2205     (decimal comma, for the text)
    \\tnen{p01.grid.lateimag.dep.flat.std.eer}   ->  0.2205       (decimal point, the abstract in English)

and the prose quotes the key. A key that does not exist is a LaTeX ERROR (the \\tn macro in
chapters/data/visualStyle.tex refuses it), so a sentence cannot outlive the result it quotes.

The keys (printed with --list):

    p00.<signal>.<family>.<rank>.<col>      a cell of a Phase 00 table; signal voice|eegpron|
                                            eegimag, family hc|ae|pca, col label|dpen|sd|loso|
                                            wins|d|alpha|beta
    p00.<signal>.winner.<col>               the signal's winner (rank 1 over the three families)
    p00.<signal>.runnerup.<col>, .margin    the second of the signal and its distance
    p00.<signal>.best.<family>.<col>        the best of one family
    p00.<signal>.stab.<kind>.wins|n|worst   in how many speaker subsets the winner is rank 1
                                            (kind fold|loso), of how many, and its worst rank
    p00.<signal>.stab.<kind>.alt            how many other extractors win some subset
    p00.<signal>.c1.*, .c2.*                non-cepstral vs cepstral handcrafted: n, mean, best,
                                            worst
    p00.voice.scale.<linear|mel|bark>.best  best non-cepstral voice extractor per scale
    p00.<signal>.ae.beta.min|max, .pca.beta.min|max
    p00.<signal>.enc.<direct|latency|poisson>.mean|best
    p00.n.<signal>, p00.n.total             profile counts
    p01.grid.<source>.<text>.<cv>.<std>.<m> the DSNN grid; m eer|eersd|auc|aucsd|rank
    p01.ann.<source>.<text>.<m>, p01.cosine.<source>.<text>.<m>
    p01.dysph.<source>.<scope>.<level>.eer  whisper level 25|50|75|100 (level 0 = the grid cell)
    p01.cmp.<signal>.<text>.<family>.<m>    the extractor families through the same DSNN
    p01.energy.<source>.<quantity>          nJ per test sample, and the DSNN/ANN ratios
    p01.best.*, p01.text.*, p01.std.*, p01.cv.*, p01.energy.ratio.*   summaries over the grid
    p01.n.<set>, p01.n.total                profile counts
    sig.<file stem>.<condition>.<col>       a row of a significance CSV: delta|d|p|pholm
    aeinput.<signal>.<old|new>.<clf>        speaker identification from the autoencoders' input
                                            (scripts/analysis/thesis_ae_input_reference.py):
                                            old = block-averaged waveform, new = power spectrum;
                                            clf pca8|pca32|all; aeinput.chance, aeinput.trials
    bx.<kind>.<source>.<text>.<m>           the BIOMEX-DB replication (kind dsnn|ann|cosine|dysph),
    bx.vs.coretto.*, bx.best.*, bx.energy.*  its comparison with the same Coretto cell
    bxsig.*, bxsiggroup.*                   its significance (04_thesis_biomex_significance.sh)
    base.biomex.*, base.coretto.*           the two bases: volunteers kept and excluded, pairs,
                                            verification counts, power below 2 Hz
                                            (scripts/analysis/thesis_base_facts.py, read from the
                                            bases and the verifiers' stamps)
    cal.*                                   the WORLD dysphonia calibration measured with Praat
                                            (05_thesis_dysphonia_calibration.py): versions, voices,
                                            raw and control measures per speaker, read-back ratios,
                                            the shimmer ceiling, the calibrated levels
    pilot.*                                 the identity pilot verdict (06_thesis_identity_pilot.py):
                                            per classifier and scope, mean dEER, interval, verdict

The sentences, not only the numbers. A number that follows the data is not enough: "the
same extractor wins on the three signals" has no number in it, and a re-run can make it
false while every \\tn in the paragraph still compiles. So each qualitative statement the
prose makes is written in CLAIMS below as a condition on the keys, next to the sentence it
protects, and a re-run that breaks one stops this script with the list of broken
sentences. The chapter is then rewritten on purpose, not left wrong by accident.

Nothing is guessed or skipped: every argument is required, an incomplete result set is
refused (the collectors of the two table scripts do that), and a duplicate key is an error.

Usage:
    python3 scripts/pipeline/thesis/thesis_build_text_numbers.py \\
        --phase00-profiles src/experiments/thesis/profiles/phase00 \\
        --phase00-results  results/thesis/phase00 \\
        --phase01-profiles src/experiments/thesis/profiles/phase01 \\
        --phase01-results  results/thesis/phase01 \\
        --ae-input-reference results/thesis/ae_input_reference.csv \\
        --base-facts results/thesis/base_facts.csv \\
        --biomex-profiles src/experiments/thesis/profiles/biomex \\
        --biomex-results results/thesis/biomex \\
        --dysphonia-calibration results/thesis/dysphonia_calibration/calibration.json \\
        --identity-pilot results/thesis/pilot/pilot_verdict.json \\
        --identity-margin results/thesis/identity_margin/identity_margin_verdict.json \\
        --out ../../documentation/00-thesis/monography/tables/thesis_numbers.tex [--list]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P00 = load("thesis_build_phase00_paraconsistent_tables")
P01 = load("thesis_build_phase01_auth_tables")
BX = load("thesis_build_biomex_tables")

SIGNALS = ("voice", "eegpron", "eegimag")
FAMILIES = ("hc", "ae", "pca")


def fail(msg: str) -> None:
    sys.exit(f"[text-numbers] ERROR: {msg}")


class Numbers:
    """key -> (text for Portuguese prose, text for English prose)."""

    def __init__(self) -> None:
        self.values: dict[str, tuple[str, str]] = {}

    def put(self, key: str, pt: str, en: str | None = None) -> None:
        if not re.fullmatch(r"[a-z0-9_.\-]+", key):
            fail(f"key {key!r} has characters a LaTeX \\csname cannot be trusted with")
        if key in self.values:
            fail(f"duplicate key {key}")
        self.values[key] = (pt, pt if en is None else en)

    def num(self, key: str, value: float, digits: int = 4) -> None:
        en = f"{value:.{digits}f}"
        self.put(key, en.replace(".", "{,}"), en)

    def count(self, key: str, value: int) -> None:
        self.put(key, str(value))

    def text(self, key: str, value: str) -> None:
        self.put(key, value)


def phase00(n: Numbers, profiles: pathlib.Path, results: pathlib.Path) -> None:
    tables = P00.collect(profiles, results)
    cols = (("dpen", "mean_d_penalized"), ("d", "mean_d_truth"), ("alpha", "mean_alpha"),
            ("beta", "mean_beta"), ("loso", "loso_sd"))

    def cells(prefix: str, row: dict) -> None:
        n.text(f"{prefix}.label", row["label"])
        for short, col in cols:
            n.num(f"{prefix}.{short}", row[col])
        if row["std_d_penalized"] is not None:
            n.num(f"{prefix}.sd", row["std_d_penalized"])
        n.count(f"{prefix}.wins", row["fold_wins"])

    total = 0
    for s in SIGNALS:
        every = []
        for fam in FAMILIES:
            rows = tables.get(f"{fam}_{s}")
            if not rows:
                fail(f"no Phase 00 rows for {fam}_{s}")
            every.extend(rows)
            for rank, row in enumerate(rows, start=1):
                cells(f"p00.{s}.{fam}.{rank}", row)
            cells(f"p00.{s}.best.{fam}", rows[0])
            n.count(f"p00.n.{s}.{fam}", len(rows))
        n.count(f"p00.n.{s}", len(every))
        total += len(every)
        ranked = sorted(every, key=lambda r: r["mean_d_penalized"])
        cells(f"p00.{s}.winner", ranked[0])
        cells(f"p00.{s}.runnerup", ranked[1])
        n.num(f"p00.{s}.margin", ranked[1]["mean_d_penalized"] - ranked[0]["mean_d_penalized"])
        n.num(f"p00.{s}.spread.top55",
              ranked[min(54, len(ranked) - 1)]["mean_d_penalized"] - ranked[0]["mean_d_penalized"])
        # The extractor with the smallest UNPENALIZED distance: it differs from the winner
        # when the penalty on contradiction decides the selection.
        cells(f"p00.{s}.mind", min(every, key=lambda r: r["mean_d_truth"]))

        # Winner stability over the speaker subsets (the subsets every profile scored).
        for kind in ("fold", "loso"):
            keys = sorted(k for k in ranked[0]["subsets"] if k.startswith(f"{kind}:"))
            wins, worst, alt = 0, 1, set()
            for k in keys:
                order = sorted(every, key=lambda r: r["subsets"][k])
                rank = 1 + sum(1 for r in every
                               if r["subsets"][k] < ranked[0]["subsets"][k] - 1e-5)
                wins += rank == 1
                worst = max(worst, rank)
                if rank != 1:
                    alt.add(order[0]["label"])
            n.count(f"p00.{s}.stab.{kind}.wins", wins)
            n.count(f"p00.{s}.stab.{kind}.n", len(keys))
            n.count(f"p00.{s}.stab.{kind}.losses", len(keys) - wins)
            n.count(f"p00.{s}.stab.{kind}.worst", worst)
            n.count(f"p00.{s}.stab.{kind}.alt", len(alt))
            if kind == "fold":
                # What the subset winners share, read off their labels ("Wavelet + Scale (cN)").
                n.count(f"p00.{s}.stab.fold.alt.c1", sum(1 for a in alt if a.endswith("(c1)")))
                n.count(f"p00.{s}.stab.fold.alt.linear",
                        sum(1 for a in alt if "+ Linear " in a))

        # Cepstral (c2) against non-cepstral (c1) handcrafted extractors.
        hc = tables[f"hc_{s}"]
        for cat in ("c1", "c2"):
            ds = [r["mean_d_penalized"] for r in hc if r["label"].endswith(f"({cat})")]
            if not ds:
                fail(f"no {cat} handcrafted rows for {s}")
            n.count(f"p00.{s}.{cat}.n", len(ds))
            n.num(f"p00.{s}.{cat}.mean", statistics.mean(ds))
            n.num(f"p00.{s}.{cat}.best", min(ds))
            n.num(f"p00.{s}.{cat}.worst", max(ds))
        c1 = [r["mean_d_penalized"] for r in hc if r["label"].endswith("(c1)")]
        c2 = [r["mean_d_penalized"] for r in hc if r["label"].endswith("(c2)")]
        n.count(f"p00.{s}.c2.better_than_worst_c1", sum(1 for d in c2 if d < max(c1)))

        for fam in ("ae", "pca"):
            betas = [r["mean_beta"] for r in tables[f"{fam}_{s}"]]
            n.num(f"p00.{s}.{fam}.beta.min", min(betas))
            n.num(f"p00.{s}.{fam}.beta.max", max(betas))
        for enc in ("direct", "latency", "poisson"):
            ds = [r["mean_d_penalized"] for r in tables[f"ae_{s}"]
                  if r["label"].startswith(f"SNN-AE {enc} ")]
            if not ds:
                fail(f"no SNN-AE {enc} rows for {s}")
            n.num(f"p00.{s}.enc.{enc}.mean", statistics.mean(ds))
            n.num(f"p00.{s}.enc.{enc}.best", min(ds))
        ann = [r["mean_d_penalized"] for r in tables[f"ae_{s}"] if r["label"].startswith("ANN-AE")]
        n.num(f"p00.{s}.enc.ann.mean", statistics.mean(ann))
        # PCA against the best autoencoder of the same latent size (k = 8, 16, 32).
        sizes = {"tiny": 8, "small": 16, "base": 32}
        pca_wins = 0
        for size, k in sizes.items():
            best_ae = min(r["mean_d_penalized"] for r in tables[f"ae_{s}"]
                          if r["label"].endswith(f"({size})"))
            pca = next(r["mean_d_penalized"] for r in tables[f"pca_{s}"]
                       if r["label"] == f"PCA (k = {k})")
            n.num(f"p00.{s}.size.{k}.ae", best_ae)
            n.num(f"p00.{s}.size.{k}.pca", pca)
            pca_wins += pca < best_ae
        n.count(f"p00.{s}.size.pca_wins", pca_wins)

    for scale in ("Linear", "Mel", "Bark"):
        ds = [r["mean_d_penalized"] for r in tables["hc_voice"]
              if f"+ {scale} (c1)" in r["label"]]
        if not ds:
            fail(f"no non-cepstral voice rows for scale {scale}")
        n.num(f"p00.voice.scale.{scale.lower()}.best", min(ds))
        n.num(f"p00.voice.scale.{scale.lower()}.mean", statistics.mean(ds))
    bests = [float(n.values[f"p00.voice.scale.{x}.best"][1]) for x in ("linear", "mel", "bark")]
    n.num("p00.voice.scale.spread", max(bests) - min(bests))
    n.count("p00.n.total", total)


def phase01(n: Numbers, profiles: pathlib.Path, results: pathlib.Path) -> None:
    sets = P01.collect(profiles, results)

    def metrics(prefix: str, run: dict) -> None:
        n.num(f"{prefix}.eer", run["mean_eer"])
        n.num(f"{prefix}.auc", run["mean_auc"])
        if run["std_eer"] is not None:
            n.num(f"{prefix}.eersd", run["std_eer"])
            n.num(f"{prefix}.aucsd", run["std_auc"])

    grid = sets["grid"]
    order = sorted(grid, key=lambda k: grid[k]["mean_eer"])
    for rank, key in enumerate(order, start=1):
        prefix = "p01.grid." + ".".join(key)
        metrics(prefix, grid[key])
        n.count(f"{prefix}.rank", rank)
    for kind in ("ann", "cosine"):
        for key, run in sets[kind].items():
            metrics(f"p01.{kind}." + ".".join(key), run)
    for (src, scope, level), run in sets["dysph"].items():
        metrics(f"p01.dysph.{src}.{scope}.{level}", run)
    for (src, text, fam), run in sets["cmp"].items():
        metrics(f"p01.cmp.{src}.{text}.{fam}", run)
        n.text(f"p01.cmp.{src}.{text}.{fam}.label",
               P01.extractor_label(run["profile"]["feature_extraction"]))
    for kind, runs in sets.items():
        n.count(f"p01.n.{kind}", len(runs))
    n.num("p01.min.eer", min(r["mean_eer"] for runs in sets.values() for r in runs.values()))
    n.count("p01.n.total", sum(len(r) for r in sets.values()))
    n.count("p01.n.units", sum(r["n_reps"] for runs in sets.values() for r in runs.values()))

    sources = P01.SOURCE_ORDER
    best = order[0]
    n.text("p01.best.label", f"{P01.SOURCE_LABELS[best[0]]}, {P01.TEXT_LABELS[best[1]]}, "
                             f"{P01.CV_LABELS[best[2]]}, {P01.STD_LABELS[best[3]]}")
    metrics("p01.best", grid[best])
    voice = [k for k in order if k[0] == "voice"]
    n.count("p01.voice.first_rank", order.index(voice[0]) + 1)
    metrics("p01.voice.best", grid[voice[0]])

    def mean_eer(keys) -> float:
        return statistics.mean(grid[k]["mean_eer"] for k in keys)

    # Standardization: std against raw over the 28 (source, text, cv) pairs.
    pairs = [(s, t, c) for s in sources for t in ("dep", "indep") for c in ("flat", "nested")]
    n.count("p01.std.pairs", len(pairs))
    n.count("p01.std.better", sum(1 for p in pairs
                                  if grid[p + ("std",)]["mean_eer"] < grid[p + ("raw",)]["mean_eer"]))
    n.num("p01.std.mean.std", mean_eer(p + ("std",) for p in pairs))
    n.num("p01.std.mean.raw", mean_eer(p + ("raw",) for p in pairs))
    for group, members in (("voice", ["voice"]), ("eegfusion", [s for s in sources if s != "voice"])):
        sub = [p for p in pairs if p[0] in members]
        n.num(f"p01.std.{group}.std", mean_eer(p + ("std",) for p in sub))
        n.num(f"p01.std.{group}.raw", mean_eer(p + ("raw",) for p in sub))
        raws = [grid[p + ("raw",)]["mean_eer"] for p in sub]
        n.num(f"p01.std.{group}.raw.min", min(raws))
        n.num(f"p01.std.{group}.raw.max", max(raws))

    # Flat against nested CV, standardized, over the 14 (source, text) pairs.
    cells = [(s, t) for s in sources for t in ("dep", "indep")]
    n.count("p01.cv.pairs", len(cells))
    n.count("p01.cv.flat_better", sum(1 for c in cells if grid[c + ("flat", "std")]["mean_eer"]
                                      < grid[c + ("nested", "std")]["mean_eer"]))
    n.num("p01.cv.mean.flat", mean_eer(c + ("flat", "std") for c in cells))
    n.num("p01.cv.mean.nested", mean_eer(c + ("nested", "std") for c in cells))

    # Seen against unseen phrases, flat CV, standardized, per source.
    n.count("p01.text.sources", len(sources))
    seen_better = [s for s in sources if grid[(s, "dep", "flat", "std")]["mean_eer"]
                   < grid[(s, "indep", "flat", "std")]["mean_eer"]]
    n.count("p01.text.seen_better", len(seen_better))
    n.num("p01.text.mean.seen", mean_eer((s, "dep", "flat", "std") for s in sources))
    n.num("p01.text.mean.unseen", mean_eer((s, "indep", "flat", "std") for s in sources))
    deltas = [grid[(s, "indep", "flat", "std")]["mean_eer"] - grid[(s, "dep", "flat", "std")]["mean_eer"]
              for s in sources]
    n.num("p01.text.delta.min", min(deltas))
    n.num("p01.text.delta.max", max(deltas))

    # Controls: the DSNN against its twin and against no network, flat, standardized.
    n.count("p01.ctrl.cells", len(cells))
    n.count("p01.ctrl.dsnn_better_ann", sum(1 for c in cells if grid[c + ("flat", "std")]["mean_eer"]
                                            < sets["ann"][c]["mean_eer"]))
    n.count("p01.ctrl.dsnn_better_cosine", sum(1 for c in cells if grid[c + ("flat", "std")]["mean_eer"]
                                               < sets["cosine"][c]["mean_eer"]))
    diffs = [grid[c + ("flat", "std")]["mean_eer"] - sets["ann"][c]["mean_eer"] for c in cells]
    n.num("p01.ctrl.dsnn_minus_ann.min", min(diffs))
    n.num("p01.ctrl.dsnn_minus_ann.max", max(diffs))
    n.num("p01.ctrl.dsnn_minus_ann.mean", statistics.mean(diffs))
    n.num("p01.ctrl.mean.dsnn", mean_eer(c + ("flat", "std") for c in cells))
    n.num("p01.ctrl.mean.ann", statistics.mean(sets["ann"][c]["mean_eer"] for c in cells))
    n.num("p01.ctrl.mean.cosine", statistics.mean(sets["cosine"][c]["mean_eer"] for c in cells))

    # Energy: nJ per test sample of the classifier, and the DSNN/ANN ratio under every check.
    ratios: dict[str, list[float]] = {k: [] for k, _ in P01.ENERGY_CHECKS}
    for s in sources:
        d = grid[(s, "dep", "flat", "std")]["energy"]
        a = sets["ann"][(s, "dep")]["energy"]
        n.num(f"p01.energy.{s}.dsnn.nj", d["nj"], 0)
        n.num(f"p01.energy.{s}.ann.nj", a["nj"], 0)
        n.num(f"p01.energy.{s}.spikes", d["spikes"], 0)
        for k, _ in P01.ENERGY_CHECKS:
            ratios[k].append(d[k] / a[k])
            n.num(f"p01.energy.{s}.ratio.{k}", d[k] / a[k], 2)
    for k, vals in ratios.items():
        n.num(f"p01.energy.ratio.{k}.min", min(vals), 2)
        n.num(f"p01.energy.ratio.{k}.max", max(vals), 2)
    every = [r for vals in ratios.values() for r in vals]
    n.num("p01.energy.ratio.all.min", min(every), 2)
    n.num("p01.energy.ratio.all.max", max(every), 2)
    n.count("p01.energy.dsnn_cheaper", sum(1 for r in every if r < 1))

    # Dysphonia: the rise of the EER from the clean cell to the full whisper.
    for (src, scope) in sorted({(k[0], k[1]) for k in sets["dysph"]}):
        clean = grid[(src, "dep", "flat", "std")]["mean_eer"]
        full = sets["dysph"][(src, scope, 100)]["mean_eer"]
        n.num(f"p01.dysph.{src}.{scope}.rise", full - clean)

    # Extractor families through the same classifier (Question 3).
    cmp_cells = sorted({(k[0], k[1]) for k in sets["cmp"]})
    n.count("p01.cmp.cells", len(cmp_cells))
    for fam in ("ae", "pca"):
        n.count(f"p01.cmp.hc_better_{fam}",
                sum(1 for c in cmp_cells
                    if sets["cmp"][c + ("hc",)]["mean_eer"] < sets["cmp"][c + (fam,)]["mean_eer"]))
        diffs = [sets["cmp"][c + (fam,)]["mean_eer"] - sets["cmp"][c + ("hc",)]["mean_eer"]
                 for c in cmp_cells]
        n.num(f"p01.cmp.{fam}_minus_hc.min", min(diffs))
        n.num(f"p01.cmp.{fam}_minus_hc.max", max(diffs))
    for fam in FAMILIES:
        vals = [sets["cmp"][c + (fam,)]["mean_eer"] for c in cmp_cells]
        n.num(f"p01.cmp.{fam}.min", min(vals))
        n.num(f"p01.cmp.{fam}.max", max(vals))
    n.count("p01.cmp.pca_better_ae",
            sum(1 for c in cmp_cells
                if sets["cmp"][c + ("pca",)]["mean_eer"] < sets["cmp"][c + ("ae",)]["mean_eer"]))


def significance(n: Numbers, results: pathlib.Path, sig: str = "sig", group_key: str = "siggroup",
                 tag: str = "p01", splits: tuple = (("ctrl", ("ann", "cosine")),
                                                     ("cmp", ("cmp-ae", "cmp-pca"))),
                 script: str = "03_thesis_phase01_significance.sh") -> None:
    """Keys of every significance_*.csv in `results`: <sig>.<file stem>.<condition>.<col> and
    <group_key>.<family>.*. Coretto: sig / siggroup / p01; BIOMEX-DB: bxsig / bxsiggroup / bx01."""
    files = sorted(results.glob("significance_*.csv"))
    if not files:
        fail(f"no significance_*.csv in {results}; run {script} first")
    groups: dict[str, list[dict]] = {}
    for path in files:
        stem = path.stem[len("significance_"):]
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            fail(f"{path} has no comparison rows")
        # The family is the stem up to its first "_" (03_thesis_phase01_significance.sh).
        groups.setdefault(stem.split("_")[0], []).extend(rows)
        for r in rows:
            cond = r["condition"].replace(f"e05_{tag}_", "")
            prefix = f"{sig}.{stem}.{cond}"
            n.num(f"{prefix}.delta", float(r["delta_mean"]))
            n.num(f"{prefix}.d", float(r["cohens_d"]), 2)
            n.num(f"{prefix}.p", float(r["p_value"]), 3)
            n.num(f"{prefix}.pholm", float(r["p_holm"]), 3)
        n.count(f"{sig}.{stem}.n", len(rows))
        n.count(f"{sig}.{stem}.significant", sum(1 for r in rows if float(r["p_holm"]) < 0.05))
        n.num(f"{sig}.{stem}.pholm.min", min(float(r["p_holm"]) for r in rows), 3)
        n.num(f"{sig}.{stem}.p.min", min(float(r["p_value"]) for r in rows), 3)
        n.num(f"{sig}.{stem}.d.min", min(float(r["cohens_d"]) for r in rows), 2)
        n.num(f"{sig}.{stem}.d.max", max(float(r["cohens_d"]) for r in rows), 2)

    def summary(prefix: str, rows: list[dict]) -> None:
        n.count(f"{prefix}.n", len(rows))
        n.count(f"{prefix}.significant", sum(1 for r in rows if float(r["p_holm"]) < 0.05))
        n.count(f"{prefix}.nominal", sum(1 for r in rows if float(r["p_value"]) < 0.05))
        n.num(f"{prefix}.pholm.min", min(float(r["p_holm"]) for r in rows), 3)
        n.num(f"{prefix}.p.min", min(float(r["p_value"]) for r in rows), 3)
        n.num(f"{prefix}.p.max", max(float(r["p_value"]) for r in rows), 3)

    for group, rows in groups.items():
        summary(f"{group_key}.{group}", rows)
    # Within the controls and the extractor comparison, the two kinds of contrast apart.
    for group, kinds in splits:
        if group not in groups:
            fail(f"no significance family {group!r}")
        for kind in kinds:
            rows = [r for r in groups[group] if f"_{tag}_{kind}_" in r["condition"]]
            if not rows:
                fail(f"no {kind} rows in the {group} significance family")
            summary(f"{group_key}.{group}.{kind.replace('cmp-', '')}", rows)
            # Which way the significant ones go: a positive delta is the condition's EER
            # above the reference's (the reference better).
            n.count(f"{group_key}.{group}.{kind.replace('cmp-', '')}.significant_ref_better",
                    sum(1 for r in rows if float(r["p_holm"]) < 0.05
                        and float(r["delta_mean"]) > 0))


def biomex(n: Numbers, profiles: pathlib.Path, results: pathlib.Path,
           co_profiles: pathlib.Path, co_results: pathlib.Path) -> None:
    """The BIOMEX-DB replication: bx.<kind>.<source>.<text>.<m>, its comparison with the same
    Coretto cell (bx.vs.*), and its significance families (bxsig.*, bxsiggroup.*)."""
    sets = BX.collect(profiles, results)
    co = P01.collect(co_profiles, co_results)

    def metrics(prefix: str, run: dict) -> None:
        n.num(f"{prefix}.eer", run["mean_eer"])
        n.num(f"{prefix}.auc", run["mean_auc"])
        n.num(f"{prefix}.eersd", run["std_eer"])

    for kind in ("grid", "ann", "cosine"):
        for (src, text), run in sets[kind].items():
            metrics(f"bx.{'dsnn' if kind == 'grid' else kind}.{src}.{text}", run)
    for (src, scope, level), run in sets["dysph"].items():
        metrics(f"bx.dysph.{src}.{scope}.{level}", run)
    for kind, runs in sets.items():
        n.count(f"bx.n.{kind}", len(runs))
    n.count("bx.n.total", sum(len(r) for r in sets.values()))
    n.count("bx.n.units", sum(r["n_reps"] for runs in sets.values() for r in runs.values()))

    grid = sets["grid"]
    cells = [(s, t) for s in BX.SOURCES for t in ("dep", "indep")]
    n.count("bx.cells", len(cells))
    order = sorted(cells, key=lambda c: grid[c]["mean_eer"])
    n.text("bx.best.label", f"{P01.SOURCE_LABELS[order[0][0]]}, {P01.TEXT_LABELS[order[0][1]]}")
    metrics("bx.best", grid[order[0]])
    n.text("bx.worst.label", f"{P01.SOURCE_LABELS[order[-1][0]]}, {P01.TEXT_LABELS[order[-1][1]]}")
    metrics("bx.worst", grid[order[-1]])
    co_order = sorted(cells, key=lambda c: co["grid"][c + ("flat", "std")]["mean_eer"])
    n.count("bx.same_best_as_coretto", int(order[0] == co_order[0]))
    n.count("bx.same_worst_as_coretto", int(order[-1] == co_order[-1]))
    diffs = [grid[c]["mean_eer"] - co["grid"][c + ("flat", "std")]["mean_eer"] for c in cells]
    n.num("bx.vs.coretto.diff.min", min(diffs))
    n.num("bx.vs.coretto.diff.max", max(diffs))
    n.num("bx.vs.coretto.absdiff.mean", statistics.mean(abs(d) for d in diffs))
    n.count("bx.vs.coretto.lower", sum(1 for d in diffs if d < 0))
    n.num("bx.mean.dsnn", statistics.mean(grid[c]["mean_eer"] for c in cells))
    n.num("bx.coretto.mean.dsnn", statistics.mean(co["grid"][c + ("flat", "std")]["mean_eer"]
                                                  for c in cells))
    n.num("bx.mean.ann", statistics.mean(sets["ann"][c]["mean_eer"] for c in cells))
    n.num("bx.mean.cosine", statistics.mean(sets["cosine"][c]["mean_eer"] for c in cells))
    n.count("bx.ctrl.dsnn_better_cosine",
            sum(1 for c in cells if grid[c]["mean_eer"] < sets["cosine"][c]["mean_eer"]))
    voice = [c for c in order if c[0] == "voice"]
    n.count("bx.voice.first_rank", order.index(voice[0]) + 1)
    ratios = [grid[c]["nj"] / sets["ann"][c]["nj"] for c in cells]
    n.num("bx.energy.ratio.min", min(ratios), 2)
    n.num("bx.energy.ratio.max", max(ratios), 2)
    n.count("bx.energy.dsnn_cheaper", sum(1 for r in ratios if r < 1))
    for (src, scope) in sorted({(k[0], k[1]) for k in sets["dysph"]}):
        n.num(f"bx.dysph.{src}.{scope}.rise",
              sets["dysph"][(src, scope, 100)]["mean_eer"] - grid[(src, "dep")]["mean_eer"])
    significance(n, results, sig="bxsig", group_key="bxsiggroup", tag="bx01",
                 splits=(("ctrl", ("ann", "cosine")),),
                 script="04_thesis_biomex_significance.sh")


def ae_input(n: Numbers, path: pathlib.Path) -> None:
    if not path.is_file():
        fail(f"{path} is missing; run scripts/analysis/thesis_ae_input_reference.py --out {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["signal"] == "chance":
            n.num("aeinput.chance", float(r["accuracy"]), 3)
        elif r["signal"] == "trials":
            n.count("aeinput.trials", int(r["accuracy"]))
        else:
            n.num(f"aeinput.{r['signal']}.{r['input']}.{r['classifier']}", float(r["accuracy"]), 2)


def base_facts(n: Numbers, path: pathlib.Path) -> None:
    if not path.is_file():
        fail(f"{path} is missing; run scripts/analysis/thesis_base_facts.py --out {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "count":
                n.count(r["key"], int(r["value"]))
            elif r["kind"] == "num1":
                n.num(r["key"], float(r["value"]), 1)
            elif r["kind"] == "text":
                n.text(r["key"], r["value"])
            else:
                fail(f"{path}: unknown kind {r['kind']!r} for {r['key']}")


def dysphonia_calibration(n: Numbers, path: pathlib.Path) -> None:
    """The WORLD profiles' calibration (05_thesis_dysphonia_calibration.py)."""
    import json
    if not path.is_file():
        fail(f"{path} is missing; run scripts/pipeline/thesis/05_thesis_dysphonia_calibration.py")
    r = json.loads(path.read_text(encoding="utf-8"))
    n.text("cal.praat", r["praat"])
    n.text("cal.parselmouth", r["parselmouth"])
    n.count("cal.excluded", len(r["excluded"]))
    n.count("cal.tolerance.pct", round(100 * r["tolerance"]))
    total = 0
    for subj, d in r["per_subject"].items():
        k = f"cal.{subj.lower()}"
        total += d["voices"]
        n.count(f"{k}.voices", d["voices"])
        n.num(f"{k}.f0", d["f0_hz_mean"], 0)
        for arm in ("raw", "control"):
            n.num(f"{k}.{arm}.jitter", d[arm]["jitter_pct"], 2)
            n.num(f"{k}.{arm}.shimmer", d[arm]["shimmer_db"], 2)
            n.num(f"{k}.{arm}.hnr", d[arm]["hnr_db"], 1)
        n.num(f"{k}.control.hnrchange", d["control"]["hnr_change_db"], 1)
        n.num(f"{k}.control.rmschange", d["control"]["rms_change_db"], 2)
        n.count(f"{k}.control.rmsok", d["control_within_1dB_rms"])
        n.count(f"{k}.control.hnrok", d["control_within_1_5dB_hnr"])
    n.count("cal.voices", total)
    curves = r["curves"]

    def at(axis, req):
        c = curves[axis]
        return c["mean_measured"][c["request"].index(req)]

    # Read-back ratio: measured / requested in the middle of each sweep.
    n.num("cal.jitter.ratio", at("jitter_pct", 6) / 6, 2)
    n.num("cal.tremor.ratio", at("f0_depth_pct", 6) / 6, 2)
    n.num("cal.shimmer.ceiling", max(curves["shimmer_db"]["mean_measured"]), 2)
    n.num("cal.shimmer.maxreq", max(curves["shimmer_db"]["request"]), 0)
    n.num("cal.diplo.ratio", at("period_alt_pct", 6) / 6, 2)
    # Per speaker, from the rows (calibration.json keeps the mean over both speakers).
    rows_path = path.with_name("measurements.csv")
    if not rows_path.is_file():
        fail(f"{rows_path} is missing; it is written with {path}")
    with rows_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def speaker_ratio(axis, column, subject, request):
        vals = [float(row[column]) for row in rows
                if row["set"] == f"sweep:{axis}" and row["subject"] == subject
                and float(row[f"req_{axis}"]) == request]
        if not vals:
            fail(f"{rows_path}: no {subject} row of the {axis} sweep at request {request}")
        return statistics.mean(vals) / request

    n.num("cal.diplo.s10.ratio", speaker_ratio("period_alt_pct", "alt_period_pct", "S10", 6.0), 2)
    for subj in ("S10", "S03"):
        n.num(f"cal.jitter.{subj.lower()}.ratio",
              speaker_ratio("jitter_pct", "jitter_pct", subj, 6.0), 2)
    for i, lv in enumerate(r["levels"]):
        for axis, target in lv["targets"].items():
            key = f"cal.lv.{lv['profile']}.{axis.replace('_', '')}.{str(target).replace('.', 'p')}"
            if key + ".req" in n.values:
                continue
            n.num(key + ".req", lv["request"][axis], 2)
            within = sum(v["within_tolerance"] for v in lv[axis].values())
            count = sum(v["n"] for v in lv[axis].values())
            n.count(key + ".within", within)
            n.count(key + ".n", count)
    n.count("cal.unreachable", len(r["unreachable_levels"]))
    rates = [b["measured_mean_per_s"] / b["requested_per_s"] for b in r["breaks"]]
    n.num("cal.breaks.ratio.min", min(rates), 2)
    n.num("cal.breaks.ratio.max", max(rates), 2)
    n.num("cal.minutes", r["seconds"] / 60, 0)


def identity_pilot(n: Numbers, path: pathlib.Path) -> None:
    """The identity pilot's verdict (06_thesis_identity_pilot.py)."""
    import json
    if not path.is_file():
        fail(f"{path} is missing; run the pilot profiles, then "
             f"scripts/pipeline/thesis/06_thesis_identity_pilot.py")
    r = json.loads(path.read_text(encoding="utf-8"))
    verdicts = {"transparent": "transparente", "not transparent": "não transparente",
                "inconclusive": "inconclusivo"}
    n.num("pilot.margin", r["margin"], 2)
    n.put("pilot.overall", verdicts[r["overall"]], r["overall"])
    for kind, scopes in r["arms"].items():
        for scope, d in scopes.items():
            k = f"pilot.{kind}.{scope}"
            n.num(f"{k}.raw", d["mean_raw_eer"])
            n.num(f"{k}.control", d["mean_control_eer"])
            n.num(f"{k}.delta", d["mean_delta"])
            n.num(f"{k}.lo", d["ci95"][0])
            n.num(f"{k}.hi", d["ci95"][1])
            n.put(f"{k}.verdict", verdicts[d["verdict"]], d["verdict"])
            n.num(f"{k}.half", (d["ci95"][1] - d["ci95"][0]) / 2)
            n.count(f"{k}.reppos", sum(m > 0 for m in d["per_repeat_mean_delta"]))
            n.count(f"{k}.reps", len(d["per_repeat_mean_delta"]))


def identity_margin(n: Numbers, path: pathlib.Path) -> None:
    """The powered identity test's verdict (07_thesis_identity_margin.py)."""
    import json
    if not path.is_file():
        fail(f"{path} is missing; run tools/thesis_features on the two profiles, then "
             f"scripts/pipeline/thesis/07_thesis_identity_margin.py")
    r = json.loads(path.read_text(encoding="utf-8"))
    verdicts = {"transparent": "transparente", "not transparent": "não transparente",
                "inconclusive": "inconclusivo"}
    n.count("margin.utterances", r["utterances"])
    n.count("margin.speakers", r["speakers"])
    n.num("margin.clean.mean", r["clean_margin_mean"], 3)
    n.num("margin.clean.positive", 100 * r["clean_margin_positive_share"], 1)
    n.num("margin.whisper.mean", r["whisper_l25"]["mean"])
    n.num("margin.whisper.lo", r["whisper_l25"]["ci95"][0])
    n.num("margin.whisper.hi", r["whisper_l25"]["ci95"][1])
    n.num("margin.delta", r["delta"])
    n.num("margin.control.mean", r["control"]["mean"])
    n.num("margin.control.lo", r["control"]["ci95"][0])
    n.num("margin.control.hi", r["control"]["ci95"][1])
    n.count("margin.control.below", r["control"]["speakers_below_zero"])
    n.num("margin.control.times", r["control"]["ratio_to_whisper"], 1)
    n.put("margin.verdict", verdicts[r["verdict"]], r["verdict"])
    x = r["exploratory"]   # after the verdict, not part of the rule
    for name, d in x["mean_signed_shift_in_clean_sd"].items():
        n.num(f"margin.explore.shift.{name}.control", d["control"], 2)
        n.num(f"margin.explore.shift.{name}.whisper", d["whisper_l25"], 2)
    for arm, key in (("control", "control"), ("whisper_l25", "whisper")):
        d = x["without_jitter_shimmer"][arm]
        n.num(f"margin.explore.nojs.{key}.mean", d["mean"])
        n.num(f"margin.explore.nojs.{key}.lo", d["ci95"][0])
        n.num(f"margin.explore.nojs.{key}.hi", d["ci95"][1])


# What the prose of chapters 4 (abstract), 8, 9 and 10 states in words. Each entry: the
# sentence (as the thesis says it), and a condition on the keys; v(key) is the value as a
# number, t(key) as text. Keep the two in step: change a sentence, change its claim.
S3 = ("voice", "eegpron", "eegimag")
CLAIMS = [
    ("ch9: the same extractor wins on the three signals",
     lambda v, t: len({t(f"p00.{s}.winner.label") for s in S3}) == 1),
    ("ch9: the winner is handcrafted (it carries the dagger in the hc tables only)",
     lambda v, t: all(t(f"p00.{s}.winner.label") == t(f"p00.{s}.best.hc.label") for s in S3)),
    ("ch9: for voice the smallest unpenalized d is not the winner",
     lambda v, t: t("p00.voice.mind.label") != t("p00.voice.winner.label")),
    ("ch9: on EEG the winner is rank 1 in every fold and leave-one-speaker-out subset",
     lambda v, t: all(v(f"p00.{s}.stab.{k}.losses") == 0
                      for s in ("eegpron", "eegimag") for k in ("fold", "loso"))),
    ("ch9: for voice the margin to the runner-up is below the winner's sigma_loc, and the "
     "winner loses some folds",
     lambda v, t: v("p00.voice.margin") < v("p00.voice.winner.loso")
     and v("p00.voice.stab.fold.losses") > 0),
    ("ch9: every extractor that wins a voice fold instead of the winner is Category 1",
     lambda v, t: v("p00.voice.stab.fold.alt.c1") == v("p00.voice.stab.fold.alt")),
    ("ch9: the best PCA is k = 8 and beats the best autoencoder on the three signals",
     lambda v, t: all(t(f"p00.{s}.best.pca.label") == "PCA (k = 8)"
                      and v(f"p00.{s}.best.pca.dpen") < v(f"p00.{s}.best.ae.dpen") for s in S3)),
    ("ch9: the handcrafted winner is ahead of the best automated extractor on the three signals",
     lambda v, t: all(v(f"p00.{s}.winner.dpen") < v(f"p00.{s}.best.pca.dpen") for s in S3)),
    ("ch9: the three ANN-AE are the first three autoencoders on the three signals",
     lambda v, t: all(t(f"p00.{s}.ae.{r}.label").startswith("ANN-AE")
                      for s in S3 for r in (1, 2, 3))),
    ("ch8/ch9: SNN-AE encodings rank latency < poisson < direct (mean d_pen) on the three signals",
     lambda v, t: all(v(f"p00.{s}.enc.latency.mean") < v(f"p00.{s}.enc.poisson.mean")
                      < v(f"p00.{s}.enc.direct.mean") for s in S3)),
    ("ch9 Q1: the scales rank linear < Mel < Bark for voice",
     lambda v, t: v("p00.voice.scale.linear.best") < v("p00.voice.scale.mel.best")
     < v("p00.voice.scale.bark.best")),
    ("ch9 Q2: for voice no cepstral extractor beats the worst non-cepstral one",
     lambda v, t: v("p00.voice.c2.better_than_worst_c1") == 0),
    ("ch9 Q2: on EEG the best is non-cepstral but the cepstral mean is (slightly) lower",
     lambda v, t: all(v(f"p00.{s}.c1.best") < v(f"p00.{s}.c2.best")
                      and v(f"p00.{s}.c2.mean") < v(f"p00.{s}.c1.mean")
                      for s in ("eegpron", "eegimag"))),
    ("ch9: every standardized EEG/fusion cell of the grid ranks before the first voice cell",
     lambda v, t: v("p01.voice.first_rank") == 25),
    ("ch9: no comparison with voice is significant",
     lambda v, t: v("siggroup.vsvoice.significant") == 0),
    ("ch9: no fusion differs significantly from its EEG; with unseen phrases an EEG is ahead "
     "of its late fusion",
     lambda v, t: v("siggroup.fusion.significant") == 0
     and v("p01.grid.eegpron.indep.flat.std.eer") < v("p01.grid.latepron.indep.flat.std.eer")),
    ("ch9: standardization lowers the EER in every cell",
     lambda v, t: v("p01.std.better") == v("p01.std.pairs")),
    ("ch9: flat CV gives the lower EER in most cells",
     lambda v, t: v("p01.cv.flat_better") >= 0.75 * v("p01.cv.pairs")),
    ("ch9: the EEG condition makes no significant difference",
     lambda v, t: v("siggroup.cond.significant") == 0),
    ("ch9: the DSNN differs significantly from neither control",
     lambda v, t: v("siggroup.ctrl.significant") == 0),
    ("ch9: mean EER orders DSNN < ANN < cosine",
     lambda v, t: v("p01.ctrl.mean.dsnn") < v("p01.ctrl.mean.ann") < v("p01.ctrl.mean.cosine")),
    ("ch9: the DSNN is never the cheaper network, under any check",
     lambda v, t: v("p01.energy.dsnn_cheaper") == 0),
    ("ch9: under full whisper voice rises more than any fusion, whose largest rise is "
     "late fusion (imagined), enrolment and probes",
     lambda v, t: all(v("p01.dysph.lateimag.test.rise") >= v(f"p01.dysph.{s}.{sc}.rise")
                      for s in ("earlypron", "earlyimag", "latepron", "lateimag")
                      for sc in ("probes", "test"))
     and v("p01.dysph.voice.probes.rise") > v("p01.dysph.lateimag.test.rise")),
    ("ch9: no degradation effect is significant",
     lambda v, t: v("siggroup.dysph.significant") == 0),
    ("ch9/ch10/abstract: through the classifier no automated extractor differs "
     "significantly from the handcrafted one",
     lambda v, t: v("siggroup.cmp.significant") == 0),
    ("ch9: voice, both automated extractors within 0.03 EER of the handcrafted one, one on "
     "each side (seen phrases)",
     lambda v, t: abs(v("p01.cmp.voice.dep.ae.eer") - v("p01.cmp.voice.dep.hc.eer")) < 0.03
     and abs(v("p01.cmp.voice.dep.pca.eer") - v("p01.cmp.voice.dep.hc.eer")) < 0.03
     and (v("p01.cmp.voice.dep.ae.eer") - v("p01.cmp.voice.dep.hc.eer"))
     * (v("p01.cmp.voice.dep.pca.eer") - v("p01.cmp.voice.dep.hc.eer")) < 0),
    ("ch9: pronounced EEG, handcrafted ahead of both; imagined EEG, behind both (both text modes)",
     lambda v, t: all(v(f"p01.cmp.eegpron.{x}.hc.eer") < v(f"p01.cmp.eegpron.{x}.{f}.eer")
                      and v(f"p01.cmp.eegimag.{x}.hc.eer") > v(f"p01.cmp.eegimag.{x}.{f}.eer")
                      for x in ("dep", "indep") for f in ("ae", "pca"))),
    ("ch9/ch10: PCA on imagined EEG, seen phrases, is the lowest EER of the whole experiment",
     lambda v, t: v("p01.cmp.eegimag.dep.pca.eer") == v("p01.min.eer")),
    ("ch9 Q3: the autoencoder Phase 00 picked is the ANN-AE on the three signals",
     lambda v, t: all(t(f"p00.{s}.best.ae.label").startswith("ANN-AE") for s in S3)),
    ("abstract: the best cell of the main grid is late fusion with imagined-speech EEG",
     lambda v, t: t("p01.best.label").startswith("Fusão tardia (imaginada)")),
    ("ch8: every Coretto trial equals the Zenodo copy",
     lambda v, t: v("base.coretto.mirror.equal") == v("base.coretto.trials")),
    ("ch8: the base holds one S01 pronounced /o/ trial more than the paper's Table 4",
     lambda v, t: v("base.coretto.s01o.base") == v("base.coretto.s01o.paper") + 1),
    ("ch8: with the same digital filter, BIOMEX-DB holds less power below 2 Hz than Coretto",
     lambda v, t: v("base.biomex.below2hz.pct") < v("base.coretto.below2hz.pct")),
    ("ch8: the pairs left out are exactly the edge ones plus the unlabelled ones",
     lambda v, t: v("base.biomex.pairs.dropped")
     == v("base.biomex.pairs.edge") + v("base.biomex.pairs.unlabelled")),
    ("ch8: BIOMEX-DB is upsampled to the Coretto rates (written at 44.1 kHz / 1024 Hz)",
     lambda v, t: v("base.biomex.voice.sample.rate") == 44100
     and v("base.biomex.eeg.sample.rate") == 1024),
    ("ch9 bx: the best and the worst cell are the same on both bases",
     lambda v, t: v("bx.same_best_as_coretto") == 1 and v("bx.same_worst_as_coretto") == 1),
    ("ch9 bx: every EEG/fusion cell ranks before both voice cells",
     lambda v, t: v("bx.voice.first_rank") == v("bx.cells") - 1),
    ("ch9 bx: late fusion beats voice significantly in both text modes; EEG alone does not",
     lambda v, t: all(v(f"bxsig.vsvoice_{x}.dsnn_latepron_{x}_flat_std.pholm") < 0.05
                      and v(f"bxsig.vsvoice_{x}.dsnn_latepron_{x}_flat_std.delta") < 0
                      and v(f"bxsig.vsvoice_{x}.dsnn_eegpron_{x}_flat_std.pholm") >= 0.05
                      for x in ("dep", "indep"))),
    ("ch9 bx: on Coretto the same late-fusion-vs-voice difference was of the same order "
     "(same sign, at least half the BIOMEX size), so the change is one of power",
     lambda v, t: all(v(f"sig.vsvoice_{x}.dsnn_latepron_{x}_flat_std.delta") < 0
                      and abs(v(f"sig.vsvoice_{x}.dsnn_latepron_{x}_flat_std.delta"))
                      >= 0.5 * abs(v(f"bxsig.vsvoice_{x}.dsnn_latepron_{x}_flat_std.delta"))
                      for x in ("dep", "indep"))),
    ("ch9 bx: no fusion differs significantly from its EEG",
     lambda v, t: v("bxsiggroup.fusion.significant") == 0),
    ("ch9 bx: the DSNN differs from its ANN twin in no cell, not even nominally",
     lambda v, t: v("bxsiggroup.ctrl.ann.significant") == 0
     and v("bxsiggroup.ctrl.ann.nominal") == 0),
    ("ch9 bx: cosine is behind the DSNN in every cell, significantly in some, all in the "
     "DSNN's favour",
     lambda v, t: v("bx.ctrl.dsnn_better_cosine") == v("bx.cells")
     and v("bxsiggroup.ctrl.cosine.significant") > 0
     and v("bxsiggroup.ctrl.cosine.significant")
     == v("bxsiggroup.ctrl.cosine.significant_ref_better")),
    ("ch9 bx: the DSNN is never the cheaper network",
     lambda v, t: v("bx.energy.dsnn_cheaper") == 0),
    ("ch9 bx: under full whisper voice rises more than any fusion; no degradation effect is "
     "significant",
     lambda v, t: min(v("bx.dysph.voice.probes.rise"), v("bx.dysph.voice.test.rise"))
     > max(v(f"bx.dysph.{s}.{sc}.rise") for s in ("earlypron", "latepron")
           for sc in ("probes", "test"))
     and v("bxsiggroup.dysph.significant") == 0),
    ("ch9 bx: unseen phrases give the higher EER in every source, none significantly",
     lambda v, t: all(v(f"bxsig.text_{s}.dsnn_{s}_indep_flat_std.delta") > 0
                      for s in ("voice", "eegpron", "earlypron", "latepron"))
     and v("bxsiggroup.text.significant") == 0),
    ("ch9/ch10/abstract bx: of the Coretto conclusions BIOMEX-DB can test, five repeat (fusion "
     "vs EEG, DSNN vs ANN, energy, dysphonia, text mode) and exactly two change (late fusion vs "
     "voice, network vs cosine), each from not significant to significant",
     lambda v, t: [v("bxsiggroup.fusion.significant") == 0 and v("siggroup.fusion.significant") == 0,
                   v("bxsiggroup.ctrl.ann.significant") == 0 and v("siggroup.ctrl.ann.significant") == 0,
                   v("bx.energy.dsnn_cheaper") == 0 and v("p01.energy.dsnn_cheaper") == 0,
                   v("bxsiggroup.dysph.significant") == 0 and v("siggroup.dysph.significant") == 0,
                   v("bxsiggroup.text.significant") == 0 and v("siggroup.text.significant") == 0,
                   ].count(True) == 5
     and v("siggroup.vsvoice.significant") == 0 and v("bxsiggroup.vsvoice.significant") > 0
     and v("siggroup.ctrl.cosine.significant") == 0
     and v("bxsiggroup.ctrl.cosine.significant") > 0),
    ("ch8 cal: tremor does what is asked",
     lambda v, t: v("cal.tremor.ratio") >= 0.95),
    ("ch8 cal: jitter reads below the request, the same on both speakers",
     lambda v, t: v("cal.jitter.ratio") < 0.9
     and abs(v("cal.jitter.s10.ratio") - v("cal.jitter.s03.ratio")) <= 0.05),
    ("ch8 cal: shimmer saturates near 2 dB although far more is requested",
     lambda v, t: v("cal.shimmer.ceiling") < 2.1
     and v("cal.shimmer.ceiling") < v("cal.shimmer.maxreq") / 2),
    ("ch8 cal: the 1.5 dB shimmer of the strong rough levels is still followed by the "
     "measurement (at least 90 % of the trials within tolerance), and every level is reachable",
     lambda v, t: v("cal.lv.rough.shimmerdb.1p5.within") >= 0.9 * v("cal.lv.rough.shimmerdb.1p5.n")
     and v("cal.unreachable") == 0),
    ("ch8 cal: diplophonia is not validated (the low voice reads well under the request)",
     lambda v, t: v("cal.diplo.s10.ratio") < 0.6),
    ("ch8 cal: the control keeps the energy of every trial (within 1 dB)",
     lambda v, t: all(v(f"cal.{s}.control.rmsok") == v(f"cal.{s}.voices") for s in ("s10", "s03"))),
    ("ch8 cal: the control makes both voices more periodic, few trials within 1.5 dB of HNR",
     lambda v, t: all(v(f"cal.{s}.control.hnrchange") > 1.5
                      and v(f"cal.{s}.control.hnrok") < v(f"cal.{s}.voices") / 2
                      for s in ("s10", "s03"))),
    ("ch8 pilot: inconclusive on the deciding classifier (cosine), both scopes",
     lambda v, t: t("pilot.overall") == "inconclusive"
     and all(t(f"pilot.cosine.{s}.verdict") == "inconclusive" for s in ("probes", "test"))),
    ("ch8 pilot: the cosine interval is at least three times wider than the margin",
     lambda v, t: all(v(f"pilot.cosine.{s}.half") >= 3 * v("pilot.margin")
                      for s in ("probes", "test"))),
    ("ch8 pilot: with the DSNN resynthesis raised the mean EER in both scopes and in every "
     "repeat, by more than the full whisper did in the same Phase 01 cell (probes), and still "
     "not significantly (the interval contains 0)",
     lambda v, t: all(v(f"pilot.dsnn.{s}.delta") > 0 and v(f"pilot.dsnn.{s}.lo") < 0
                      and v(f"pilot.dsnn.{s}.reppos") == v(f"pilot.dsnn.{s}.reps")
                      for s in ("probes", "test"))
     and v("pilot.dsnn.probes.delta") > v("p01.dysph.voice.probes.100.eer")
     - v("p01.grid.voice.dep.flat.std.eer")),
    ("ch8 margin: the yardstick holds (whisper l25 lowers the margin, interval below 0)",
     lambda v, t: v("margin.whisper.hi") < 0),
    ("ch8 margin: the control is not transparent; it lowers the margin by more than whisper l25, "
     "its whole interval below -delta, in all but at most one speaker",
     lambda v, t: t("margin.verdict") == "not transparent"
     and v("margin.control.hi") < -v("margin.delta")
     and v("margin.control.mean") < v("margin.whisper.mean")
     and v("margin.control.times") > 1
     and v("margin.control.below") >= v("margin.speakers") - 1),
    ("ch8 margin (exploratory): shimmer is the descriptor the control moves most, upward, while "
     "whisper l25 moves it downward",
     lambda v, t: max(("energy", "zcr", "entropy", "teager", "jitter", "shimmer"),
                      key=lambda d: abs(v(f"margin.explore.shift.{d}.control"))) == "shimmer"
     and v("margin.explore.shift.shimmer.control") > 0
     and v("margin.explore.shift.shimmer.whisper") < 0),
    ("ch8 margin (exploratory): without the jitter and shimmer descriptors neither shift is "
     "distinguishable from 0",
     lambda v, t: all(v(f"margin.explore.nojs.{a}.lo") < 0 < v(f"margin.explore.nojs.{a}.hi")
                      for a in ("control", "whisper"))),
    ("ch9 Q4: seen phrases give the lower EER in most sources, none significantly",
     lambda v, t: v("p01.text.seen_better") > v("p01.text.sources") / 2
     and v("siggroup.text.significant") == 0 and v("siggroup.text.nominal") == 0),
]


def check_claims(n: Numbers) -> None:
    def text(key: str) -> str:
        if key not in n.values:
            fail(f"a claim reads the key {key}, which does not exist")
        return n.values[key][1]

    def value(key: str) -> float:
        return float(text(key))

    broken = [sentence for sentence, holds in CLAIMS if not holds(value, text)]
    if broken:
        listed = "\n  - ".join(broken)
        fail(f"{len(broken)} statement(s) of the thesis prose no longer hold on these results:"
             f"\n  - {listed}\nRewrite those passages (and their entries in CLAIMS), then re-run.")
    print(f"[text-numbers] {len(CLAIMS)} prose statements checked against the results")


def main() -> int:
    ap = argparse.ArgumentParser(description="Write the numbers the thesis prose quotes.")
    for arg in ("--phase00-profiles", "--phase00-results", "--phase01-profiles",
                "--phase01-results", "--biomex-profiles", "--biomex-results",
                "--ae-input-reference", "--base-facts", "--dysphonia-calibration",
                "--identity-pilot", "--identity-margin", "--out"):
        ap.add_argument(arg, required=True)
    ap.add_argument("--list", action="store_true", help="also print every key and value")
    args = ap.parse_args()

    n = Numbers()
    phase00(n, pathlib.Path(args.phase00_profiles), pathlib.Path(args.phase00_results))
    phase01(n, pathlib.Path(args.phase01_profiles), pathlib.Path(args.phase01_results))
    significance(n, pathlib.Path(args.phase01_results))
    ae_input(n, pathlib.Path(args.ae_input_reference))
    base_facts(n, pathlib.Path(args.base_facts))
    biomex(n, pathlib.Path(args.biomex_profiles), pathlib.Path(args.biomex_results),
           pathlib.Path(args.phase01_profiles), pathlib.Path(args.phase01_results))
    dysphonia_calibration(n, pathlib.Path(args.dysphonia_calibration))
    identity_pilot(n, pathlib.Path(args.identity_pilot))
    identity_margin(n, pathlib.Path(args.identity_margin))
    check_claims(n)

    lines = ["% GENERATED by software/nn/scripts/pipeline/thesis/thesis_build_text_numbers.py.",
             "% Do not edit: re-run the script. Read with \\tn{key} (decimal comma) and",
             "% \\tnen{key} (decimal point); both are defined in chapters/data/visualStyle.tex",
             "% and refuse a key that is not here."]
    for key in sorted(n.values):
        pt, en = n.values[key]
        lines.append(f"\\expandafter\\def\\csname tn@{key}\\endcsname{{{pt}}}%")
        lines.append(f"\\expandafter\\def\\csname tnen@{key}\\endcsname{{{en}}}%")
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[text-numbers] wrote {len(n.values)} keys -> {args.out}")
    if args.list:
        for key in sorted(n.values):
            print(f"{key} = {n.values[key][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
