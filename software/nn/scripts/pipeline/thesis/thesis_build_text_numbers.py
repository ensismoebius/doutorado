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
    base.biomex.*, base.coretto.*           the two bases: volunteers kept and excluded, pairs,
                                            verification counts, power below 2 Hz
                                            (scripts/analysis/thesis_base_facts.py, read from the
                                            bases and the verifiers' stamps)

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


def significance(n: Numbers, results: pathlib.Path) -> None:
    files = sorted(results.glob("significance_*.csv"))
    if not files:
        fail(f"no significance_*.csv in {results}; run 03_thesis_phase01_significance.sh first")
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
            cond = r["condition"].replace("e05_p01_", "")
            prefix = f"sig.{stem}.{cond}"
            n.num(f"{prefix}.delta", float(r["delta_mean"]))
            n.num(f"{prefix}.d", float(r["cohens_d"]), 2)
            n.num(f"{prefix}.p", float(r["p_value"]), 3)
            n.num(f"{prefix}.pholm", float(r["p_holm"]), 3)
        n.count(f"sig.{stem}.n", len(rows))
        n.count(f"sig.{stem}.significant", sum(1 for r in rows if float(r["p_holm"]) < 0.05))
        n.num(f"sig.{stem}.pholm.min", min(float(r["p_holm"]) for r in rows), 3)
        n.num(f"sig.{stem}.p.min", min(float(r["p_value"]) for r in rows), 3)
        n.num(f"sig.{stem}.d.min", min(float(r["cohens_d"]) for r in rows), 2)
        n.num(f"sig.{stem}.d.max", max(float(r["cohens_d"]) for r in rows), 2)

    def summary(prefix: str, rows: list[dict]) -> None:
        n.count(f"{prefix}.n", len(rows))
        n.count(f"{prefix}.significant", sum(1 for r in rows if float(r["p_holm"]) < 0.05))
        n.count(f"{prefix}.nominal", sum(1 for r in rows if float(r["p_value"]) < 0.05))
        n.num(f"{prefix}.pholm.min", min(float(r["p_holm"]) for r in rows), 3)
        n.num(f"{prefix}.p.min", min(float(r["p_value"]) for r in rows), 3)
        n.num(f"{prefix}.p.max", max(float(r["p_value"]) for r in rows), 3)

    for group, rows in groups.items():
        summary(f"siggroup.{group}", rows)
    # Within the controls and the extractor comparison, the two kinds of contrast apart.
    for group, kinds in (("ctrl", ("ann", "cosine")), ("cmp", ("cmp-ae", "cmp-pca"))):
        if group not in groups:
            fail(f"no significance family {group!r}")
        for kind in kinds:
            rows = [r for r in groups[group] if f"_p01_{kind}_" in r["condition"]]
            if not rows:
                fail(f"no {kind} rows in the {group} significance family")
            summary(f"siggroup.{group}.{kind.replace('cmp-', '')}", rows)


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
                "--phase01-results", "--ae-input-reference", "--base-facts", "--out"):
        ap.add_argument(arg, required=True)
    ap.add_argument("--list", action="store_true", help="also print every key and value")
    args = ap.parse_args()

    n = Numbers()
    phase00(n, pathlib.Path(args.phase00_profiles), pathlib.Path(args.phase00_results))
    phase01(n, pathlib.Path(args.phase01_profiles), pathlib.Path(args.phase01_results))
    significance(n, pathlib.Path(args.phase01_results))
    ae_input(n, pathlib.Path(args.ae_input_reference))
    base_facts(n, pathlib.Path(args.base_facts))
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
