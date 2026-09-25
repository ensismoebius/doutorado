/* charts.js — Plotly wrapper helpers for the meeting01 dashboard */

const Charts = (() => {
    const LAYOUT_BASE = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#e1e4ed", size: 11 },
        margin: { l: 50, r: 20, t: 30, b: 40 },
        xaxis: { gridcolor: "#2e3142", zerolinecolor: "#2e3142" },
        yaxis: { gridcolor: "#2e3142", zerolinecolor: "#2e3142" },
        legend: { orientation: "h", y: -0.15 },
    };

    const CONFIG = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["lasso2d", "selectd"],
        displaylogo: false,
    };

    const MAX_POINTS = 500;

    // ── LTTB downsampling (JS port of downsample.py) ─────────────────────

    function lttbDownsample(xs, ys, target) {
        const n = xs.length;
        if (n <= target || n <= 2) return { x: xs, y: ys };
        if (target < 3) target = 3;

        const bucketSize = (n - 2) / (target - 2);
        const outX = [xs[0]];
        const outY = [ys[0]];
        let aIdx = 0;

        for (let i = 1; i < target - 1; i++) {
            const bStart = Math.floor((i - 1) * bucketSize) + 1;
            const bEnd = Math.min(Math.floor(i * bucketSize) + 1, n - 1);
            const nStart = Math.floor(i * bucketSize) + 1;
            const nEnd = Math.min(Math.floor((i + 1) * bucketSize) + 1, n - 1);

            let avgX = 0, avgY = 0;
            const nCount = nEnd - nStart;
            for (let j = nStart; j < nEnd; j++) { avgX += xs[j]; avgY += ys[j]; }
            if (nCount > 0) { avgX /= nCount; avgY /= nCount; }

            let maxArea = -1, maxIdx = bStart;
            for (let j = bStart; j < bEnd; j++) {
                const area = Math.abs(
                    (xs[aIdx] - avgX) * (ys[j] - ys[aIdx])
                    - (xs[aIdx] - xs[j]) * (avgY - ys[aIdx])
                );
                if (area > maxArea) { maxArea = area; maxIdx = j; }
            }
            outX.push(xs[maxIdx]);
            outY.push(ys[maxIdx]);
            aIdx = maxIdx;
        }

        outX.push(xs[n - 1]);
        outY.push(ys[n - 1]);
        return { x: outX, y: outY };
    }

    function downsample(xs, ys) {
        if (xs.length <= MAX_POINTS) return { x: xs, y: ys };
        return lttbDownsample(xs, ys, MAX_POINTS);
    }

    // ── Loss vs Epoch ───────────────────────────────────────────────────

    function lossVsEpoch(el, epochs) {
        if (!el) return;
        if (!epochs || !epochs.length) { Plotly.purge(el); return; }

        const xs = epochs.map(e => e.epoch);
        const traces = [];

        if (epochs.some(e => e.train != null)) {
            const ys = epochs.map(e => e.train);
            const ds = downsample(xs, ys);
            traces.push({
                x: ds.x, y: ds.y,
                name: "train", mode: "lines",
                line: { color: "#6c8cff", width: 1.5 },
            });
        }
        if (epochs.some(e => e.val != null)) {
            const ys = epochs.map(e => e.val);
            const ds = downsample(xs, ys);
            traces.push({
                x: ds.x, y: ds.y,
                name: "val", mode: "lines",
                line: { color: "#fb923c", width: 1.5 },
            });
        }
        // Mark best epoch
        let bestEp = null, bestVal = Infinity;
        epochs.forEach(e => { if (e.val != null && e.val < bestVal) { bestVal = e.val; bestEp = e.epoch; } });
        if (bestEp != null) {
            traces.push({
                x: [bestEp], y: [bestVal],
                name: `best @ ep ${bestEp}`, mode: "markers",
                marker: { color: "#4ade80", size: 8, symbol: "star" },
            });
        }
        const layout = {
            ...LAYOUT_BASE,
            title: { text: "Loss vs Epoch", font: { size: 12 } },
            xaxis: { ...LAYOUT_BASE.xaxis, title: "Epoch" },
            yaxis: { ...LAYOUT_BASE.yaxis, title: "Loss" },
        };
        Plotly.react(el, traces, layout, CONFIG);
    }

    // ── Pareto scatter (GA) ─────────────────────────────────────────────

    function paretoScatter(el, individuals, title) {
        if (!el) return;
        if (!individuals || !individuals.length) { Plotly.purge(el); return; }
        const feasible = individuals.filter(i => i.feasible !== false);
        const infeasible = individuals.filter(i => i.feasible === false);
        const traces = [];
        if (feasible.length) {
            traces.push({
                x: feasible.map(i => i.inference_cost),
                y: feasible.map(i => i.val_mse),
                text: feasible.map(i => `gen ${i.born_generation ?? "?"} params ${i.param_count ?? "?"}`),
                mode: "markers", name: "feasible",
                marker: { color: "#6c8cff", size: 7, opacity: 0.8 },
                hovertemplate: "cost: %{x}<br>mse: %{y:.4f}<br>%{text}<extra></extra>",
            });
        }
        if (infeasible.length) {
            traces.push({
                x: infeasible.map(i => i.inference_cost),
                y: infeasible.map(i => i.val_mse),
                text: infeasible.map(i => `cv ${i.constraint_violation ?? "?"}`),
                mode: "markers", name: "infeasible",
                marker: { color: "#f87171", size: 6, symbol: "x", opacity: 0.6 },
                hovertemplate: "cost: %{x}<br>mse: %{y}<br>%{text}<extra></extra>",
            });
        }
        const layout = {
            ...LAYOUT_BASE,
            title: { text: title || "Pareto Front", font: { size: 12 } },
            xaxis: { ...LAYOUT_BASE.xaxis, title: "Inference Cost" },
            yaxis: { ...LAYOUT_BASE.yaxis, title: "Val MSE" },
        };
        Plotly.react(el, traces, layout, CONFIG);
    }

    // ── GA generation progress ──────────────────────────────────────────

    function gaGenerationProgress(el, byGeneration, title) {
        if (!el) return;
        const gens = Object.keys(byGeneration).map(Number).sort((a, b) => a - b);
        if (!gens.length) { Plotly.purge(el); return; }
        const bestPerGen = gens.map(g => {
            const inds = byGeneration[g];
            const vals = inds
                .filter(i => i.feasible !== false && i.val_mse != null)
                .map(i => i.val_mse);
            return vals.length ? Math.min(...vals) : null;
        });
        const costPerGen = gens.map(g => {
            const inds = byGeneration[g];
            const costs = inds
                .filter(i => i.feasible !== false && i.inference_cost != null)
                .map(i => i.inference_cost);
            return costs.length ? Math.min(...costs) : null;
        });
        const traces = [
            {
                x: gens, y: bestPerGen,
                name: "best val_mse", mode: "lines+markers",
                line: { color: "#6c8cff", width: 2 },
                marker: { size: 5 },
                yaxis: "y",
            },
            {
                x: gens, y: costPerGen,
                name: "best cost", mode: "lines+markers",
                line: { color: "#fb923c", width: 2, dash: "dot" },
                marker: { size: 5 },
                yaxis: "y2",
            },
        ];
        const layout = {
            ...LAYOUT_BASE,
            title: { text: title || "Generation Progress", font: { size: 12 } },
            xaxis: { ...LAYOUT_BASE.xaxis, title: "Generation" },
            yaxis: { ...LAYOUT_BASE.yaxis, title: "Best Val MSE", side: "left" },
            yaxis2: {
                ...LAYOUT_BASE.yaxis, title: "Best Cost",
                overlaying: "y", side: "right",
                gridcolor: "rgba(0,0,0,0)",
            },
            legend: { orientation: "h", y: -0.2 },
        };
        Plotly.react(el, traces, layout, CONFIG);
    }

    // ── Multi-config loss overlay ───────────────────────────────────────

    function multiLossOverlay(el, configs, maxLines) {
        if (!el) return;
        maxLines = maxLines || 8;
        if (!configs || !configs.length) { Plotly.purge(el); return; }
        const colors = ["#6c8cff", "#fb923c", "#4ade80", "#f87171", "#a78bfa", "#facc15", "#22d3ee", "#f472b6"];
        const traces = configs.slice(0, maxLines).map((cfg, i) => {
            const epochs = cfg.epochs || [];
            const xs = epochs.map(e => e.epoch);
            const ys = epochs.map(e => e.val);
            const ds = downsample(xs, ys);
            return {
                x: ds.x, y: ds.y,
                name: `${cfg.model} ${cfg.encoding}`,
                mode: "lines",
                line: { color: colors[i % colors.length], width: 1.5 },
            };
        });
        const layout = {
            ...LAYOUT_BASE,
            title: { text: "Val Loss Comparison (active)", font: { size: 12 } },
            xaxis: { ...LAYOUT_BASE.xaxis, title: "Epoch" },
            yaxis: { ...LAYOUT_BASE.yaxis, title: "Val Loss" },
        };
        Plotly.react(el, traces, layout, CONFIG);
    }

    // ── Cleanup ─────────────────────────────────────────────────────────

    function clear(el) {
        if (el) Plotly.purge(el);
    }

    return { lossVsEpoch, paretoScatter, gaGenerationProgress, multiLossOverlay, clear, LAYOUT_BASE, CONFIG };
})();
