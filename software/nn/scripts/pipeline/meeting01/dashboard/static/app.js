/* app.js — main application: tab routing, SSE loop, panel renderers */

const App = (() => {
    let activeTab = "overview";
    let eventSource = null;
    let paused = false;
    let eventsBuffer = [];
    let sortState = { col: null, asc: true };

    // ── Debounced render scheduler ───────────────────────────────────────
    // SSE events arrive every POLL_INTERVAL (2s).  We batch them and flush
    // at most once per FLUSH_MS to avoid layout thrash when multiple panels
    // update in the same tick.

    const FLUSH_MS = 400;
    let _pendingPanels = new Set();
    let _flushTimer = null;

    function scheduleFlush() {
        if (_flushTimer) return;
        _flushTimer = setTimeout(() => {
            _flushTimer = null;
            const panels = new Set(_pendingPanels);
            _pendingPanels.clear();
            flushPanels(panels);
        }, FLUSH_MS);
    }

    function flushPanels(panels) {
        if (panels.has("training")) rerenderTrainingCharts();
        if (panels.has("ga")) rerenderGACharts();
        if (panels.has("comparison")) buildComparisonTable();
    }

    // ── Tab routing ──────────────────────────────────────────────────────

    function initTabs() {
        document.querySelectorAll(".tab").forEach(btn => {
            btn.addEventListener("click", () => {
                setActiveTab(btn.dataset.tab);
            });
        });
    }

    function setActiveTab(tab) {
        activeTab = tab;
        document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${tab}`));
        // Render charts for the now-visible tab
        if (tab === "training") rerenderTrainingCharts();
        if (tab === "ga") rerenderGACharts();
    }

    // ── SSE connection ───────────────────────────────────────────────────

    function connectSSE() {
        if (eventSource) eventSource.close();
        eventSource = new EventSource(API.sseUrl());

        eventSource.onopen = () => {
            document.getElementById("sb-conn").className = "dot dot-green";
            document.getElementById("sb-conn").title = "SSE connected";
        };

        eventSource.onerror = () => {
            document.getElementById("sb-conn").className = "dot dot-red";
            document.getElementById("sb-conn").title = "SSE disconnected — reconnecting…";
        };

        const panels = [
            "summary", "fold_grid", "active_configs", "completed_configs",
            "marginals", "aggregation", "events", "ga_summary", "heartbeat",
        ];
        panels.forEach(name => {
            eventSource.addEventListener(name, (e) => {
                try {
                    const data = JSON.parse(e.data);
                    handlePanel(name, data);
                } catch { /* ignore parse errors */ }
            });
        });
    }

    // ── Panel dispatcher ─────────────────────────────────────────────────

    function handlePanel(name, data) {
        switch (name) {
            case "summary": renderSummary(data); break;
            case "fold_grid": renderFoldGrid(data); break;
            case "active_configs": renderActiveConfigs(data); break;
            case "completed_configs": _completedConfigs = data || []; _pendingPanels.add("comparison"); scheduleFlush(); break;
            case "events": renderEvents(data); break;
            case "ga_summary": _gaSummaries = data || []; _pendingPanels.add("ga"); scheduleFlush(); break;
            case "heartbeat": renderHeartbeat(data); break;
        }
    }

    // ── Overview: summary ────────────────────────────────────────────────

    function renderSummary(s) {
        const c = s.counts || {};
        document.getElementById("sb-counts").textContent =
            `${c.done || 0} done · ${c.running || 0} running · ${c.failed || 0} failed · ${c.total || 0} total`;

        const eta = s.eta_seconds;
        document.getElementById("sb-eta").textContent = eta != null
            ? `ETA ${formatDuration(eta)}`
            : "ETA —";

        const ss = s.session || {};
        const lines = [
            `run_tag:      ${ss.run_tag || "—"}`,
            `datasets:     ${(ss.all_datasets || []).join(", ") || "—"}`,
            `folds:        ${ss.cv_num_folds || "—"}`,
            `repeats:      ${ss.repeats || "—"}`,
            `total_outer:  ${ss.total_outer_runs || "—"}`,
            `grid_size:    ${s.grid_size || "—"}`,
            `per_fold:     ${s.per_fold_trainings || "—"}`,
        ];
        document.getElementById("session-detail").textContent = lines.join("\n");

        const started = s.started_wall;
        if (started) {
            const elapsed = Date.now() / 1000 - started;
            const done = c.done || 0;
            const total = c.total || 0;
            const pct = total > 0 ? (done / total * 100).toFixed(1) : "0.0";
            document.getElementById("runtime-stats").innerHTML =
                `Elapsed: ${formatDuration(elapsed)}<br>` +
                `Done: ${done} / ${total} (${pct}%)<br>` +
                `Rate: ${done > 0 ? (done / elapsed * 3600).toFixed(1) : "—"} configs/h`;
        }
    }

    // ── Overview: fold grid ──────────────────────────────────────────────

    function renderFoldGrid(fg) {
        const el = document.getElementById("fold-grid-table");
        if (!fg || !fg.length) {
            el.innerHTML = "<p class='dim'>No fold data yet</p>";
            return;
        }
        const byDs = {};
        fg.forEach(cell => {
            if (!byDs[cell.dataset]) byDs[cell.dataset] = {};
            byDs[cell.dataset][cell.fold] = cell;
        });

        const datasets = Object.keys(byDs).sort();
        const folds = [...new Set(fg.map(c => c.fold))].sort((a, b) => a - b);

        let html = "<table><thead><tr><th></th>";
        folds.forEach(f => html += `<th>fold ${f}</th>`);
        html += "</tr></thead><tbody>";

        datasets.forEach(ds => {
            html += `<tr><th>${ds}</th>`;
            folds.forEach(f => {
                const cell = byDs[ds]?.[f];
                if (cell) {
                    const cls = `cell-${cell.status}`;
                    html += `<td class="${cls}">${cell.done}/${cell.total}</td>`;
                } else {
                    html += "<td class='cell-unstarted'>—</td>";
                }
            });
            html += "</tr>";
        });
        html += "</tbody></table>";
        el.innerHTML = html;
    }

    // ── Training Now: active configs with per-config charts ──────────────

    let _activeConfigs = [];

    function renderActiveConfigs(configs) {
        _activeConfigs = configs || [];
        const el = document.getElementById("training-configs");
        if (!_activeConfigs.length) {
            el.innerHTML = "<p class='dim'>No active configs — training idle</p>";
            document.getElementById("training-charts").innerHTML = "";
            return;
        }

        let html = "";
        _activeConfigs.forEach(cfg => {
            const uncertain = cfg.sweep_uncertain;
            const cls = uncertain ? "config-card sweep-uncertain" : "config-card";
            const badge = uncertain ? ' <span class="sweep-badge">sweep — identity uncertain</span>' : '';

            const lastTrain = cfg.last_train != null ? cfg.last_train.toFixed(4) : "—";
            const lastVal = cfg.last_val != null ? cfg.last_val.toFixed(4) : "—";
            const bestVal = cfg.best_val != null ? cfg.best_val.toFixed(4) : "—";
            const bestEpoch = cfg.best_epoch != null ? cfg.best_epoch : "—";
            const gap = cfg.gap != null ? cfg.gap.toFixed(4) : "—";
            const noImprove = cfg.no_improve || 0;
            const epochsRun = cfg.epochs_run || 0;
            const maxEp = cfg.max_epochs || 0;
            const pct = maxEp > 0 ? (epochsRun / maxEp * 100).toFixed(0) : "0";
            const chartId = `chart-${cfg.config_id}`;

            html += `<div class="${cls}" data-config-id="${cfg.config_id}">
                <h4>${cfg.dataset} · fold ${cfg.fold} · ${cfg.model} · ${cfg.encoding}${badge}</h4>
                <div class="meta">
                    <span>epoch ${epochsRun}/${maxEp}</span>
                    <span>seed ${cfg.seed}</span>
                    <span>run ${cfg.run_id}</span>
                    ${cfg.param_count != null ? `<span>params ${cfg.param_count.toLocaleString()}</span>` : ""}
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
                <div class="loss-row">
                    train: ${lastTrain} · val: ${lastVal} · best: ${bestVal} @ ep ${bestEpoch}
                </div>
                <div class="loss-row">
                    gap: ${gap} · no-improve: ${noImprove}/${cfg.patience ?? "?"}
                    ${cfg.cur_batch_loss != null ? ` · batch loss: ${cfg.cur_batch_loss.toFixed(4)}` : ""}
                </div>
                <div id="${chartId}" class="config-chart" style="height:220px"></div>
            </div>`;
        });
        el.innerHTML = html;
        _pendingPanels.add("training");
        scheduleFlush();
    }

    function rerenderTrainingCharts() {
        if (activeTab !== "training") return;
        _activeConfigs.forEach(cfg => {
            const el = document.getElementById(`chart-${cfg.config_id}`);
            if (el) Charts.lossVsEpoch(el, cfg.epochs);
        });
        const overlayEl = document.getElementById("training-overlay-chart");
        if (overlayEl) Charts.multiLossOverlay(overlayEl, _activeConfigs);
    }

    // ── Completed configs: sortable comparison table ─────────────────────

    let _completedConfigs = [];

    function buildComparisonTable() {
        const el = document.getElementById("comp-table");
        if (!_completedConfigs.length) {
            el.innerHTML = "<p class='dim'>No completed configs yet</p>";
            return;
        }

        const cols = [
            { key: "_idx", label: "#", fmt: v => v },
            { key: "dataset", label: "Dataset", fmt: v => v },
            { key: "fold", label: "Fold", fmt: v => v },
            { key: "model", label: "Model", fmt: v => v },
            { key: "encoding", label: "Encoding", fmt: v => v },
            { key: "best_val", label: "Best Val", fmt: v => v != null ? v.toFixed(4) : "—" },
            { key: "best_epoch", label: "Best Ep", fmt: v => v ?? "—" },
            { key: "epochs_run", label: "Epochs", fmt: v => v },
            { key: "param_count", label: "Params", fmt: v => v != null ? v.toLocaleString() : "—" },
            { key: "status", label: "Status", fmt: v => v },
        ];

        const bestPerFold = {};
        _completedConfigs.forEach(c => {
            if (c.best_val == null) return;
            const key = `${c.dataset}|${c.fold}`;
            if (!(key in bestPerFold) || c.best_val < bestPerFold[key]) {
                bestPerFold[key] = c.best_val;
            }
        });

        let data = _completedConfigs.map((c, i) => ({ ...c, _idx: i + 1 }));
        if (sortState.col) {
            const col = cols.find(c => c.key === sortState.col);
            if (col) {
                data.sort((a, b) => {
                    let va = a[col.key], vb = b[col.key];
                    if (va == null) return 1;
                    if (vb == null) return -1;
                    if (typeof va === "string") return sortState.asc ? va.localeCompare(vb) : vb.localeCompare(va);
                    return sortState.asc ? va - vb : vb - va;
                });
            }
        }

        let html = "<table><thead><tr>";
        cols.forEach(col => {
            const arrow = sortState.col === col.key ? (sortState.asc ? " ▲" : " ▼") : "";
            html += `<th data-sort="${col.key}" class="sortable">${col.label}${arrow}</th>`;
        });
        html += "</tr></thead><tbody>";

        data.forEach(row => {
            const key = `${row.dataset}|${row.fold}`;
            const isBest = row.best_val != null && row.best_val === bestPerFold[key];
            html += `<tr${isBest ? ' class="best-row"' : ""}>`;
            cols.forEach(col => {
                const cls = isBest && col.key === "best_val" ? ' class="best-val"' : "";
                html += `<td${cls}>${col.fmt(row[col.key])}</td>`;
            });
            html += "</tr>";
        });
        html += "</tbody></table>";
        el.innerHTML = html;

        el.querySelectorAll("th.sortable").forEach(th => {
            th.addEventListener("click", () => {
                const key = th.dataset.sort;
                if (sortState.col === key) {
                    sortState.asc = !sortState.asc;
                } else {
                    sortState.col = key;
                    sortState.asc = true;
                }
                buildComparisonTable();
            });
        });
    }

    // ── Architecture Search: GA cards with Pareto + generation charts ────

    let _gaSummaries = [];

    function rerenderGACharts() {
        if (activeTab !== "ga") return;
        const el = document.getElementById("ga-cards");
        if (!_gaSummaries.length) {
            el.innerHTML = "<p class='dim'>No GA search data yet — searches appear here when running</p>";
            return;
        }
        let html = "";
        _gaSummaries.forEach(s => {
            const chartId = `pareto-${s.dataset}-${s.fold}-${s.run_id}-${s.family}`;
            const genChartId = `gen-${s.dataset}-${s.fold}-${s.run_id}-${s.family}`;
            const nGens = s.generations?.length || 0;
            const maxGen = s.max_generation || 0;

            html += `<div class="ga-card">
                <h4>${s.dataset} · fold ${s.fold} · run ${s.run_id} · ${s.family.toUpperCase()}</h4>
                <div class="stats">
                    <span>individuals: ${s.n_individuals}</span>
                    <span>generations: ${nGens} (max ${maxGen})</span>
                    <span>best val_mse: ${s.best_val_mse != null ? s.best_val_mse.toFixed(4) : "—"}</span>
                    <span>best cost: ${s.best_inference_cost ?? "—"}</span>
                    <span>pareto size: ${s.pareto?.length || 0}</span>
                </div>
                <div class="null-notice">${s.mutation_crossover_note || ""}</div>
            </div>`;
            html += `<div class="ga-chart-pair">
                <div id="${chartId}" class="ga-chart" style="height:250px"></div>
                <div id="${genChartId}" class="ga-chart" style="height:250px"></div>
            </div>`;
        });
        el.innerHTML = html;

        requestAnimationFrame(() => {
            _gaSummaries.forEach(s => {
                const paretoEl = document.getElementById(`pareto-${s.dataset}-${s.fold}-${s.run_id}-${s.family}`);
                if (paretoEl && s.pareto) {
                    Charts.paretoScatter(paretoEl, s.pareto,
                        `Pareto: ${s.dataset} f${s.fold} ${s.family}`);
                }
            });
        });
    }

    // ── Events log ───────────────────────────────────────────────────────

    function renderEvents(events) {
        if (!paused) {
            eventsBuffer = events || [];
        }
        const el = document.getElementById("events-log");
        const filter = document.getElementById("events-filter").value.toLowerCase();
        const filtered = filter
            ? eventsBuffer.filter(e => JSON.stringify(e).toLowerCase().includes(filter))
            : eventsBuffer;

        const display = filtered.slice(-500);
        el.innerHTML = display.map(e => {
            const ts = e.ts_unix ? new Date(e.ts_unix * 1000).toLocaleTimeString() : "";
            const type = e.type || "?";
            const detail = eventDetail(e);
            return `<div class="event-line"><span class="ev-ts">${ts}</span><span class="ev-type">${type}</span>${detail}</div>`;
        }).join("");
        el.scrollTop = el.scrollHeight;
    }

    function eventDetail(e) {
        switch (e.type) {
            case "config_begin":
                return ` — ${e.dataset} fold${e.fold} ${e.model} ${e.encoding} run${e.run_id}`;
            case "epoch_end":
                return ` — ep ${e.epoch} train ${e.train_loss?.toFixed(4) ?? "?"} val ${e.val_loss?.toFixed(4) ?? "?"}`;
            case "config_end":
                return ` — ${e.dataset} fold${e.fold} ${e.model} ${e.encoding} best ${e.best_val?.toFixed(4) ?? "?"} @ ep ${e.best_epoch ?? "?"} (${e.stop_reason})`;
            case "session_begin":
                return ` — tag ${e.run_tag} ${e.all_datasets?.join(",")}`;
            case "family_begin":
                return ` — ${e.dataset} fold${e.fold} run${e.run_id} ${e.family}`;
            default:
                return "";
        }
    }

    // ── Heartbeat ────────────────────────────────────────────────────────

    function renderHeartbeat(h) {
        const ts = h.ts ? new Date(h.ts * 1000).toLocaleTimeString() : "—";
        document.getElementById("sb-poll").textContent = `last poll ${ts}`;
    }

    // ── Utilities ────────────────────────────────────────────────────────

    function formatDuration(seconds) {
        if (seconds == null || !isFinite(seconds)) return "—";
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    }

    // ── Init ─────────────────────────────────────────────────────────────

    function init() {
        initTabs();

        document.getElementById("events-filter").addEventListener("input", () => {
            renderEvents(eventsBuffer);
        });
        document.getElementById("events-pause").addEventListener("change", (e) => {
            paused = e.target.checked;
        });

        document.getElementById("btn-collect-ga").addEventListener("click", collectRemoteGA);

        connectSSE();
    }

    async function collectRemoteGA() {
        const btn = document.getElementById("btn-collect-ga");
        const statusEl = document.getElementById("remote-ga-status");
        const resultsEl = document.getElementById("remote-ga-results");
        btn.disabled = true;
        statusEl.textContent = "connecting to GridUnesp…";
        try {
            const resp = await fetch(`/api/ga/remote/collect?results_dir=${encodeURIComponent(API.resultsDir)}&run_tag=${encodeURIComponent(API.runTag)}`, { method: "POST" });
            const data = await resp.json();
            if (data.error) {
                statusEl.textContent = `error: ${data.error}`;
                statusEl.className = "";
            } else {
                const cells = data.cells || [];
                statusEl.textContent = `${cells.length} cells collected at ${new Date(data.ts * 1000).toLocaleTimeString()}`;
                statusEl.className = "";
                if (cells.length) {
                    let html = "<table><thead><tr><th>Dataset</th><th>Fold</th><th>Run</th><th>Family</th><th>Individuals</th><th>Gens</th><th>Best MSE</th><th>Best Cost</th></tr></thead><tbody>";
                    cells.forEach(c => {
                        html += `<tr><td>${c.dataset}</td><td>${c.fold}</td><td>${c.run_id}</td><td>${c.family}</td><td>${c.n_individuals}</td><td>${c.n_generations}</td><td>${c.best_val_mse != null ? c.best_val_mse.toFixed(4) : "—"}</td><td>${c.best_inference_cost ?? "—"}</td></tr>`;
                    });
                    html += "</tbody></table>";
                    resultsEl.innerHTML = html;
                }
            }
        } catch (exc) {
            statusEl.textContent = `error: ${exc.message}`;
        }
        btn.disabled = false;
    }

    document.addEventListener("DOMContentLoaded", init);

    return { setActiveTab, connectSSE };
})();
