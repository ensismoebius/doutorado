/* api.js — REST + SSE client for the meeting01 dashboard */

const API = (() => {
    const params = new URLSearchParams(window.location.search);
    const resultsDir = params.get("results_dir") || "results/meeting01";
    const runTag = params.get("run_tag") || "meeting01_loso";
    const baseQS = `results_dir=${encodeURIComponent(resultsDir)}&run_tag=${encodeURIComponent(runTag)}`;

    async function get(path) {
        const sep = path.includes("?") ? "&" : "?";
        const resp = await fetch(`${path}${sep}${baseQS}`);
        if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
        return resp.json();
    }

    function sseUrl() {
        return `/api/stream?${baseQS}`;
    }

    return { get, sseUrl, resultsDir, runTag };
})();
