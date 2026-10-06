#!/usr/bin/env python3
"""gridunesp_tui.py — one multi-panel dashboard for every GridUnesp script in this
directory: dataset-fetch progress, build state, Slurm queue, live training progress,
and action keys to deploy / submit / pull results / validate the toolchain, all
without leaving the terminal.

This is a CONTROLLER, not a reimplementation: every panel's data and every action
shells out to (or imports) the SAME scripts already documented in
.wiki/Guides/GridUnesp-Deployment.md --

  panel data  <- gridunesp_status_remote.py (one persistent SSH session, JSON Lines,
                 reusing monitor.py's own SessionState/EventTailer for the training
                 numbers -- never a second, parallel reimplementation of that logic)
  d Deploy    -> gridunesp_deploy.sh
  s Submit    -> the same `sbatch ...sbatch` command the deployment scripts print
  m Monitor   -> remote_monitor.sh (full monitor.py rich dashboard, fullscreen)
  p Pull      -> pull_progress.sh
  v Validate  -> run_gridunesp_docker_sim.sh (local Docker toolchain check)

Actions that hand off to another interactive program (Deploy/Monitor/Validate) use
Textual's App.suspend() -- this app's screen steps aside, the other program gets the
real terminal, and this dashboard resumes when it exits. Submit is NOT suspended: it
is one non-interactive `ssh ... sbatch ...` call behind a confirmation modal, since
it starts a job that can run for up to 30 days.

One persistent SSH connection for the whole session (opened once, on start, kept
open) -- NOT a reconnect-per-poll loop. This project's Fail2Ban lockout concern
(.wiki/Guides/GridUnesp-Deployment.md) means a dropped connection is NOT retried
automatically; press 'r' to reconnect by hand when you actually want to.

Usage:
  .venv/bin/python3 scripts/pipeline/meeting01/gridunesp_tui.py
  .venv/bin/python3 scripts/pipeline/meeting01/gridunesp_tui.py --interval 30

Credentials/connection config are handled by gridunesp_config.py (shared with
gridunesp_status_remote.py's one-shot GA collector and the web dashboard backend):
first run prompts for your GridUnesp username, then a password (or leave it blank
to use an SSH key instead), and saves whichever applies to
scripts/pipeline/meeting01/.env (chmod 600, git-ignored); later runs read it
silently. Password mode needs `sshpass` locally (never on GridUnesp itself); key
mode needs neither `sshpass` nor a stored password.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any, Optional

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.screen import ModalScreen
    from textual.widgets import Button, Footer, Header, Label, Static
except ImportError:
    print("gridunesp_tui.py: 'textual' is not installed -- this is a LOCAL-machine-only\n"
          "dependency (never needed on GridUnesp itself). Install it:\n"
          "  pip install 'textual>=0.80'\n"
          "or, if this checkout has a .venv: .venv/bin/pip install -r scripts/requirements.txt",
          file=sys.stderr)
    raise SystemExit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Shared SSH/credential config (gridunesp_config.py) -- same .env file, same format,
# as _gridunesp_env.sh, so every script in this directory (this one, the dashboard
# backend, collect_ga_local.py) reads/writes the identical file and stays in sync
# automatically instead of each keeping its own copy of this logic.
from gridunesp_config import GridUnespConfig, load_config  # noqa: E402


# --------------------------------------------------------------------------------------
# status model -- mirrors gridunesp_status_remote.py's JSON schema; all-defaults means
# "nothing received yet", rendered as such rather than as zeros that look like real data.
# --------------------------------------------------------------------------------------
@dataclass
class Status:
    connected: bool = False
    last_update: Optional[float] = None
    error: str = ""
    processes: list[str] = field(default_factory=list)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    build: dict[str, bool] = field(default_factory=dict)
    squeue: list[dict[str, str]] = field(default_factory=list)
    training: dict[str, Any] = field(default_factory=dict)


def _fmt_ago(ts: Optional[float]) -> str:
    if ts is None:
        return "never"
    s = int(time.time() - ts)
    return f"{s}s ago" if s < 90 else f"{s // 60}m ago"


def _fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "?"
    s = int(seconds)
    if s >= 86400:
        return f"{s // 86400}d {(s % 86400) // 3600}h"
    return f"{s // 3600}h {(s % 3600) // 60}m"


# --------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------
def _fmt_rate(bytes_s: Optional[float]) -> str:
    if not bytes_s or bytes_s <= 0:
        return "-"
    for unit, div in (("G/s", 1024**3), ("M/s", 1024**2), ("K/s", 1024)):
        if bytes_s >= div:
            return f"{bytes_s / div:.1f}{unit}"
    return f"{bytes_s:.0f}B/s"


_PHASE_LABEL = {
    "pending": "[dim]pending[/dim]",
    "staging": "[yellow]cloning/resampling[/yellow]",
    "downloading": "[cyan]downloading[/cyan]",
    "complete": "[bold green]DONE[/bold green]",
}


class DatasetsPanel(Static):
    def update_status(self, st: Status) -> None:
        from rich.table import Table
        t = Table(title="Datasets", expand=True)
        t.add_column("name")
        t.add_column("progress", ratio=1)
        t.add_column("count", justify="right")
        t.add_column("size", justify="right")
        t.add_column("rate", justify="right")
        t.add_column("ETA", justify="right")
        if not st.datasets:
            self.update(t)
            return
        for name, d in st.datasets.items():
            done, total = d.get("done", 0), max(1, d.get("total", 1))
            frac = min(1.0, done / total)
            bar_w = 16
            filled = int(round(frac * bar_w))
            bar = "[green]" + "█" * filled + "[/green]" + "░" * (bar_w - filled)
            phase = d.get("phase", "pending")
            label = _PHASE_LABEL.get(phase, phase)
            progress_cell = label if phase in ("pending", "complete") else f"{label} {bar}"
            rate = _fmt_rate(d.get("rate_bytes_s")) if phase != "complete" else "-"
            eta = _fmt_hms(d.get("eta_s")) if phase != "complete" and d.get("eta_s") else "-"
            t.add_row(name, progress_cell, f"{done}/{d.get('total', 0)}",
                      d.get("size", "-"), rate, eta)
        self.update(t)


class BuildPanel(Static):
    def update_status(self, st: Status) -> None:
        from rich.table import Table
        t = Table(title="Build", expand=True)
        t.add_column("step")
        t.add_column("state", justify="right")
        if not st.build:
            self.update(t)
            return
        for step, key in (("configured", "configured"), ("binary built", "built")):
            ok = st.build.get(key, False)
            t.add_row(step, "[bold green]yes[/bold green]" if ok else "[yellow]no[/yellow]")
        self.update(t)


class QueuePanel(Static):
    def update_status(self, st: Status) -> None:
        from rich.table import Table
        t = Table(title="Slurm queue", expand=True)
        for col in ("job", "partition", "name", "state", "elapsed", "nodes"):
            t.add_column(col)
        if not st.squeue:
            t.add_row("-", "-", "(nothing queued/running)", "-", "-", "-")
        for row in st.squeue:
            t.add_row(row["jobid"], row["partition"], row["name"], row["state"],
                      row["elapsed"], row["nodes"])
        self.update(t)


class TrainingPanel(Static):
    def update_status(self, st: Status) -> None:
        from rich.table import Table
        from rich.console import Group

        tr = st.training
        if not tr.get("has_data"):
            self.update("[dim]no training run data yet (results/meeting01/ empty on the "
                        "remote, or the run hasn't started)[/dim]")
            return

        head = Table.grid(padding=(0, 2))
        head.add_column()
        head.add_column()
        total = max(1, tr.get("total", 1))
        done = tr.get("done", 0)
        frac = min(1.0, done / total)
        bar_w = 30
        filled = int(round(frac * bar_w))
        bar = "[green]" + "█" * filled + "[/green]" + "░" * (bar_w - filled)
        head.add_row(f"[bold]{tr.get('run_tag', '?')}[/bold]", "")
        head.add_row(bar, f"{done} done, {tr.get('running', 0)} running, "
                          f"{tr.get('failed', 0)} failed / ~{tr.get('total', '?')} "
                          f"({tr.get('total_kind', '?')})")
        head.add_row(f"elapsed {_fmt_hms(tr.get('elapsed_s'))}",
                     f"ETA {_fmt_hms(tr.get('eta_s'))} (rough)")

        active_t = Table(title="training now", expand=True)
        for col in ("where", "model/encoding", "epoch", "train", "val", "best val"):
            active_t.add_column(col)
        for a in tr.get("active", []):
            ep = f"{a.get('epoch', '?')}/{a.get('max_epochs', '?')}"
            active_t.add_row(a.get("where", "?"), f"{a.get('model', '')}/{a.get('encoding', '')}",
                             ep, f"{a.get('train_loss'):.5f}" if a.get("train_loss") is not None else "-",
                             f"{a.get('val_loss'):.5f}" if a.get("val_loss") is not None else "-",
                             f"{a.get('best_val'):.5f}" if a.get("best_val") is not None else "-")
        if not tr.get("active"):
            active_t.add_row("-", "(nothing training right now)", "-", "-", "-", "-")

        parts: list[Any] = [head, active_t]
        for f in tr.get("failed_procs", []):
            parts.append(f"[bold red]FAILED[/bold red] {f}")
        self.update(Group(*parts))


class LogPanel(Static):
    def update_status(self, st: Status) -> None:
        lines = []
        conn = "[green]connected[/green]" if st.connected else "[bold red]disconnected[/bold red]"
        lines.append(f"{conn}   last update: {_fmt_ago(st.last_update)}")
        if st.error:
            lines.append(f"[yellow]{st.error}[/yellow]")
        lines.append("")
        lines.append("[bold]active remote processes[/bold]")
        if st.processes:
            lines.extend(f"  {p}" for p in st.processes)
        else:
            lines.append("  (none matching ensure_datasets/wget/cmake/srun)")
        self.update("\n".join(lines))


# --------------------------------------------------------------------------------------
# confirm modal -- used before Submit (starts a job that can run up to 30 days)
# --------------------------------------------------------------------------------------
class ConfirmScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    #dialog { width: 60; height: auto; border: thick $warning; padding: 1 2; background: $surface; }
    #buttons { align: center middle; height: auto; padding-top: 1; }
    Button { margin: 0 2; }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._message)
            with Horizontal(id="buttons"):
                yield Button("Yes", id="yes", variant="error")
                yield Button("No", id="no", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


# --------------------------------------------------------------------------------------
# main app
# --------------------------------------------------------------------------------------
class GridUnespTUI(App):
    TITLE = "GridUnesp Control"
    BINDINGS = [
        Binding("d", "deploy", "Deploy"),
        Binding("s", "submit", "Submit"),
        Binding("m", "full_monitor", "Monitor"),
        Binding("p", "pull", "Pull results"),
        Binding("v", "validate", "Validate (Docker)"),
        Binding("r", "reconnect", "Reconnect"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    #main { layout: grid; grid-size: 2; grid-columns: 1fr 1fr; grid-rows: 1fr 1fr 1fr; }
    #datasets { row-span: 1; }
    #build { row-span: 1; }
    #queue { row-span: 1; }
    #training { row-span: 3; }
    #log { row-span: 1; }
    Static { border: round $primary; padding: 0 1; }
    """

    status: reactive[Status] = reactive(Status, always_update=True)

    def __init__(self, config: GridUnespConfig, interval: float) -> None:
        super().__init__()
        self._config = config
        self._interval = interval
        self._proc: Optional[subprocess.Popen] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main"):
            yield DatasetsPanel(id="datasets")
            yield TrainingPanel(id="training")
            yield BuildPanel(id="build")
            yield QueuePanel(id="queue")
            yield LogPanel(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self._start_stream()

    def watch_status(self, st: Status) -> None:
        self.query_one("#datasets", DatasetsPanel).update_status(st)
        self.query_one("#build", BuildPanel).update_status(st)
        self.query_one("#queue", QueuePanel).update_status(st)
        self.query_one("#training", TrainingPanel).update_status(st)
        self.query_one("#log", LogPanel).update_status(st)

    # ---- SSH status stream ---------------------------------------------------------
    def _start_stream(self) -> None:
        self.run_worker(self._stream_worker(), exclusive=True, thread=False, name="status")

    async def _stream_worker(self) -> None:
        import asyncio

        remote_cmd = (f"cd {shlex.quote(self._config.remote_dir)} && "
                      f"{self._config.remote_activate()} && "
                      f"python3 -u scripts/pipeline/meeting01/gridunesp_status_remote.py "
                      f"--interval {self._interval}")
        try:
            argv = self._config.ssh_argv(remote_cmd)
        except RuntimeError as exc:
            self.status = Status(connected=False, error=str(exc))
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, env=self._config.ssh_env(),
            )
        except FileNotFoundError as exc:
            self.status = Status(connected=False, error=f"'ssh' not found on PATH ({exc})")
            return

        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                blob = json.loads(text)
            except json.JSONDecodeError:
                continue  # conda's activation banner, module-load chatter, etc.
            if "error" in blob and len(blob) <= 2:
                self.status = Status(connected=True, last_update=blob.get("ts"),
                                     error=str(blob["error"]),
                                     processes=self.status.processes,
                                     datasets=self.status.datasets, build=self.status.build,
                                     squeue=self.status.squeue, training=self.status.training)
                continue
            self.status = Status(
                connected=True, last_update=blob.get("ts"),
                processes=blob.get("processes", []), datasets=blob.get("datasets", {}),
                build=blob.get("build", {}), squeue=blob.get("squeue", []),
                training=blob.get("training", {}),
            )
        await proc.wait()
        prev = self.status
        self.status = Status(connected=False, last_update=prev.last_update,
                             error="connection ended -- press 'r' to reconnect",
                             processes=prev.processes, datasets=prev.datasets,
                             build=prev.build, squeue=prev.squeue, training=prev.training)

    def action_reconnect(self) -> None:
        self._start_stream()

    # ---- actions that hand off the terminal ----------------------------------------
    def _run_foreground(self, argv: list[str], cwd: str) -> None:
        with self.suspend():
            subprocess.run(argv, cwd=cwd, check=False)

    def action_deploy(self) -> None:
        self._run_foreground(["bash", os.path.join(SCRIPT_DIR, "gridunesp_deploy.sh")], ROOT_DIR)
        self._start_stream()

    def action_full_monitor(self) -> None:
        self._run_foreground(["bash", os.path.join(SCRIPT_DIR, "remote_monitor.sh")], ROOT_DIR)

    def action_pull(self) -> None:
        self._run_foreground(["bash", os.path.join(SCRIPT_DIR, "pull_progress.sh")], ROOT_DIR)

    def action_validate(self) -> None:
        self._run_foreground(["bash", os.path.join(SCRIPT_DIR, "run_gridunesp_docker_sim.sh")], ROOT_DIR)

    def action_submit(self) -> None:
        def handle(confirmed: bool | None) -> None:
            if not confirmed:
                return
            remote_cmd = (f"cd {shlex.quote(self._config.remote_dir)} && "
                          "sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch")
            with self.suspend():
                try:
                    argv = self._config.ssh_argv(remote_cmd)
                except RuntimeError as exc:
                    print(f"\n[gridunesp-tui] {exc}")
                else:
                    subprocess.run(argv, env=self._config.ssh_env(), check=False)
                input("\n[gridunesp-tui] press Enter to return to the dashboard...")
        self.push_screen(
            ConfirmScreen("Submit the real training job to Slurm?\n"
                          "This can run for up to 30 days (the `long` partition's cap).\n"
                          "Make sure datasets are complete and the binary is built first "
                          "-- check the Datasets/Build panels."),
            handle,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=20.0,
                    help="seconds between remote status samples (default 20)")
    ap.add_argument("--host", default=None,
                    help="default: GRIDUNESP_HOST from .env/environment, else access.grid.unesp.br")
    ap.add_argument("--remote-dir", default=None,
                    help="default: GRIDUNESP_REMOTE_DIR from .env/environment, else software/nn")
    args = ap.parse_args()

    config = load_config()
    overrides: dict[str, str] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.remote_dir is not None:
        overrides["remote_dir"] = args.remote_dir
    if overrides:
        config = dataclass_replace(config, **overrides)

    if config.uses_password and shutil.which("sshpass") is None:
        print("gridunesp_tui.py: 'sshpass' not found on PATH -- needed to use the saved\n"
              "password non-interactively. Install it: 'sudo apt install sshpass'\n"
              "(Debian/Ubuntu) or 'conda install -c conda-forge sshpass'. Alternatively "
              "remove GRIDUNESP_PASSWORD from .env and use an SSH key.", file=sys.stderr)
        return 1

    app = GridUnespTUI(config, args.interval)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
