"""gridunesp_config.py — shared SSH/connection config for GridUnesp.

Single source of truth for credentials and connection parameters, imported by
every LOCAL script that itself opens an SSH connection to GridUnesp:
gridunesp_tui.py, collect_ga_local.py, and the dashboard server's remote mode.
gridunesp_status_remote.py is deliberately NOT a consumer here -- it runs ON
the GridUnesp login node, inside a connection one of the scripts above already
established, so it never needs credentials of its own.

Reads from the .env file (same format as _gridunesp_env.sh) with environment
variable fallback.  File is chmod 600, git-ignored.

Authentication
--------------
Two supported modes, chosen automatically from whether a password is available:

* **password** — ``sshpass -e ssh ...`` with the password passed through the
  ``SSHPASS`` environment variable, never argv, so it does not appear in ``ps``
  output.  This matches ``_gridunesp_env.sh`` and ``gridunesp_tui.py``.
* **SSH key** — plain ``ssh -o BatchMode=yes`` with no password at all.  The
  password is then never required, prompted for, or stored.

Note that ``BatchMode=yes`` and password auth are mutually exclusive: BatchMode
suppresses every interactive prompt, so a password can never be typed.  The two
are therefore selected together, never mixed.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from dataclasses import dataclass
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(_SCRIPT_DIR, ".env")

# GridUnesp connection defaults.  These MUST match the values the rest of the
# pipeline already uses (gridunesp_tui.py's argparse defaults and
# GridUnesp-Deployment.md's "confirmed" cluster-facts table): the login node is
# access.grid.unesp.br, and the checkout lives at software/nn under $HOME.
DEFAULT_HOST = "access.grid.unesp.br"
DEFAULT_PORT = 22
# Remote checkout root (where the nn repo lives), relative to the login account's
# home directory.  Relative, not absolute and not "~"-prefixed: the remote
# command `cd`s into it, and os.path.join() composes result paths from it, and
# neither expands "~" the way an interactive shell would.
DEFAULT_REMOTE_DIR = "software/nn"
# Conda environment to activate on the remote.  Required: GridUnesp's bare
# python3 is 3.6.8, which cannot even parse these scripts (they use
# `from __future__ import annotations`).  See GridUnesp-Deployment.md.
DEFAULT_CONDA_ENV = "meeting01-build"
DEFAULT_CONDA_MODULE = "miniconda/24.4.0-libmamba"


@dataclass(frozen=True)
class GridUnespConfig:
    user: str
    password: Optional[str] = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    remote_dir: str = DEFAULT_REMOTE_DIR
    conda_env: str = DEFAULT_CONDA_ENV
    conda_module: str = DEFAULT_CONDA_MODULE

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def uses_password(self) -> bool:
        return bool(self.password)

    def remote_activate(self) -> str:
        """Shell fragment that activates the remote conda env.

        This exact three-step form is required on GridUnesp and is not
        interchangeable with any shorter variant:

        * ``module load`` puts the conda *binary* on PATH — it does not install
          the ``conda activate`` shell function.
        * ``eval "$(conda shell.bash hook)"`` installs that function.  Without
          it every ``conda activate`` fails with
          ``CondaError: Run 'conda init' before 'conda activate'`` because the
          call falls through to the raw binary's ``activate`` subcommand.
        * Only then does ``conda activate`` work.

        A bare ``source activate`` / ``conda activate`` pair does NOT work:
        ``source activate`` names a file that does not exist, and the fallback
        ``conda activate`` hits the missing-hook error above — leaving the
        system python3 (3.6.8) in place.
        """
        return (
            f"module load {self.conda_module} && "
            'eval "$(conda shell.bash hook)" && '
            f"conda activate {self.conda_env}"
        )

    def ssh_argv(self, remote_cmd: str, port: Optional[int] = None) -> list[str]:
        """Build a non-interactive SSH argv list for subprocess.run().

        Returns a command that self-authenticates: ``sshpass -e ssh`` when a
        password is configured, otherwise ``ssh -o BatchMode=yes`` for key-based
        auth.  Pair it with :meth:`ssh_env` so ``sshpass -e`` has SSHPASS to read.
        """
        common = [
            "-o", "ConnectTimeout=15",
            "-p", str(port or self.port),
            self.ssh_target,
            remote_cmd,
        ]
        if self.uses_password:
            if shutil.which("sshpass") is None:
                raise RuntimeError(
                    "'sshpass' not found on PATH — needed to use the configured "
                    "GRIDUNESP_PASSWORD non-interactively. Install it: "
                    "'sudo apt install sshpass' or "
                    "'conda install -c conda-forge sshpass'. Alternatively remove "
                    "GRIDUNESP_PASSWORD from .env and use an SSH key."
                )
            return ["sshpass", "-e", "ssh", *common]
        return ["ssh", "-o", "BatchMode=yes", *common]

    def ssh_cmd(self, remote_cmd: str, port: Optional[int] = None) -> list[str]:
        """Backwards-compatible alias for :meth:`ssh_argv`."""
        return self.ssh_argv(remote_cmd, port)

    def ssh_env(self) -> dict[str, str]:
        """Environment for subprocess.run() carrying SSHPASS for ``sshpass -e``."""
        env = dict(os.environ)
        if self.password:
            env["SSHPASS"] = self.password
        return env


def _read_env_file() -> dict[str, str]:
    env: dict[str, str] = {}
    if not os.path.isfile(ENV_FILE):
        return env
    with open(ENV_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def load_config() -> GridUnespConfig:
    """Load connection config from .env, environment variables, or a prompt.

    Only the *username* is mandatory.  A password is used when present
    (``sshpass`` path) and otherwise omitted entirely, letting SSH key
    authentication work with ``BatchMode=yes``.  Set ``GRIDUNESP_PASSWORD`` in
    .env to force password auth; leave it unset to use a key.
    """
    env = _read_env_file()

    def pick(key: str, default: Optional[str] = None) -> Optional[str]:
        return env.get(key) or os.environ.get(key) or default

    user = pick("GRIDUNESP_USER")
    password = pick("GRIDUNESP_PASSWORD")
    host = pick("GRIDUNESP_HOST", DEFAULT_HOST) or DEFAULT_HOST
    remote_dir = pick("GRIDUNESP_REMOTE_DIR", DEFAULT_REMOTE_DIR) or DEFAULT_REMOTE_DIR
    conda_env = pick("GRIDUNESP_CONDA_ENV", DEFAULT_CONDA_ENV) or DEFAULT_CONDA_ENV
    conda_module = pick("GRIDUNESP_CONDA_MODULE", DEFAULT_CONDA_MODULE) or DEFAULT_CONDA_MODULE
    try:
        port = int(pick("GRIDUNESP_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
    except ValueError:
        port = DEFAULT_PORT

    if user and password:
        return GridUnespConfig(user=user, password=password, host=host, port=port,
                               remote_dir=remote_dir, conda_env=conda_env,
                               conda_module=conda_module)

    # No password configured: either an SSH key is in play, or we need a username.
    if not user:
        if not sys.stdin.isatty():
            raise SystemExit(
                "gridunesp_config: GRIDUNESP_USER is unset and stdin is not a "
                "terminal, so there is nobody to prompt. Set GRIDUNESP_USER in "
                f"{ENV_FILE} or the environment."
            )
        user = input("GridUnesp username: ").strip()
        if not user:
            raise SystemExit("gridunesp_config: username cannot be empty")

    # Only persist a password if the user actually gave us one; a key-based
    # setup stores nothing at all.
    if not password and sys.stdin.isatty() and pick("GRIDUNESP_NO_PASSWORD_PROMPT") != "1":
        import getpass
        answer = getpass.getpass(
            "GridUnesp password (blank to use an SSH key instead): "
        ).strip()
        password = answer or None

    if password:
        fd = os.open(ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(f"GRIDUNESP_USER={user}\nGRIDUNESP_PASSWORD={password}\n")
        os.chmod(ENV_FILE, stat.S_IRUSR | stat.S_IWUSR)
        print(f"[gridunesp] saved credentials to {ENV_FILE} (chmod 600, git-ignored)")
    else:
        print("[gridunesp] no password configured — using SSH key authentication",
              file=sys.stderr)

    return GridUnespConfig(user=user, password=password, host=host, port=port,
                           remote_dir=remote_dir, conda_env=conda_env,
                           conda_module=conda_module)
