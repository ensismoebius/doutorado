"""gridunesp_config.py — shared SSH/connection config for GridUnesp.

Single source of truth for credentials and connection parameters, imported
by gridunesp_tui.py, gridunesp_status_remote.py, and the dashboard server.

Reads from .env file (same format as _gridunesp_env.sh) with environment
variable fallback.  File is chmod 600, git-ignored.
"""
from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(_SCRIPT_DIR, ".env")

# GridUnesp connection defaults
DEFAULT_HOST = "gridunesp.caaresearch.com.br"
DEFAULT_PORT = 22
# Remote checkout root (where the nn repo lives)
DEFAULT_REMOTE_DIR = "~/doutorado/software/nn"
# Conda environment to activate on the remote
DEFAULT_CONDA_ENV = "meeting01-build"


@dataclass(frozen=True)
class GridUnespConfig:
    user: str
    password: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    remote_dir: str = DEFAULT_REMOTE_DIR
    conda_env: str = DEFAULT_CONDA_ENV

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_cmd(self, remote_cmd: str, port: Optional[int] = None) -> list[str]:
        """Build a non-interactive SSH command list for subprocess.run()."""
        return [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-p", str(port or self.port),
            self.ssh_target,
            remote_cmd,
        ]


def load_config() -> GridUnespConfig:
    """Load credentials from .env file, environment variables, or interactive prompt."""
    env: dict[str, str] = {}
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()

    user = env.get("GRIDUNESP_USER") or os.environ.get("GRIDUNESP_USER")
    password = env.get("GRIDUNESP_PASSWORD") or os.environ.get("GRIDUNESP_PASSWORD")
    host = env.get("GRIDUNESP_HOST", DEFAULT_HOST)
    port = int(env.get("GRIDUNESP_PORT", str(DEFAULT_PORT)))
    remote_dir = env.get("GRIDUNESP_REMOTE_DIR", DEFAULT_REMOTE_DIR)
    conda_env = env.get("GRIDUNESP_CONDA_ENV", DEFAULT_CONDA_ENV)

    if user and password:
        return GridUnespConfig(user=user, password=password, host=host,
                               port=port, remote_dir=remote_dir, conda_env=conda_env)

    # Interactive prompt fallback (only when stdin is a terminal)
    if not sys.stdin.isatty():
        print("gridunesp_config: credentials missing (no .env, stdin is not a terminal)",
              file=sys.stderr)
        raise SystemExit(1)

    if not user:
        user = input("GridUnesp username: ").strip()
        if not user:
            print("gridunesp_config: username cannot be empty", file=sys.stderr)
            raise SystemExit(1)
    if not password:
        import getpass
        password = getpass.getpass("GridUnesp password: ")
        if not password:
            print("gridunesp_config: password cannot be empty", file=sys.stderr)
            raise SystemExit(1)

    # Save for next time
    fd = os.open(ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"GRIDUNESP_USER={user}\nGRIDUNESP_PASSWORD={password}\n")
    os.chmod(ENV_FILE, stat.S_IRUSR | stat.S_IWUSR)
    print(f"[gridunesp] saved credentials to {ENV_FILE} (chmod 600, git-ignored)")

    return GridUnespConfig(user=user, password=password, host=host,
                           port=port, remote_dir=remote_dir, conda_env=conda_env)
