#!/usr/bin/env python3
"""source_revision.py -- which code produced a run.

The problem. Weeks into a cluster run, "which binary is this?" had no answer. The tree shipped
to GridUnesp carries no git metadata (software/nn/.git is an empty directory; the repository
root is its parent), so every run's session_begin event said git_commit "unknown". A commit
hash alone would not have helped either: a deploy ships the WORKING tree, uncommitted edits
included.

What is recorded. gridunesp_deploy.sh runs `source_revision.py compute software/nn` before it
syncs, and ships the output as SOURCE_REVISION next to the sources:

    commit=414e2422            HEAD when the tree was shipped
    dirty=yes                  uncommitted changes under software/nn ("yes" / "no")
    tree_sha256=3fa9c01b2d4e.. hash of the CONTENT of every source file
    files=3159                 how many files went into it

01_meeting01_run_loso.sh exports a one-line label of it as MEETING01_GIT_COMMIT, which the
binary writes into every fold's events file (session_begin) and the monitor prints:

    414e2422+dirty/3fa9c01b2d4e     commit, a "+dirty" mark, first 12 hex digits of the tree hash

Commit and tree hash answer different questions:

                                        commit         tree_sha256
    answers                             which history  which bytes
    sees an uncommitted edit            no             yes
    equal after an edit and its undo    --             yes

"Source file" means what git tracks plus what it would track (untracked, not ignored):
`git ls-files --cached --others --exclude-standard`. Local noise that git ignores -- the code
index, logs, compile_commands.json -- therefore never changes the hash. Each file contributes
one line, and the lines, sorted by path, are hashed:

    F <path> <sha256 of the content>    a regular file
    L <path> <link target>              a symlink
    D <path>                            tracked, but deleted from the working tree
    G <path>                            a directory (a submodule)

Failure modes. Loud: not inside a git checkout, or no commit yet (RevisionError). Not covered,
and silent: a file git ignores that a run nevertheless needs. None exists today (the production
profile is tracked; the credentials file .env is neither tracked nor shipped); a new one would
have to be added to git, or it is simply not part of the identity.

Usage:
    source_revision.py compute ROOT      print the SOURCE_REVISION block for the tree at ROOT
    source_revision.py --self-test       known-answer checks, also of _provenance.sh (CI)
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

#: The deployed copy of the block. Never part of its own hash.
REVISION_FILE = "SOURCE_REVISION"


class RevisionError(RuntimeError):
    """The revision cannot be determined; the message names cause and remedy."""


def _git(root: pathlib.Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True).stdout
    except OSError as e:
        raise RevisionError(f"cannot run git ({e}): install git; the tree hash is defined "
                            "over the files git tracks") from None
    except subprocess.CalledProcessError as e:
        raise RevisionError(
            f"`git {' '.join(args)}` failed in {root}: {e.stderr.decode().strip()}. The "
            "revision is defined over a git checkout with at least one commit; run this from "
            "the repository, not from a copy of the tree") from None


def source_files(root: pathlib.Path) -> list[str]:
    """Paths under root, relative to it, of every file git tracks or would track."""
    out = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", ".")
    return sorted({p for p in out.decode().split("\0") if p and p != REVISION_FILE})


def tree_digest(root: pathlib.Path) -> tuple[str, int]:
    """(sha256 hex, number of entries) of the source files' paths and content."""
    digest = hashlib.sha256()
    entries = source_files(root)
    for rel in entries:
        path = root / rel
        if path.is_symlink():
            line = f"L {rel} {os.readlink(path)}\n"
        elif path.is_file():
            line = f"F {rel} {hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        elif not path.exists():
            line = f"D {rel}\n"
        else:
            line = f"G {rel}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest(), len(entries)


def compute(root: pathlib.Path) -> str:
    """The SOURCE_REVISION block (key=value lines; `#` lines are comments)."""
    root = root.resolve()
    tree, n = tree_digest(root)
    commit = _git(root, "rev-parse", "--short=8", "HEAD").decode().strip()
    dirty = "yes" if _git(root, "status", "--porcelain", "--", ".").strip() else "no"
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return "\n".join([
        "# SOURCE_REVISION -- written by gridunesp_deploy.sh; see "
        "scripts/pipeline/meeting01/source_revision.py",
        f"commit={commit}",
        f"dirty={dirty}",
        f"tree_sha256={tree}",
        f"files={n}",
        f"created_utc={created}",
        "",
    ])


# ── self-test ────────────────────────────────────────────────────────────────

_PROV = pathlib.Path(__file__).resolve().parent / "_provenance.sh"
_GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.org",
                GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.org",
                GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")


def _run(cmd: list[str], cwd: pathlib.Path) -> None:
    subprocess.run(cmd, cwd=cwd, env=_GIT_ENV, check=True, capture_output=True)


def _field(block: str, key: str) -> str:
    return next(line.split("=", 1)[1] for line in block.splitlines()
                if line.startswith(key + "="))


def _bash(script: str, cwd: pathlib.Path, **env: str) -> subprocess.CompletedProcess:
    """Run `script` the way 01_meeting01_run_loso.sh would: strict mode, helpers sourced."""
    return subprocess.run(["bash", "-c", f'set -euo pipefail; source "{_PROV}"; {script}'],
                          cwd=cwd, env=dict(os.environ, **env), capture_output=True, text=True)


def _self_test_checks(tmp: pathlib.Path) -> list[str]:
    failures: list[str] = []

    def expect(ok: bool, what: str) -> None:
        if not ok:
            failures.append(what)

    repo = tmp / "repo"
    (repo / "b").mkdir(parents=True)
    (repo / "a.txt").write_text("alpha\n")
    (repo / "b" / "c.py").write_text("print('c')\n")
    (repo / ".gitignore").write_text("cache/\n*.log\n")
    _run(["git", "init", "-q"], repo)
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-q", "-m", "init"], repo)

    base = compute(repo)
    tree0 = _field(base, "tree_sha256")
    expect(_field(base, "dirty") == "no" and _field(base, "files") == "3",
           f"clean tree: dirty/files wrong\n{base}")
    expect(len(_field(base, "commit")) == 8, "commit is an 8-digit short hash")
    expect(_field(compute(repo), "tree_sha256") == tree0, "the hash is stable between runs")

    # The recipe, recomputed with other tools: a human can follow it.
    shell = ('for f in .gitignore a.txt b/c.py; do printf "F %s %s\\n" "$f" '
             '"$(sha256sum < "$f" | cut -d" " -f1)"; done | sha256sum | cut -d" " -f1')
    cross = subprocess.run(["bash", "-c", shell], cwd=repo, capture_output=True,
                           text=True).stdout.strip()
    expect(cross == tree0, f"recipe differs from the shell recomputation: {cross} vs {tree0}")

    # Ignored files are noise, not source.
    (repo / "cache").mkdir()
    (repo / "cache" / "x.bin").write_text("index\n")
    (repo / "run.log").write_text("log\n")
    ignored = compute(repo)
    expect(_field(ignored, "tree_sha256") == tree0 and _field(ignored, "files") == "3",
           "git-ignored files changed the hash")
    expect(_field(ignored, "dirty") == "no", "git-ignored files made the tree dirty")

    # An uncommitted edit is seen, and its undo restores the hash exactly.
    (repo / "a.txt").write_text("alpha, edited\n")
    edited = compute(repo)
    expect(_field(edited, "tree_sha256") != tree0 and _field(edited, "dirty") == "yes",
           "an uncommitted edit went unseen")
    (repo / "a.txt").write_text("alpha\n")
    expect(_field(compute(repo), "tree_sha256") == tree0
           and _field(compute(repo), "dirty") == "no", "undoing the edit did not restore the hash")

    # A new untracked file that is not ignored is source.
    (repo / "new.py").write_text("x = 1\n")
    untracked = compute(repo)
    expect(_field(untracked, "tree_sha256") != tree0 and _field(untracked, "files") == "4"
           and _field(untracked, "dirty") == "yes", "an untracked source file went unseen")
    (repo / "new.py").unlink()

    # The path is part of the identity: same bytes under another name is another tree.
    (repo / "a.txt").rename(repo / "d.txt")
    expect(_field(compute(repo), "tree_sha256") != tree0, "a rename went unseen")
    (repo / "d.txt").rename(repo / "a.txt")

    # A tracked file deleted from the working tree is recorded, not skipped.
    (repo / "b" / "c.py").unlink()
    deleted = compute(repo)
    expect(_field(deleted, "tree_sha256") != tree0 and _field(deleted, "dirty") == "yes",
           "a deleted tracked file went unseen")
    (repo / "b" / "c.py").write_text("print('c')\n")

    # The deployed block is never part of its own hash.
    (repo / REVISION_FILE).write_text(base)
    expect(_field(compute(repo), "tree_sha256") == tree0, "SOURCE_REVISION hashed itself")
    (repo / REVISION_FILE).unlink()

    # Outside a checkout the revision is refused, not guessed.
    bare = tmp / "not_a_repo"
    bare.mkdir()
    try:
        compute(bare)
        failures.append("a directory outside any git checkout was accepted")
    except RevisionError as e:
        expect("git" in str(e), f"the refusal does not say why: {e}")

    # ── _provenance.sh, as the run script uses it ──
    label_of = lambda block: _bash(  # noqa: E731
        'provenance_label "$F"', repo, F=str(_write(tmp / "rev", block))).stdout
    clean = "commit=abc12345\ndirty=no\ntree_sha256=" + "ab" * 32 + "\n"
    expect(label_of(clean) == "abc12345/" + "ab" * 6, f"clean label: {label_of(clean)!r}")
    expect(label_of(clean.replace("dirty=no", "dirty=yes")) == "abc12345+dirty/" + "ab" * 6,
           "dirty label")
    cut = _bash('provenance_label "$F"', repo, F=str(_write(tmp / "cut", "commit=abc12345\n")))
    expect(cut.returncode != 0 and "gridunesp_deploy.sh" in cut.stderr,
           f"a truncated block was accepted: {cut.stdout!r} {cut.stderr!r}")

    # In a git checkout (no SOURCE_REVISION in the tree) the label is computed live.
    live = _bash(f'provenance_resolve "{repo}" "{sys.executable}"; echo "$MEETING01_GIT_COMMIT"',
                 repo)
    want = f"{_field(base, 'commit')}/{tree0[:12]}"
    expect(live.returncode == 0 and live.stdout.strip() == want,
           f"live label {live.stdout!r} {live.stderr!r}, wanted {want!r}")

    # In a deployed tree the binary must have been built from the sources now on disk.
    deployed = tmp / "deployed"
    (deployed / "out" / "build" / "p").mkdir(parents=True)
    (deployed / REVISION_FILE).write_text(base)
    check = f'provenance_check_binary_is_current "{deployed}" p'
    missing = _bash(check, deployed)
    expect(missing.returncode != 0 and "no build stamp" in missing.stderr
           and "REFUSED" in missing.stderr, f"missing stamp accepted: {missing.stderr!r}")
    (deployed / "out" / "build" / "p" / "SOURCE_REVISION.built").write_text(
        base.replace(tree0, "cd" * 32))
    stale = _bash(check, deployed)
    expect(stale.returncode != 0 and tree0[:12] in stale.stderr and "cdcdcdcdcdcd" in stale.stderr,
           f"a stale stamp was accepted or not explained: {stale.stderr!r}")
    expect(_bash(f'provenance_mark_built "{deployed}" p; {check}', deployed).returncode == 0,
           "a freshly stamped build was refused")
    expect(_bash(f'provenance_check_binary_is_current "{repo}" p', repo).returncode == 0,
           "a plain checkout was refused for lacking a stamp")

    logged = _bash(f'MEETING01_GIT_COMMIT=abc12345/xyz; provenance_log_start "{deployed}" 1',
                   deployed)
    text = (deployed / "results" / "meeting01" / "source_revisions.log").read_text()
    expect(logged.returncode == 0 and "resume=1" in text and "revision=abc12345/xyz" in text,
           f"start was not logged: {text!r}")
    return failures


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text)
    return path


def self_test() -> int:
    for tool in ("git", "bash", "sha256sum"):
        if shutil.which(tool) is None:
            print(f"[self-test] needs `{tool}` on PATH", file=sys.stderr)
            return 2
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="source_revision_selftest_"))
    try:
        failures = _self_test_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in failures:
        print(f"[self-test] FAIL: {f}", file=sys.stderr)
    if failures:
        return 1
    print("[self-test] all source-revision checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--self-test"]:
        return self_test()
    if len(args) == 2 and args[0] == "compute":
        try:
            sys.stdout.write(compute(pathlib.Path(args[1])))
        except RevisionError as e:
            print(f"source_revision.py: {e}", file=sys.stderr)
            return 1
        return 0
    print(__doc__.split("Usage:")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
