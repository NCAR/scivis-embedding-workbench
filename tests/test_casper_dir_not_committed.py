"""Guardrail: personal credentials in scripts/casper/ must never reach git.

The framework files (launcher, template, README, .env.example) are tracked
so the tooling is recoverable and shareable. But:

1. `casper-marimo.env` (the filled-in copy with PBS account code and NCAR
   username) must stay gitignored.
2. `casper-marimo.env.example` (which IS tracked) must contain only
   placeholder values like `<YOUR_PROJECT_CODE>` for sensitive variables —
   never real account codes or usernames. This stops someone from copying
   their real `.env` over the example and committing it.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CASPER_DIR = REPO_ROOT / "scripts" / "casper"
ENV_FILE_REL = "scripts/casper/casper-marimo.env"
ENV_EXAMPLE = CASPER_DIR / "casper-marimo.env.example"
GITIGNORE = REPO_ROOT / ".gitignore"

# Variables whose values must be placeholders in the tracked .env.example.
SENSITIVE_VARS = ("PBS_ACCOUNT", "REMOTE_PROJECT_DIR", "GATEWAY_USER")


def _git_available() -> bool:
    return shutil.which("git") is not None and (REPO_ROOT / ".git").exists()


@pytest.mark.skipif(not _git_available(), reason="not in a git checkout")
def test_env_file_not_tracked():
    """The filled-in casper-marimo.env must never be tracked by git."""
    result = subprocess.run(
        ["git", "ls-files", ENV_FILE_REL],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert not tracked, (
        f"{ENV_FILE_REL} must never be committed — it holds your "
        f"PBS account code and NCAR username. git ls-files reported:\n"
        + "\n".join(f"  {p}" for p in tracked)
    )


@pytest.mark.skipif(not _git_available(), reason="not in a git checkout")
def test_env_file_is_gitignored():
    """Belt-and-suspenders: confirm the .gitignore entry exists."""
    assert GITIGNORE.exists(), ".gitignore not found at repo root"
    text = GITIGNORE.read_text()
    assert ENV_FILE_REL in text or "casper-marimo.env" in text, (
        f".gitignore must contain an entry that matches {ENV_FILE_REL}"
    )


@pytest.mark.skipif(not _git_available(), reason="not in a git checkout")
def test_env_file_is_check_ignored():
    """`git check-ignore` must mark the env file as ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", ENV_FILE_REL],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{ENV_FILE_REL} is NOT ignored by git — the gitignore rule is "
        f"missing or wrong.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_env_example_uses_only_placeholders():
    """Sensitive vars in the tracked .env.example must be `<PLACEHOLDER>` values.

    This catches the common mistake of editing the example file with real
    values (instead of editing the gitignored .env) and committing.
    """
    assert ENV_EXAMPLE.exists(), f"{ENV_EXAMPLE} not found"
    text = ENV_EXAMPLE.read_text()
    for var in SENSITIVE_VARS:
        match = re.search(rf"(?m)^\s*{re.escape(var)}\s*=\s*(.+?)\s*$", text)
        assert match, (
            f"{var} assignment missing from {ENV_EXAMPLE.name}"
        )
        value = match.group(1).strip().strip('"').strip("'")
        assert value.startswith("<") and value.endswith(">"), (
            f"{var} in {ENV_EXAMPLE.name} must be a placeholder like "
            f"<YOUR_VALUE>, got: {value!r}. If you have a real value, put "
            f"it in casper-marimo.env (gitignored), not the example."
        )


def test_no_known_credential_strings_in_tracked_casper_files():
    """Hard fail on a small denylist of credential-shaped strings in any
    tracked file under scripts/casper/.

    The denylist intentionally avoids matching the user's own values
    directly — it looks for *shapes* (e.g. an `NVST` prefix is the NCAR
    project-code convention; a `/glade/work/<word>/` path includes a real
    NCAR username). Add to this list if new sensitive shapes appear.
    """
    if not _git_available():
        pytest.skip("not in a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "scripts/casper/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    # Filename → list of regex denylist patterns it must NOT contain.
    denylist = [
        # NCAR project codes start with letters like NVST, UNSL, etc. and
        # end in digits. The example must keep the `<...>` placeholder form.
        re.compile(r"^\s*PBS_ACCOUNT\s*=\s*[A-Z]{2,}\d+\s*$", re.MULTILINE),
        # GLADE work paths embed the user's NCAR username.
        re.compile(r"/glade/work/[a-z][a-z0-9_-]+/"),
        # An assignment that fills GATEWAY_USER with a bare alnum username.
        re.compile(r"^\s*GATEWAY_USER\s*=\s*[a-z][a-z0-9_-]+\s*$", re.MULTILINE),
    ]

    failures = []
    for rel in tracked:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        for pattern in denylist:
            if pattern.search(content):
                failures.append(f"{rel}: matched {pattern.pattern!r}")

    assert not failures, (
        "Personal credentials appear to have leaked into tracked files:\n"
        + "\n".join(f"  {f}" for f in failures)
        + "\nMove these values to scripts/casper/casper-marimo.env (gitignored)."
    )
