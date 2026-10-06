from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings

ROOT = Path(__file__).resolve().parents[1]
_HISTORY_PATTERN = (
    r"sk-[A-Za-z0-9_-]{16,}"
    r"|gh[opurs]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|xox[baprs]-[0-9A-Za-z-]{20,}"
    r"|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan the repository without printing secrets")
    parser.add_argument("--history", action="store_true", help="also scan every reachable commit")
    args = parser.parse_args()

    findings = _scan_candidate_files()
    if args.history:
        findings.extend(_scan_history())
    if findings:
        for finding in findings:
            print(f"potential secret: {finding}")
        raise SystemExit(1)
    print("PASS: no potential secrets found in candidate files or selected Git history")


def _scan_candidate_files() -> list[str]:
    tracked = _git(
        "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    ).split("\0")
    collection = SecretsCollection()
    with default_settings():
        for relative in tracked:
            path = ROOT / relative
            if relative and path.is_file():
                collection.scan_file(str(path))
    findings: list[str] = []
    for filename, secrets in collection.json().items():
        path = Path(filename)
        try:
            displayed_path = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            displayed_path = path.name
        for secret in secrets:
            findings.append(
                f"working tree {displayed_path}:{secret['line_number']} "
                f"({secret['type']})"
            )
    return findings


def _scan_history() -> list[str]:
    commits = _git("rev-list", "--all").splitlines()
    completed = subprocess.run(
        ["git", "grep", "-I", "-l", "-E", _HISTORY_PATTERN, *commits, "--"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError("git grep failed while scanning repository history")
    findings: list[str] = []
    for match in completed.stdout.splitlines():
        commit, _, name = match.partition(":")
        findings.append(f"history {commit[:12]}:{name}")
    return findings


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout


if __name__ == "__main__":
    main()
