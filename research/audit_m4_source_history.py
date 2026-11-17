"""Append-only audit of registered M3 source hashes against reachable Git history."""

import hashlib
import json
import subprocess
from pathlib import Path

from metabolism.m2_experiments import save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m4-source-history-audit"
REGISTRATIONS = (
    "results/m3/acceptance.json",
    "results/m3-continuation/acceptance.json",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def matching_commit(path, expected):
    commits = git("log", "--all", "--format=%H", "--", path).decode().splitlines()
    for commit in commits:
        try:
            content = git("show", f"{commit}:{path}")
        except subprocess.CalledProcessError:
            continue
        if sha256(content) == expected:
            meta = git("show", "-s", "--format=%cI%x09%s", commit).decode().rstrip("\n")
            date, subject = meta.split("\t", 1)
            return dict(commit=commit, committed_at=date, subject=subject)
    return None


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing audit: {OUT}")
    OUT.mkdir(parents=True)
    entries = []
    for registration_path in REGISTRATIONS:
        document = json.loads((ROOT / registration_path).read_text())
        for path, expected in document["evidence_sha256"].items():
            if (len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)
                    or Path(path).is_absolute() or ".." in Path(path).parts):
                raise ValueError(f"Invalid registered path or digest: {path}")
            file_path = ROOT / path
            current = sha256(file_path.read_bytes()) if file_path.is_file() else None
            row = dict(registration=registration_path, path=path,
                       registered_sha256=expected, current_sha256=current,
                       current_match=current == expected)
            if current != expected:
                row["historical_match"] = matching_commit(path, expected)
            entries.append(row)
    mismatches = [row for row in entries if not row["current_match"]]
    historical_resolved = all(row.get("historical_match") for row in mismatches)
    result = dict(
        scope="Registered M3 source provenance only; does not reproduce old runs or alter admission",
        registrations=list(REGISTRATIONS),
        registered_files=len(entries),
        current_worktree_matches=len(entries) - len(mismatches),
        current_worktree_mismatches=len(mismatches),
        all_mismatched_sources_found_in_reachable_git_history=historical_resolved,
        historical_sources=mismatches,
        interpretation=(
            "All registered source versions were found in reachable Git history; current-source "
            "mismatch remains and historical execution is not reproduced."
            if historical_resolved else
            "At least one mismatched registered source version was not found in reachable Git history."
        ),
    )
    save(OUT / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
