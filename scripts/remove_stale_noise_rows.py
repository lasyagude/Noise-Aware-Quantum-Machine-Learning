"""One-off utility: remove result rows computed under noise probabilities
that have since been corrected, so resume logic in the experiment runners
cannot skip them as "already completed".

Background: `config.NOISE_PROFILES["low"]`/`["medium"]` were corrected on
2026-10-11 to match the assignment's noise-level guideline table exactly.
Any row evaluated under the `"low"` or `"medium"` noise condition before
that date used the old probabilities. `"ideal"` rows are unaffected (no
noise model is attached) and are left alone.

Usage::

    python scripts/remove_stale_noise_rows.py

Then rerun the normal (non --force) commands -- see PROGRESS.md -- and they
will resume-skip the still-valid ideal rows and regenerate only what was
removed here.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (file, set of stale experiment_ids to drop)
TARGETS: list[tuple[Path, set[str]]] = [
    (PROJECT_ROOT / "results" / "vqc_experiment_results.jsonl", {"E5", "E6", "E8"}),
    (
        PROJECT_ROOT / "results" / "extension_breast_cancer_results.jsonl",
        {"BC-E5", "BC-E6", "BC-E8"},
    ),
]

# Wholly derived from the file above; simplest to delete and let the normal
# command regenerate it from scratch (a few seconds, no retraining).
DERIVED_TO_DELETE: list[Path] = [
    PROJECT_ROOT / "results" / "extension_pooled_analysis.json",
]


def _filter_jsonl(path: Path, stale_ids: set[str]) -> None:
    if not path.exists():
        print(f"skip (not found): {path}")
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in rows if r.get("experiment_id") not in stale_ids]
    removed = len(rows) - len(kept)
    path.write_text("\n".join(json.dumps(r) for r in kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"{path.relative_to(PROJECT_ROOT)}: removed {removed} stale row(s), kept {len(kept)}")


def main() -> None:
    for path, stale_ids in TARGETS:
        _filter_jsonl(path, stale_ids)
    for path in DERIVED_TO_DELETE:
        if path.exists():
            path.unlink()
            print(f"deleted (derived, will regenerate): {path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"skip (not found): {path}")


if __name__ == "__main__":
    main()
