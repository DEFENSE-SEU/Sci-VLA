"""TimestampColumnValidator — ensure per-frame timestamp info exists.

Most downstream tools (e.g. video-sync, episode trimming, action-interp
probes) rely on a timestamp column in the per-episode parquet. LabVLA's
training pipeline does NOT strictly require it (delta_timestamps are
derived from fps), so this is a MED-severity heads-up rather than a
blocker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import Validator, ValidationIssue


class TimestampColumnValidator(Validator):
    name = "timestamp_column"
    description = "Verify each repo's parquet schema exposes a timestamp column."

    def validate(self, repos: Iterable[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            return [
                ValidationIssue(
                    # An explicitly-requested check whose dependency is missing
                    # must not exit 0 — CI would read "validated" when nothing
                    # was inspected.
                    severity="HIGH",
                    repo="<validator>",
                    message=(
                        "pyarrow is not installed; timestamp_column validator "
                        f"skipped ({e}). Install pyarrow to inspect parquet metadata."
                    ),
                    context={},
                )
            ]

        for repo in repos:
            repo_path = Path(repo)
            if (repo_path / "meta" / "vqa_manifest.json").exists():
                # VQA-only repos are JSON/JSONL+image datasets, not LeRobot
                # parquet repos. Preflight validates their manifest separately.
                continue
            _all_parquets = sorted(repo_path.rglob("data/**/*.parquet"))
            if not _all_parquets:
                issues.append(ValidationIssue(
                    severity="HIGH",
                    repo=str(repo_path),
                    message="no parquet file found under data/",
                    context={},
                ))
                continue
            # One file proves nothing about shard-level schema drift — sample a
            # deterministic spread (first, last, and evenly spaced interior
            # files, up to 8).
            _n = len(_all_parquets)
            _idx = sorted({0, _n - 1, *(int(_n * k / 7) for k in range(1, 7))})
            _samples = [_all_parquets[i] for i in _idx if 0 <= i < _n]
            _missing_ts = []
            cols = []
            _read_fail = False
            for parquet in _samples:
                try:
                    cols = pq.read_metadata(parquet).schema.names
                except Exception as e:
                    issues.append(ValidationIssue(
                        severity="HIGH",
                        repo=str(repo_path),
                        message=f"failed to read parquet metadata: {e}",
                        context={"path": str(parquet)},
                    ))
                    _read_fail = True
                    break
                if not any("timestamp" in c.lower() or "time" in c.lower() for c in cols):
                    _missing_ts.append(str(parquet))
            if _read_fail:
                continue
            if _missing_ts:
                issues.append(ValidationIssue(
                    severity="MED",
                    repo=str(repo_path),
                    message=(
                        f"{len(_missing_ts)}/{len(_samples)} sampled shard(s) "
                        f"lack a timestamp/time column (first: {_missing_ts[0]}; M108)"
                    ),
                    context={"missing": _missing_ts[:8],
                             "sampled": len(_samples), "total": _n},
                ))
                continue
            if False:
                issues.append(ValidationIssue(
                    severity="MED",
                    repo=str(repo_path),
                    message=(
                        f"no timestamp/time column found (columns: {cols}). "
                        "Training works via fps-derived delta_timestamps, but "
                        "downstream tooling (video sync, episode trim) may break."
                    ),
                    context={"columns": cols, "sample_parquet": str(parquet)},
                ))
        return issues
