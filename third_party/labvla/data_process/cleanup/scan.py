"""Run all detectors against a v2.1 dataset and emit a JSON report.

Usage:

  python -m data_process scan \\
      --root /path/to/pretrain_data/dataset \\
      --out  /tmp/cleanup.json

  # or select specific detectors:
  python -m data_process scan \\
      --root ... --out ... \\
      --detectors video_concat_hazards,stats_orphans

The output JSON has shape:
  {
    "root": "...",
    "total_episodes": N,
    "bad_episodes": [ep_idx, ...],           # union across detectors
    "per_episode_reasons": {"45": [{...}], ...},
    "orphan_stats_episodes": [ep_idx, ...],  # for stats_orphans detector
    "detector_summaries": {"video_stubs": {"n_bad": 2, "n_reasons": 2}, ...}
  }

The `python -m data_process clean` step consumes this report directly.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import sys
from pathlib import Path

# Direct-file execution support (python data_process/.../<file>.py): repo root
# on sys.path before the data_process/src imports below.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_process.detectors import ALL_DETECTORS, Detector


def _select_detectors(name_csv: str | None) -> list[type[Detector]]:
    if not name_csv:
        return ALL_DETECTORS
    wanted = {x.strip() for x in name_csv.split(",") if x.strip()}
    if not wanted:
        # A tokenless --detectors (" , , ") strips to empty -> would select
        # zero detectors and silently pass. Reject it (mirrors validators/run.py).
        raise SystemExit(
            f"--detectors={name_csv!r} contains no detector names. "
            f"Available: {[D.name for D in ALL_DETECTORS]}"
        )
    out = [D for D in ALL_DETECTORS if D.name in wanted]
    missing = wanted - {D.name for D in out}
    if missing:
        raise SystemExit(
            f"Unknown detector(s): {missing}. "
            f"Available: {[D.name for D in ALL_DETECTORS]}"
        )
    return out


def _count_episodes(root: Path) -> int:
    eps_path = root / "meta" / "episodes.jsonl"
    if not eps_path.exists():
        return 0
    with open(eps_path) as f:
        return sum(1 for line in f if line.strip())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--detectors", type=str, default=None,
                    help="Comma-separated detector names; default = all.")
    ap.add_argument("--stub-threshold", type=int, default=3,
                    help="VideoStubDetector: flag ep if any camera mp4 has "
                         "≤ this many packets (default 3).")
    ap.add_argument("--action-cap", type=int, default=32,
                    help="ActionDimCapDetector: max allowed action_dim (default 32).")
    ap.add_argument("--state-cap", type=int, default=32,
                    help="ActionDimCapDetector: max allowed state_dim (default 32).")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not (args.root / "meta" / "info.json").exists():
        raise SystemExit(f"{args.root} doesn't look like a LeRobot v2.1 dataset (no meta/info.json)")

    # scan/clean only support the v2.1 per-episode layout. A v3.0 root would
    # pass the info.json existence check, every detector would then find no
    # episodes.jsonl / per-episode mp4s and return empty → the report would
    # show "0 bad episodes", reading as a CLEAN dataset rather than an
    # unsupported one. Fail loud with guidance instead.
    from data_process.meta_reader.registry import detect_codebase_version
    _version = detect_codebase_version(args.root)
    if not _version.startswith("v2"):
        raise SystemExit(
            f"{args.root} has codebase_version={_version!r}: `data_process "
            f"scan`/`clean` only support LeRobot v2.x (per-episode parquet+mp4 "
            f"layout). For v3.0 datasets, integrity is handled by the merge "
            f"tooling (data_process/merge_v3/) and `python -m data_process "
            f"validate|preflight`; re-cleaning requires rebuilding the merge."
        )

    det_classes = _select_detectors(args.detectors)
    logging.info("Running detectors: %s", [D.name for D in det_classes])

    # Bind the report to its source dataset + version the contract so
    # `clean --report` cannot consume a report from another dataset / era.
    _info_p = args.root / "meta" / "info.json"
    _info_stat = _info_p.stat() if _info_p.exists() else None
    report = {
        "report_version": 2,
        "root": str(args.root),
        "source_binding": {
            "root_resolved": str(args.root.resolve()),
            "info_json_size": int(_info_stat.st_size) if _info_stat else None,
            "info_json_mtime_ns": int(_info_stat.st_mtime_ns) if _info_stat else None,
        },
        "total_episodes": _count_episodes(args.root),
        "bad_episodes": [],
        "per_episode_reasons": {},
        "orphan_stats_episodes": [],
        # Top-level blocking signal. Some detectors (e.g. action_dim_cap) decide
        # the WHOLE dataset is incompatible and only set
        # ``extra["dataset_incompatible"]``, leaving ``bad_episodes`` empty (no
        # per-episode fix). Hoisting the flag here lets ``apply`` refuse by
        # default; ``dataset_incompatible_reasons`` records which tripped it.
        "dataset_incompatible": False,
        "dataset_incompatible_reasons": [],
        "detector_summaries": {},
    }
    bad_union: set[int] = set()

    for D in det_classes:
        logging.info("--- %s ---", D.name)
        opts = {}
        if D.name == "video_stubs":
            opts["threshold"] = args.stub_threshold
        if D.name == "action_dim_cap":
            opts["action_cap"] = args.action_cap
            opts["state_cap"] = args.state_cap
        if D.name in ("video_stubs", "video_concat_hazards", "video_decode", "missing_videos"):
            opts["workers"] = args.workers
        det = D(**opts)
        res = det.scan(args.root)
        report["detector_summaries"][D.name] = {
            "n_bad": len(res.bad_episodes),
            "n_reasons": sum(len(v) for v in res.reasons.values()),
            "n_warnings": sum(len(v) for v in res.warnings.values()),
            "extra": res.extra,
        }
        logging.info("  bad=%d reasons=%d warnings=%d",
                     len(res.bad_episodes),
                     sum(len(v) for v in res.reasons.values()),
                     sum(len(v) for v in res.warnings.values()))
        bad_union |= res.bad_episodes
        for ep, rs in res.reasons.items():
            report["per_episode_reasons"].setdefault(str(ep), []).extend([asdict(r) for r in rs])
        if D.name == "stats_orphans":
            report["orphan_stats_episodes"].extend(res.extra.get("orphan_stats_eps", []))
        # Hoist any detector's dataset-level incompatibility flag to the top of
        # the report so the cleaner can block on it.
        if res.extra.get("dataset_incompatible"):
            report["dataset_incompatible"] = True
            report["dataset_incompatible_reasons"].append({
                "detector": D.name,
                "violations": res.extra.get("violations", []),
                "extra": {k: v for k, v in res.extra.items()
                          if k not in ("dataset_incompatible", "violations")},
            })
            logging.warning(
                "  %s flagged DATASET-INCOMPATIBLE: %s",
                D.name, res.extra.get("violations", []),
            )

    report["bad_episodes"] = sorted(bad_union)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: if scan is killed mid-dump, callers must not see partial JSON.
    import os as _os
    import tempfile as _tempfile
    _fd, _tmp = _tempfile.mkstemp(
        prefix=args.out.name + ".", suffix=".tmp", dir=str(args.out.parent)
    )
    try:
        with _os.fdopen(_fd, "w") as f:
            json.dump(report, f, indent=2)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(_tmp, str(args.out))
    except Exception:
        try:
            _os.unlink(_tmp)
        except FileNotFoundError:
            pass
        raise
    logging.info("Wrote report: %s", args.out)
    logging.info("Summary: %d bad episodes union, %d stats orphans",
                 len(bad_union), len(report["orphan_stats_episodes"]))


if __name__ == "__main__":
    main()
