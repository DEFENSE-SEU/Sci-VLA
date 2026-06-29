"""CLI entry for `python -m data_process validate`.

Usage:
    python -m data_process validate --repos <repo1> [<repo2> ...]
                                    [--validators fps_consistency,timestamp_column,frame_count]
                                    [--sample-size 20]

Exit code:
    0 — no issues, or only LOW/MED severity
    1 — at least one HIGH or CRIT severity issue
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sys
from pathlib import Path

# Direct-file execution support (python data_process/.../<file>.py): repo root
# on sys.path before the data_process/src imports below.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_process.validators import ALL_VALIDATORS, ValidationIssue


_SEVERITY_ORDER = {"LOW": 0, "MED": 1, "HIGH": 2, "CRIT": 3}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="data_process validate",
        description="Run dataset-level validators across one or more LeRobot repos.",
    )
    p.add_argument("--repos", nargs="+", required=True,
                   help="Absolute or relative paths to LeRobot repo roots.")
    p.add_argument("--validators", default=None,
                   help="Comma-separated validator names. Default: run all.")
    p.add_argument("--json", type=str, default=None,
                   help="Also write the structured ValidationIssue list "
                        "(incl. context payloads) to this JSON file.")
    p.add_argument("--sample-size", type=int, default=20,
                   help="Sample size for FrameCountValidator (default: 20).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repos = [Path(r) for r in args.repos]

    name_filter = None
    if args.validators:
        name_filter = {s.strip() for s in args.validators.split(",") if s.strip()}
        # A tokenless value (" , , ") strips to an empty set, which the
        # unknown-name guard below cannot catch -> zero validators run + "OK"
        # exit 0 (a fake clean pass). Reject it.
        if not name_filter:
            raise SystemExit(
                f"--validators={args.validators!r} contains no validator names. "
                f"Available: {sorted(V.name for V in ALL_VALIDATORS)}"
            )
        # Reject unknown validator names: a typo would match nothing, run zero
        # validators, and print "OK" + exit 0 — faking a clean pass. Mirrors the
        # detector-side check in cleanup/scan.py.
        known = {V.name for V in ALL_VALIDATORS}
        unknown = name_filter - known
        if unknown:
            raise SystemExit(
                f"Unknown validator(s): {sorted(unknown)}. "
                f"Available: {sorted(known)}"
            )

    all_issues: list[ValidationIssue] = []
    for V in ALL_VALIDATORS:
        if name_filter is not None and V.name not in name_filter:
            continue
        # Pass kwargs that a validator accepts; graceful for those that don't.
        try:
            instance = V(sample_size=args.sample_size)
        except TypeError:
            instance = V()
        try:
            issues = instance.validate(repos)
        except Exception as exc:  # noqa: BLE001 — crash -> structured issue
            issues = [ValidationIssue(
                severity="HIGH",
                repo="<crash>",
                message=f"{V.name} validator crashed: {type(exc).__name__}: {exc}",
                context={"validator": V.name},
            )]
        all_issues.extend(issues)

    if not all_issues:
        print("[validate] OK — no issues found across:", ", ".join(str(r) for r in repos))
        sys.exit(0)

    # Sort by severity descending for human readability
    all_issues.sort(key=lambda i: -_SEVERITY_ORDER.get(i.severity, -1))
    if getattr(args, "json", None):
        import dataclasses
        with open(args.json, "w") as _jf:
            json.dump([dataclasses.asdict(i) for i in all_issues], _jf, indent=2)
        print(f"[validate] structured issues written to {args.json}")
    for issue in all_issues:
        print(f"[{issue.severity}] {issue.repo}: {issue.message}")

    has_blocker = any(i.severity in ("HIGH", "CRIT") for i in all_issues)
    sys.exit(1 if has_blocker else 0)


if __name__ == "__main__":
    main()
