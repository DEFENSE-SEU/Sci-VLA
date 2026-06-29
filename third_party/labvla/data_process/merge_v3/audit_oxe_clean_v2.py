"""Post-merge GLOBAL spot check for oxe-auge_clean_v2.

Run this AFTER `build_oxe_auge_clean.py` produces oxe-auge_clean_v2/ for a
coarse, repo-wide sanity pass:
  1. meta/info.json + meta/stats.json presence.
  2. GLOBAL gripper dim 7 distribution over sampled data shards:
     range ⊆ [0, 1], direction 1=open, binary-vs-continuous composition.
  3. labvla_manifest schema_id / robot_type.

Scope caveat: this is a GLOBAL spot check, NOT per-subset attribution. The
merged repo loses original sub-repo identity at the data/chunk level, so this
script CANNOT verify "all 12 INCLUDED subsets present / 4 EXCLUDED absent" or
per-subset gripper direction. A PASS here does NOT certify subset-level merge
integrity — a single missing/mis-mixed/direction-flipped subset can be masked by
the global distribution. For subset-level guarantees, audit the per-subset
whitelist at build time (see is_included_subset / canonicalize_gripper).

Usage:
    python -m data_process.merge_v3.audit_oxe_clean_v2  # from the repo root \\
        --root /path/to/pretrain_data/oxe-auge_clean_v2

Output: human-readable report to stdout, exit 0 on full pass, 1 on any finding
(including any sampled shard that failed to read).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

def _sample_data_shards(root: Path, max_shards: int = 200) -> list[Path]:
    """Sample data parquet shards from the merged repo for spot inspection."""
    data_root = root / "data"
    if not data_root.exists():
        return []
    shards = sorted(data_root.rglob("file-*.parquet"))
    if len(shards) <= max_shards:
        return shards
    # Stratify: take evenly-spaced shards
    step = max(1, len(shards) // max_shards)
    return shards[::step][:max_shards]


def _read_grip_from_shard(
    path: Path, max_rows: int = 500
) -> tuple[np.ndarray | None, str | None]:
    """Return ``(grip_values, failure_reason)``.

    Distinguishes a genuine read failure (unreadable parquet, missing column,
    wrong width) — which the caller treats as an audit FAIL — from a legitimately
    empty shard (``grip=None, reason=None``), which is benign. Collapsing both to
    a bare ``None`` would silently drop failures and let the audit PASS on
    partial evidence.
    """
    try:
        t = pq.read_table(path, columns=["observation.joints"])
    except Exception as exc:
        return None, f"read_error: {type(exc).__name__}: {exc}"
    col = t["observation.joints"].to_numpy()
    if len(col) == 0:
        return None, None  # empty shard — benign, not a failure
    n = min(max_rows, len(col))
    try:
        arr = np.stack([col[i] for i in range(n)])
    except Exception as exc:
        return None, f"stack_error: {type(exc).__name__}: {exc}"
    if arr.ndim != 2 or arr.shape[-1] != 8:
        return None, f"bad_shape: {tuple(arr.shape)} (expected (*, 8))"
    return arr[:, 7].astype(np.float32), None


def _read_state_and_joints(path: Path, max_rows: int = 300):
    """Return (state (n,ws), joints (n,8), failure_reason).

    Unlike `_read_grip_from_shard`, the state column has NO width expectation —
    measuring its real width distribution is the whole point (states of width
    {2, 7, 15, 24} occur across the merge while the schema claims canonical 8)."""
    try:
        t = pq.read_table(path, columns=["observation.state", "observation.joints"])
    except Exception as exc:
        return None, None, f"read_error: {type(exc).__name__}: {exc}"
    if len(t) == 0:
        return None, None, None
    n = min(max_rows, len(t))
    try:
        st = np.stack([t["observation.state"][i].as_py() for i in range(n)]).astype(np.float64)
        jt = np.stack([t["observation.joints"][i].as_py() for i in range(n)]).astype(np.float64)
    except Exception as exc:
        return None, None, f"stack_error: {type(exc).__name__}: {exc}"
    return st, jt, None


def _resolve_registered_oxe_schema():
    """Best-effort: resolve the REGISTERED oxe schema (not the on-disk
    manifest) so the verdict tracks what training actually binds via
    --dataset_schema overrides. Returns (schema, error_str)."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import importlib
        importlib.import_module("schemas.oxe_auge")
        from src.schema.registry import resolve
        return resolve("oxe_auge"), None
    except Exception as exc:  # pragma: no cover - env-dependent
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(prog="audit_oxe_clean_v2")
    ap.add_argument("--root", required=True, type=Path,
                    help="Path to oxe-auge_clean_v2 directory.")
    ap.add_argument("--max-shards", type=int, default=300,
                    help="Max data shards to sample for distribution probe.")
    args = ap.parse_args()

    root = args.root
    failed = False
    print(f"=== oxe-auge_clean_v2 post-merge audit: {root} ===\n")

    # 1. Top-level metadata sanity
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    print(f"[1/5] meta/info.json + meta/stats.json existence")
    if not info_path.exists():
        print(f"  ❌ missing {info_path}")
        failed = True
    else:
        info = json.load(open(info_path))
        print(f"  ✓ info.json: total_eps={info.get('total_episodes')}, "
              f"total_frames={info.get('total_frames')}, "
              f"codebase_version={info.get('codebase_version')}")
    if not stats_path.exists():
        print(f"  ⚠ stats.json missing — won't be a blocker (may be regenerated "
              f"separately via `python -m data_process stats`)")
    print()

    # 2. Sample data shards & probe gripper distribution
    print(f"[2/5] data shard distribution probe (up to {args.max_shards} shards)")
    shards = _sample_data_shards(root, max_shards=args.max_shards)
    if not shards:
        print(f"  ❌ no data shards under {root / 'data'}")
        return 1
    print(f"  found {len(shards)} shards (sampled)")
    grips_all = []
    failed_shards: list[tuple[Path, str]] = []
    for s in shards:
        g, reason = _read_grip_from_shard(s)
        if reason is not None:
            failed_shards.append((s, reason))
        elif g is not None:
            grips_all.append(g)
    # Surface any shard that failed to read and FAIL on partial evidence: a
    # corrupt/missing-column/shape-drifted shard must not be silently dropped
    # while the remaining good shards still PASS the distribution probe.
    if failed_shards:
        print(f"  ❌ {len(failed_shards)} sampled shard(s) failed to read:")
        for sp, reason in failed_shards[:20]:
            print(f"      - {sp}: {reason}")
        if len(failed_shards) > 20:
            print(f"      ... and {len(failed_shards) - 20} more")
        failed = True
    if not grips_all:
        print(f"  ❌ no gripper data extracted from any shard")
        return 1
    grip = np.concatenate(grips_all)
    print(f"  total samples: {len(grip):,}")
    print(f"  range: [{grip.min():.4f}, {grip.max():.4f}]")
    print(f"  mean: {grip.mean():.4f}, std: {grip.std():.4f}")
    in_range = bool((grip >= -1e-6).all() and (grip <= 1.0 + 1e-6).all())
    print(f"  in [0, 1]: {in_range}")
    if not in_range:
        n_out = int(((grip < -1e-6) | (grip > 1.0 + 1e-6)).sum())
        print(f"  ❌ {n_out:,} samples out of [0, 1]")
        failed = True
    else:
        print(f"  ✓ all in [0, 1]")
    print()

    # 3. Distribution split: how much is binary vs continuous
    n_zero = float(np.mean(np.abs(grip) < 0.05))
    n_one = float(np.mean(np.abs(grip - 1) < 0.05))
    n_mid = float(np.mean((grip > 0.05) & (grip < 0.95)))
    print(f"[3/5] cross-subset distribution composition")
    print(f"  near 0 (closed): {n_zero:.1%}")
    print(f"  near 1 (open):   {n_one:.1%}")
    print(f"  continuous mid:  {n_mid:.1%}")
    # Sanity: continuous fraction should reflect bridge + fractal in mix.
    # bridge ≈ 982k frames (40% mid by audit), fractal ≈ 3.78M frames (~30% mid)
    # total mid frames ≈ 982k*0.4 + 3.78M*0.3 ≈ 1.5M out of 13.3M = ~11%.
    # Allow generous range [3%, 30%].
    if not (0.03 <= n_mid <= 0.30):
        print(f"  ⚠ mid-range fraction {n_mid:.1%} outside expected [3%, 30%] — "
              f"bridge + fractal contribute continuous values; result depends "
              f"on shard sampling stratification.")
    else:
        print(f"  ✓ continuous fraction reasonable for bridge+fractal mix")
    print()

    # 4. labvla_manifest sanity
    manifest_path = root / "meta" / "labvla_manifest.json"
    print(f"[4/5] labvla_manifest.json")
    if not manifest_path.exists():
        print(f"  ❌ missing {manifest_path}")
        failed = True
    else:
        m = json.load(open(manifest_path))
        print(f"  ✓ schema_id={m.get('schema_id')}, robot_type={m.get('robot_type')}")
        if m.get("schema_id") != "oxe_auge_v1":
            print(f"  ⚠ unexpected schema_id (expected 'oxe_auge_v1')")
    print()

    # 5. observation.state column audit. Inspecting only `observation.joints`
    # would miss heterogeneous raw state widths {2, 7, 15, 24} behind a schema
    # claiming canonical 8. Verdict depends on the REGISTERED schema's state
    # binding:
    #   * schema reads `observation.state` directly → widths must be uniformly
    #     8, else FAIL (training would truncate/zero-pad semantically-wrong
    #     state and use it as the delta reference);
    #   * schema binds state via `virtual_state_sources` → the physical column
    #     is unused by training; report distribution as informational only.
    print(f"[5/5] observation.state column audit (R3-D5)")
    schema_obj, schema_err = _resolve_registered_oxe_schema()
    vmap = dict(getattr(schema_obj, "virtual_state_sources", {}) or {}) if schema_obj else {}
    direct_binding = (
        schema_obj is None  # unresolvable → conservative: assume direct
        or ("observation.state" in tuple(schema_obj.state_keys) and not vmap)
    )
    if schema_err:
        print(f"  ⚠ registered schema unresolvable ({schema_err}); "
              f"assuming DIRECT observation.state binding (conservative)")
    else:
        print(f"  schema state_keys={tuple(schema_obj.state_keys)} "
              f"virtual_state_sources={vmap or 'none'} → "
              f"{'DIRECT binding (widths must be 8)' if direct_binding else 'virtual binding (informational)'}")
    width_rows: dict[int, int] = defaultdict(int)
    state_failed_shards: list[tuple[Path, str]] = []
    diffs_by_width: dict[int, list[float]] = defaultdict(list)
    for s in shards:
        st, jt, reason = _read_state_and_joints(s)
        if reason is not None:
            state_failed_shards.append((s, reason))
            continue
        if st is None:
            continue
        w = int(st.shape[-1])
        width_rows[w] += int(st.shape[0])
        k = min(w, 8)
        diffs_by_width[w].append(float(np.mean(np.abs(st[:, :k] - jt[:, :k]))))
    if state_failed_shards:
        print(f"  ❌ {len(state_failed_shards)} sampled shard(s) unreadable for state audit")
        for sp, reason in state_failed_shards[:5]:
            print(f"      - {sp}: {reason}")
        failed = True
    total_rows = sum(width_rows.values()) or 1
    for w in sorted(width_rows):
        med = float(np.median(diffs_by_width[w])) if diffs_by_width[w] else float("nan")
        print(f"  width {w:3d}: {width_rows[w] / total_rows:6.1%} of sampled rows | "
              f"mean|state[:, :{min(w, 8)}] - joints| median={med:.4f}")
    non_canonical = {w for w in width_rows if w != 8}
    if non_canonical and direct_binding:
        print(f"  ❌ schema binds observation.state directly but sampled widths "
              f"{sorted(width_rows)} != {{8}} — training truncates/zero-pads "
              f"semantically-wrong state AND uses it as the delta reference "
              f"(see doc/repo_audit_20260610_round3_data_arch.md R3-D1)")
        failed = True
    elif non_canonical:
        print(f"  ✓ informational: physical state column is heterogeneous "
              f"({sorted(width_rows)}) but unused by the virtual-bound schema")
    else:
        print(f"  ✓ all sampled state rows are 8-wide")
    print()

    print("=" * 60)
    print(f"AUDIT RESULT: {'❌ FAIL' if failed else '✅ PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
