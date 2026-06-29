"""Compute normalization stats for a LeRobot v2.1 or v3.0 dataset.

Single-pass pipeline:
  1. Read info.json, detect `codebase_version`.
  2. Dispatch to v2/reader or v3/reader → (acts_ep, states_ep) per episode.
  3. Apply DeltaActions (per schema.delta_mask) and accumulate into a
     RunningStats for mean/std (+ optional q01/q99).
  4. Write `<dataset>/meta/stats.json`.

For mask=True dims the action pipeline yields delta distribution; for
mask=False dims it yields absolute — one pass, no second sweep.

Usage:
  python -m data_process stats \\
      --dataset /path/to/dataset \\
      --schema  <name>_schema_file_stem \\
      [--chunk_size 50] \\
      [--no-quantile]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Expose the repo root BEFORE any data_process/src import so direct-file
# execution (python data_process/stats/compute.py) works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_process.stats.running import RunningStats, save as save_stats
# `from src.schema import resolve` is imported lazily inside main() so
# `python -m data_process stats -h` works in light tooling envs without the
# full schema/torch dependency chain.




def _canonicalize_layout_np(arr, schema):
    """Raw source-layout (T, raw_width) → canonical layout (T, canonical_dim).

    The v2/v3 readers return arrays in RAW source-column space (``source_*_keys``
    concat, e.g. a 16-wide dual-arm source or an 11-wide single-arm source), but
    ``schema.to_bool_mask()`` / ``delta_mask`` live in CANONICAL space (14 / 8).
    Raw space cannot express the canonical mask, and feeding raw arrays into the
    delta computation crashes the numpy broadcast (mask len != raw width).

    This mirrors the training-time transforms exactly:
      - DUAL  (``CanonicalArmLayoutTransformFn``):      14-dim
            out[:, :lk]      = arr[:, :lk]          (lk = min(left_dof, 6))
            out[:, 6]        = arr[:, left_grip]
            out[:, 7:7+rk]   = arr[:, rs:rs+rk]     (rs = left_dof, rk = min(right_dof, 6))
            out[:, 13]       = arr[:, right_grip]
      - SINGLE (``CanonicalSingleArmLayoutTransformFn``): 8-dim
            out[:, :ak]      = arr[:, :ak]          (ak = min(arm_dof, 7))
            out[:, 7]        = arr[:, grip_idx]

    No-op (returns ``arr`` unchanged) when the schema declares no arm_layout,
    or the input width already equals the canonical width (idempotent for
    re-runs on canonical data).
    """
    layout = getattr(schema, "arm_layout", None)
    if layout is None:
        return arr, False
    from src.schema.arm_layout import ArmCount

    canonical_dim = int(layout.canonical_dim)
    width = int(arr.shape[-1])
    if width == canonical_dim:
        return arr, False

    out = np.zeros(arr.shape[:-1] + (canonical_dim,), dtype=arr.dtype)
    if layout.arm_count == ArmCount.DUAL:
        lk = min(int(layout.left_arm_dof), 6)
        rk = min(int(layout.right_arm_dof), 6)
        rs = int(layout.left_arm_dof)
        lg = int(layout.left_gripper_index_in_raw)
        rg = int(layout.right_gripper_index_in_raw)
        if width <= max(lg, rg) or width < rs + rk:
            raise ValueError(
                f"_canonicalize_layout_np(DUAL): raw width {width} too small "
                f"for gripper indices ({lg}, {rg}) / right-arm slice "
                f"[{rs}:{rs + rk}] (schema={schema.schema_id!r})."
            )
        out[:, :lk] = arr[:, :lk]
        out[:, 6] = arr[:, lg]
        out[:, 7:7 + rk] = arr[:, rs:rs + rk]
        out[:, 13] = arr[:, rg]
    else:  # SINGLE
        ak = min(int(layout.arm_dof), 7)
        gi = int(layout.gripper_index_in_raw)
        if width <= gi or width < ak:
            raise ValueError(
                f"_canonicalize_layout_np(SINGLE): raw width {width} too small "
                f"for gripper index {gi} / arm slice [:{ak}] "
                f"(schema={schema.schema_id!r})."
            )
        out[:, :ak] = arr[:, :ak]
        out[:, 7] = arr[:, gi]
    return out, True


def _canonicalize_episodes(acts_ep, states_ep, schema):
    """Apply the raw→canonical layout mapping per episode where the schema
    declares raw source columns. Returns (acts_ep, states_ep, canonicalized).
    """
    has_src_action = bool(getattr(schema, "source_action_keys", ()) or ())
    has_src_state = bool(getattr(schema, "source_state_keys", ()) or ())
    if not (has_src_action or has_src_state):
        return acts_ep, states_ep, False

    any_canon = False
    new_acts, new_states = [], []
    for act, st in zip(acts_ep, states_ep):
        if has_src_action and act.shape[0] > 0:
            act, did = _canonicalize_layout_np(act, schema)
            any_canon = any_canon or did
        if has_src_state and st.shape[0] > 0:
            st, did = _canonicalize_layout_np(st, schema)
            any_canon = any_canon or did
        new_acts.append(act)
        new_states.append(st)
    return new_acts, new_states, any_canon


def _iter_delta_action_chunks(acts_ep, states_ep, mask, chunk_size):
    """DeltaActions: mask=True dims have state subtracted; mask=False stay
    absolute. Yields ONE big (N, D) batch per episode where
    N = Σ_{t=0}^{T-1} min(chunk_size, T-t). RunningStats.update is
    numpy-vectorized, so yielding per-episode (~5-10k rows) is far faster
    than per-chunk (~50 rows).

    When ``state_dim < action_dim`` (e.g. 7-dim joint state paired with 8-dim
    joint+gripper action), zero-pad ``st`` to action width before subtracting.
    Mirrors the runtime ``DeltaActionTransformFn`` in
    ``src/transforms/core.py`` and avoids a silent shape-broadcast error when
    ``mask`` has length ``action_dim`` but state is shorter.
    """
    mask = np.asarray(mask, dtype=bool)
    for act, st in zip(acts_ep, states_ep):
        T = act.shape[0]
        if T == 0:
            continue
        # Pad state to action width so `mask` (length action_dim) and `st[t]`
        # align. Matches src/transforms/core.py.
        if st.shape[-1] < act.shape[-1]:
            pad_width = act.shape[-1] - st.shape[-1]
            st = np.concatenate(
                [st, np.zeros(st.shape[:-1] + (pad_width,), dtype=st.dtype)],
                axis=-1,
            )
        elif st.shape[-1] > act.shape[-1]:
            # training-side DeltaActionTransformFn fails loud on a
            # state wider than action; silently truncating here would let the
            # stats tool "succeed" on a schema training must reject.
            raise ValueError(
                f"state width {st.shape[-1]} > action width {act.shape[-1]} "
                "after canonicalization — fix the schema (training rejects "
                "this layout)."
            )
        pieces = []
        for t in range(T):
            K = min(chunk_size, T - t)
            piece = act[t:t + K] - np.where(mask, st[t], 0.0)[None, :]
            pieces.append(piece)
        yield np.concatenate(pieces, axis=0)


def _iter_abs_action_chunks(acts_ep, states_ep, chunk_size):
    """Absolute action chunks (no state subtraction — ignores delta_mask).

    For training π0.5-style abs mode: stats reflect the raw target-pose
    distribution directly, regardless of what the schema's delta_mask says.
    Same chunking shape as delta version so q01/q99 are comparable.
    """
    for act in acts_ep:
        T = act.shape[0]
        if T == 0:
            continue
        pieces = []
        for t in range(T):
            K = min(chunk_size, T - t)
            pieces.append(act[t:t + K])
        yield np.concatenate(pieces, axis=0)


def _clear_stats_invalidated(ds_root: Path) -> None:
    """Remove the ``stats_invalidated`` block left by ``cleanup.apply`` after a
    successful stats recompute.

    Mirrors the inverse of ``data_process/cleanup/apply.py:rewrite_meta``:
      * deletes ``info.json["stats_invalidated"]`` (if present),
      * deletes the sibling ``meta/stats.invalidated.json`` marker.

    Best-effort and quiet on a clean dataset (no flag → nothing to do). The
    info.json rewrite is in-place; the file already exists because the reader
    parsed it earlier in this run.
    """
    meta = ds_root / "meta"
    info_path = meta / "info.json"
    if info_path.is_file():
        try:
            with open(info_path) as f:
                info = json.load(f)
        except (OSError, ValueError):
            info = None
        if isinstance(info, dict) and "stats_invalidated" in info:
            del info["stats_invalidated"]
            # MA10: atomic write (tmp + fsync + os.replace) — a crash mid-write
            # must not leave a truncated info.json (dataset_build reads it with
            # no try/except and would crash the next training launch). Mirrors
            # running.py::save().
            import os as _os
            import tempfile as _tempfile
            _fd, _tmp = _tempfile.mkstemp(
                prefix=info_path.name + ".", suffix=".tmp", dir=str(info_path.parent)
            )
            try:
                with _os.fdopen(_fd, "w") as f:
                    json.dump(info, f, indent=2)
                    f.flush()
                    _os.fsync(f.fileno())
                _os.replace(_tmp, str(info_path))
            except Exception:
                try:
                    _os.unlink(_tmp)
                except OSError:
                    pass
                raise
            print("[stats] cleared info.json['stats_invalidated'] "
                  "(stats successfully recomputed)")
    marker = meta / "stats.invalidated.json"
    if marker.is_file():
        marker.unlink()
        print(f"[stats] removed stale {marker.name}")


def _read_by_version(ds_root: Path, schema):
    """Inspect info.json, dispatch to the matching reader.

    Returns ``(acts_ep, states_ep, info)`` so the caller can validate frame
    coverage against ``info["total_frames"]`` without re-reading info.json.
    """
    info_path = ds_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json missing at {info_path}")
    with open(info_path) as f:
        info = json.load(f)
    ver = info.get("codebase_version", "")
    print(f"[stats] detected codebase_version={ver!r}")
    if ver.startswith("v2"):
        from data_process.stats.v2.reader import read_action_state
    elif ver.startswith("v3"):
        from data_process.stats.v3.reader import read_action_state
    else:
        raise ValueError(
            f"Unsupported codebase_version={ver!r}. Expected v2.x or v3.x."
        )
    acts_ep, states_ep = read_action_state(ds_root, schema)
    return acts_ep, states_ep, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=str,
                    help="schema name (file stem in schemas/) or path/to/file.py")
    ap.add_argument("--chunk_size", type=int, default=50)
    ap.add_argument("--no-quantile", action="store_true",
                    help="(DEBUG ONLY) write stats with q01/q99 set to null. "
                         "Does NOT skip the computation (quantiles are always "
                         "computed internally) and the artifact is REJECTED by "
                         "default training/deploy normalization (q01_q99 mode "
                         "fails loud on missing quantiles; H20).")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <dataset>/meta/stats.json")
    args = ap.parse_args()

    from src.schema import resolve as resolve_schema  # noqa: E402  (lazy import)

    schema = resolve_schema(args.schema)
    print(f"[stats] schema={schema.schema_id} "
          f"state_dim={sum(schema.state_dims)} action_dim={sum(schema.action_dims)}")
    mask = schema.to_bool_mask().cpu().numpy().astype(bool)
    print(f"[stats] delta_mask={mask.tolist()}")

    print(f"[stats] reading {args.dataset}/data/ ...")
    acts_ep, states_ep, info = _read_by_version(args.dataset, schema)
    total_frames = sum(a.shape[0] for a in acts_ep)
    print(f"[stats] {len(acts_ep)} episodes, {total_frames} frames")

    # Readers return RAW source-layout arrays for schemas that declare
    # source_*_keys; the delta_mask is CANONICAL. Canonicalize first so
    # (a) the np.where(mask, st, 0) broadcast below is well-formed and (b) the
    # delta distribution is computed in the same space training normalizes in.
    acts_ep, states_ep, layout_canonicalized = _canonicalize_episodes(
        acts_ep, states_ep, schema,
    )
    if layout_canonicalized:
        print(
            f"[stats] applied raw→canonical arm-layout mapping "
            f"(schema={schema.schema_id}, canonical action_dim={sum(schema.action_dims)}); "
            f"stats.json will carry _layout_canonicalization=true so train-time "
            f"hydrate skips its own remap."
        )
    # Fail loud on any residual mask/width mismatch (e.g. source keys declared
    # but no arm_layout) instead of letting numpy raise an opaque broadcast error.
    _bad_w = next(
        (a.shape[-1] for a in acts_ep if a.shape[0] > 0 and a.shape[-1] != len(mask)),
        None,
    )
    if _bad_w is not None:
        raise ValueError(
            f"[stats] action width {_bad_w} != delta_mask length {len(mask)} "
            f"after canonicalization (schema={schema.schema_id!r}). The schema "
            f"declares source_*_keys but no usable arm_layout mapping — add an "
            f"arm_layout to the schema or fix source dims."
        )

    # Fail-loud safety net for partial-coverage reader bugs (e.g. the v3
    # global/local index miss). 1% slack absorbs legitimate overwide /
    # missing-column drops; bigger gaps abort before writing stats.
    expected_frames = int(info.get("total_frames", 0))
    if expected_frames > 0:
        coverage = total_frames / expected_frames
        if coverage < 0.99:
            raise RuntimeError(
                f"[stats] partial-coverage abort: read {total_frames} frames but "
                f"info.json declares {expected_frames} (coverage={coverage:.2%}). "
                f"Aborting before writing stats — re-run after fixing the reader "
                f"or manifest."
            )
        print(f"[stats] frame coverage = {coverage:.2%} of info.json::total_frames")

    rs_state = RunningStats()
    for st in states_ep:
        rs_state.update(st)

    # Dual accumulator: delta-transformed (current "action" key) and raw abs
    # ("action_abs" key, used by action_mode="abs" training).
    rs_action_delta = RunningStats()
    for chunk in _iter_delta_action_chunks(
        acts_ep, states_ep, mask, args.chunk_size,
    ):
        rs_action_delta.update(chunk)

    rs_action_abs = RunningStats()
    for chunk in _iter_abs_action_chunks(acts_ep, states_ep, args.chunk_size):
        rs_action_abs.update(chunk)

    stats = {
        "observation.state": rs_state.get_statistics(),
        "action": rs_action_delta.get_statistics(),
        "action_abs": rs_action_abs.get_statistics(),
    }
    if args.no_quantile:
        print(
            "[stats] WARNING --no-quantile: writing stats WITHOUT q01/q99. "
            "Default training (q01_q99 normalization) and deployment will "
            "REJECT this artifact; it is only useful for debugging (H20)."
        )
        for s in stats.values():
            s.q01 = None
            s.q99 = None

    out = args.out or args.dataset / "meta" / "stats.json"
    # Persist chunk_size into stats.json so training can verify the stats were
    # computed against the chunk_size it actually uses. The
    # _layout_canonicalization marker tells train-time
    # _canonicalize_single_arm_stats that these arrays are ALREADY canonical —
    # without it, the hydrate-time raw→canonical remap would re-permute an
    # 8-dim canonical array and corrupt the gripper stats. The marker is a
    # behavior switch on the training side (hydrate skips its own remap), so
    # bind it to the schema and source geometry it was computed for: a
    # stale/copied sidecar cannot silently disable canonicalization for a
    # different layout.
    _extra = None
    if layout_canonicalized:
        _al = getattr(schema, "arm_layout", None)
        _extra = {"_layout_canonicalization": {
            "schema_id": str(schema.schema_id),
            "source_state_keys": [str(k) for k in (getattr(schema, "source_state_keys", ()) or ())],
            "source_state_dims": [int(d) for d in (getattr(schema, "source_state_dims", ()) or ())],
            "source_action_keys": [str(k) for k in (getattr(schema, "source_action_keys", ()) or ())],
            "source_action_dims": [int(d) for d in (getattr(schema, "source_action_dims", ()) or ())],
            "arm_layout": (
                {"arm_count": _al.arm_count.value,
                 "arm_dof": (int(_al.arm_dof) if _al.arm_dof is not None else None),
                 "gripper_index_in_raw": (int(_al.gripper_index_in_raw)
                                          if getattr(_al, "gripper_index_in_raw", None) is not None else None)}
                if _al is not None else None
            ),
            "producer": "data_process.stats.compute",
        }}
    save_stats(out, stats, chunk_size=int(args.chunk_size), extra_meta=_extra)
    print(
        f"[stats] wrote {out} (chunk_size={int(args.chunk_size)}"
        + (", _layout_canonicalization=true" if layout_canonicalized else "")
        + ")"
    )

    # Clear the cleanup-set ``stats_invalidated`` flag now that a correct
    # recompute has landed: cleanup.apply sets it (+ a stats.invalidated.json
    # marker) after dropping episodes, and dataset_build.py refuses to start
    # while it's set, so without this the dataset stays permanently invalid even
    # after the operator re-ran stats. Only clear when we wrote this dataset's
    # canonical meta/stats.json (not a custom --out), to avoid touching others.
    _wrote_canonical = Path(out).resolve() == (
        args.dataset / "meta" / "stats.json"
    ).resolve()
    if _wrote_canonical:
        _clear_stats_invalidated(args.dataset)
    for k, s in stats.items():
        print(f"  {k}: mean[:4]={s.mean[:4].round(4)} std[:4]={s.std[:4].round(4)}"
              + (f" q01[:4]={s.q01[:4].round(4)}" if s.q01 is not None else ""))


if __name__ == "__main__":
    main()
