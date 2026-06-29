"""Central registry of every ``LABVLA_*`` environment flag.

Why: dozens of escape-hatch/tuning flags accumulated across modules with no
single place that says what exists, what the default is, or who reads it. A
single registry prevents silent launcher divergence on the same flag.

Contract:
  * ``get(name)`` returns the raw string value with the REGISTERED default —
    call sites keep their own comparison idiom (``== "1"`` / ``!= "0"`` /
    ``int(...)``) so migration is semantics-preserving by construction.
  * Entries with ``default=None`` are DOCUMENT-ONLY ("site-managed"): the
    reading module still owns parsing/default (e.g. storage_retry's five
    numeric knobs, deployment's path/secret). They are listed so the registry
    is the one complete inventory.
  * ``validate_environment()`` warns about set-but-unregistered ``LABVLA_*``
    vars — typos like ``LABVLA_ALOW_TRUNCATE`` used to disappear silently.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Flag:
    default: str | None     # None => document-only, site keeps its own default
    doc: str
    consumers: str          # module(s) that read it


FLAGS: dict[str, _Flag] = {
    # ---- data-pipeline escape hatches (default fail-loud) -----------------
    "LABVLA_ALLOW_TRUNCATE": _Flag(
        "0", "truncate overwide state/action columns instead of raising "
        "(adapters _pad_row + v2/v3 stats readers + scan-cache key). "
        "All production launchers set 0 (fail-loud); the historical 1-vs-0 "
        "launcher split was an incident, and there is NO safe repo exception "
        "(M47). One-off diagnostics only.",
        "adapters/lerobot_base, adapters/lerobot_v21, adapters/lerobot_v30, "
        "adapters/_scan_cache, data_process/stats/v2, data_process/stats/v3"),
    "LABVLA_ALLOW_TOKENIZED_TASK_COERCION": _Flag(
        None, "coerce tokenized (list<int>) v3 task cells to str() instead of "
        "raising — inspection-only runs.", "adapters/lerobot_v30"),
    "LABVLA_DATA_SKIP_CONTRACT_ERRORS": _Flag(
        "0", "let SkipBadSamplesDataset resample on deterministic "
        "schema-contract errors (KeyError/ValueError/...) instead of raising "
        "— legacy behavior; silently reweights data.", "scripts/utils/dataloader"),
    "LABVLA_ALLOW_SCHEMA_OVERRIDE": _Flag(
        "0", "allow re-registering a schema name with DIFFERENT content "
        "(import-order dependent identity; M66 default fail-loud).",
        "schema/registry"),
    "LABVLA_ALLOW_MISSING_STATE_SUBKEYS": _Flag(
        "0", "let DiscretizeStateTransformFn warn-and-skip samples missing a "
        "declared state sub-key instead of raising (M36/M109: mixed prompt "
        "contract).", "transforms/state_discretize"),
    "LABVLA_ALLOW_Q0199_FALLBACK": _Flag(
        "0", "fall back to mean_std/min_max when q01_q99 is requested but "
        "stats lack quantiles (legacy warn-only path; breaks gripper "
        "open/close alignment).", "transforms/core"),
    "LABVLA_ALLOW_MISSING_NORM_STATS": _Flag(
        "0", "let hydrate proceed when a normalized key has no stats "
        "(silently-unnormalized training).", "transforms/core"),
    "LABVLA_ALLOW_UNPATCHED_SNAP_STATS": _Flag(
        "0", "skip the snap-gripper stats-canonicalization guard.",
        "transforms/core"),
    "LABVLA_ALLOW_STATS_INVALIDATED": _Flag(
        None, "train despite cleanup-invalidated stats.", "scripts/utils/dataset_build"),
    "LABVLA_ALLOW_HETEROGENEOUS_ACTION_DIMS": _Flag(
        None, "bypass the multi-repo posttrain action-dim guard.", "scripts/train"),
    "LABVLA_ALLOW_HETEROGENEOUS_GRIPPER": _Flag(
        None, "bypass the gripper-layout guard under gripper_loss_weight!=1.",
        "scripts/train"),
    "LABVLA_ALLOW_HETEROGENEOUS_GRIPPER_SEMANTIC": _Flag(
        None, "demote the cross-repo gripper-semantic conflict to a warning.",
        "scripts/utils/dataset_helpers"),
    "LABVLA_FAST_ACTION_NON_CHUNK_SKIP": _Flag(
        "0", "legacy silent-skip when the action reaching FAST is not a (T,D) "
        "chunk.", "transforms/fast_action"),
    # ---- caches / performance --------------------------------------------
    "LABVLA_SCAN_CACHE": _Flag(
        "1", "0 disables the on-disk adapter scan cache.", "adapters/_scan_cache"),
    "LABVLA_VIDEO_CACHE_MAX": _Flag(
        "128", "process-wide PyAV container LRU size.", "adapters/lerobot_base"),
    "LABVLA_V21_VALIDATE_PER_FILE": _Flag(
        "1", "0 trusts one sampled parquet per chunk instead of per-file scan.",
        "adapters/lerobot_v21"),
    "LABVLA_WORKER_TRIM_EVERY": _Flag(
        None, "malloc_trim cadence in DataLoader workers.", "scripts/utils/dataset_helpers"),
    "LABVLA_DATA_SKIP_LOG_EVERY": _Flag(
        None, "rate limit for bad-sample skip warnings.", "scripts/utils/dataloader"),
    # ---- assets / paths ----------------------------------------------------
    "LABVLA_FAST_TOKENIZER_PATH": _Flag(
        None, "library-level default FAST asset path override.", "transforms/fast_action"),
    "LABVLA_SCHEMA_PATH": _Flag(
        None, "extra schema dirs (PYTHONPATH-style) for the registry autoload.",
        "schema/registry"),
    # ---- VQA ----------------------------------------------------------------
    "LABVLA_VQA_SKIP_BAD_RECORDS": _Flag(
        None, "skip unreadable VQA records instead of raising.",
        "dataset/adapters/robointer_vqa_adapter"),
    "LABVLA_VQA_SKIP_MAX_ATTEMPTS": _Flag(
        None, "'auto' or int retry budget for VQA bad-record skips.",
        "dataset/adapters/robointer_vqa_adapter"),
    # ---- storage retry (site-managed numeric parsing) ----------------------
    "LABVLA_STORAGE_RETRY_ENABLE": _Flag(None, "enable storage retry wrapper.", "utils/storage_retry"),
    "LABVLA_STORAGE_RETRY_TOTAL_SECONDS": _Flag(None, "total retry budget.", "utils/storage_retry"),
    "LABVLA_STORAGE_RETRY_NOT_FOUND_SECONDS": _Flag(None, "budget for FileNotFound.", "utils/storage_retry"),
    "LABVLA_STORAGE_RETRY_INITIAL_SLEEP": _Flag(None, "initial backoff.", "utils/storage_retry"),
    "LABVLA_STORAGE_RETRY_MAX_SLEEP": _Flag(None, "max backoff.", "utils/storage_retry"),
    # ---- deployment ---------------------------------------------------------
    "LABVLA_ROOT": _Flag(None, "repo root override for the serve entrypoint sys.path bootstrap (src. imports resolve from the repo root).",
                         "deployment/serve_labvla"),
    "LABVLA_WS_AUTH_TOKEN": _Flag(None, "websocket auth secret.", "deployment/serve_labvla"),
    # Shell-only (never read by Python): node89 dispatcher passes the rg-killer
    # daemon its pidfile path through the environment. Registered so
    # validate_environment() does not flag it as a typo on dispatch nodes.
    "LABVLA_RG_KILLER_PIDFILE": _Flag(None, "rg-killer daemon pidfile (shell-only).",
                                      "launch/_rg_killer.sh, launch/*node89*.sh"),
    "LABVLA_DEPLOY_ALLOW_PARTIAL_LOAD": _Flag(
        None, "tolerate missing keys when loading deploy weights.", "deployment/serve_labvla"),
}


def get(name: str) -> str | None:
    """Raw string value of a REGISTERED flag, with the registered default.

    Raises ``KeyError`` for unregistered names — adding a flag requires adding
    a registry entry (that is the point).
    """
    flag = FLAGS[name]
    return os.environ.get(name, flag.default)


def validate_environment(log: logging.Logger | None = None) -> list[str]:
    """Warn about set-but-unregistered LABVLA_* env vars (likely typos)."""
    log = log or logger
    unknown = sorted(
        k for k in os.environ if k.startswith("LABVLA_") and k not in FLAGS
    )
    if unknown:
        log.warning(
            "[env-flags] unrecognized LABVLA_* environment variable(s) set: %s "
            "— not read by any registered flag (typo? see utils/env_flags.py).",
            unknown,
        )
    return unknown
