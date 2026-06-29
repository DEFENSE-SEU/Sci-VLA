"""Checkpoint I/O, atomic-write helpers, resume contract validation."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import torch

from src.utils.storage_retry import is_transient_storage_error, run_with_storage_retry


# Keys whose checkpoint-vs-current divergence blocks a full resume. Anything
# touching architecture shape, optimizer state, dataset semantics, or RNG seed
# must appear here — letting a mismatched arg slip past defeats the contract
# (see `_validate_full_resume_contract`).
_FULL_RESUME_CONTRACT_KEYS = (
    "batch_size",
    "gradient_accumulation_steps",
    "training_phase",
    "knowledge_isolation",
    "use_fast_tokenizer",
    "action_mode",
    "dit_num_layers",
    "dit_num_heads",
    "dit_head_dim",
    "chunk_size",
    "max_state_dim",
    "max_action_dim",
    "lr",
    "vlm_lr",
    "dit_lr",
    "warmup_steps",
    "decay_steps",
    "decay_lr",
    "weight_decay",
    "grad_clip_norm",
    "gradient_checkpointing",
    "gc_visual_encoder",
    "gc_language_model",
    "gc_dit",
    "freeze_vision_encoder",
    "train_expert_only",
    "train_vlm_only",
    "seed",
    "repo_ids",
    "dataset_weights",
    "dataset_schema",
    "external_stats_map",
    "homogeneous_mixture_batches",
    "trim_token_padding_to_batch",
    "source_shape_convergence",
    "discrete_action_vocab_size",
    "discrete_action_max_length",
    # These training-semantic args change attention pattern, gripper
    # supervision, per-dim normalization, or the FAST CE skip set; surface any
    # ckpt-vs-current divergence rather than letting it drift silently.
    "pi05_block_attention_mask",       # attention semantics
    "action_supervision_skip_repos",   # disables FAST CE per-repo
    "normalize_arm_joints",            # input normalization (per-joint stats)
    "normalize_gripper",               # input normalization (gripper channel)
    "gripper_norm_mode",               # raw/q01_q99/standard/noop
    "snap_gripper_to_binary",          # binarize gripper target
    "gripper_loss_weight",             # per-dim MSE weighting in action head
    # More keys that change training semantics, model topology, or the data
    # pipeline; resume must not succeed while their meaning has drifted.
    "attn_implementation",             # FA2 vs SDPA backend (drives pi05 mask compat)
    "dit_interleave_self_attention",   # DiT block layout (Spirit interleave on/off)
    "dit_layerwise_vlm_features",      # DiT cross-attn reads layerwise VLM features
    "dit_dropout",                     # DiT regularization
    "ki_mse_weight",                   # KI loss mixture (α; π0.5 uses 10)
    "discretize_state_in_vlm_pretrain",  # state representation in VLM pretrain
    "image_height",                    # image pipeline / token count
    "image_width",                     # image pipeline / token count
    "image_augmentation",              # train-time image aug on/off
    "external_stats_path",             # single-repo norm-stats override path
    "fast_tokenizer_path",             # FAST tokenizer source path
    "vlm_pretrained_path",             # VLM init weights source path
    "ema_decay",                       # EMA buffer existence + decay rate
    # Data-pipeline / RNG keys that change training semantics across resume:
    # gripper_max_width / gripper_canonical_dim drive the snap-gripper transform;
    # data_root flips between dataset copies; num_workers changes the per-worker
    # seed formula; resumable_dataloader switches sampler semantics;
    # data_error_skip{,_max_attempts} change the sample stream on bad samples.
    "gripper_max_width",
    "gripper_canonical_dim",
    "data_root",
    "num_workers",
    "resumable_dataloader",
    "data_error_skip",
    "data_error_skip_max_attempts",
    # The optimizer-step horizon and the LR auto-scale policy are rebuilt from
    # CURRENT CLI values before the saved scheduler state is loaded — resuming
    # with a different --steps or --allow_lr_auto_scale silently rewrites the LR
    # recipe mid-run.
    "steps",
    "allow_lr_auto_scale",
)


def _torch_load_with_retry(path: Path):
    return run_with_storage_retry(
        lambda: torch.load(str(path), map_location="cpu", weights_only=False),
        path=path,
        description="torch.load",
    )


def _atomic_write_with_retry(path: Path, writer, *, description: str) -> None:
    """Write one checkpoint file via tmp-in-same-dir + atomic replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")

    def _write() -> None:
        if tmp_path.exists():
            tmp_path.unlink()
        writer(tmp_path)
        os.replace(str(tmp_path), str(path))

    try:
        run_with_storage_retry(_write, path=path, description=description)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logging.debug("Could not remove temporary checkpoint file %s", tmp_path)


def _torch_save_with_retry(obj, path: Path) -> None:
    _atomic_write_with_retry(
        Path(path),
        lambda tmp_path: torch.save(obj, str(tmp_path)),
        description="torch.save",
    )


def _json_dump_with_retry(obj, path: Path) -> None:
    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)

    _atomic_write_with_retry(Path(path), _write, description="json checkpoint write")


def _json_safe_config_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_config_value(asdict(value))
    if isinstance(value, tuple):
        return [_json_safe_config_value(v) for v in value]
    if isinstance(value, list):
        return [_json_safe_config_value(v) for v in value]
    if isinstance(value, dict):
        return {
            str(_json_safe_config_value(k)): _json_safe_config_value(v)
            for k, v in value.items()
        }
    json.dumps(value)
    return value


def _normalize_resume_contract_value(key: str, value):
    if key == "repo_ids":
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(str(v).strip() for v in value if str(v).strip())
        return tuple(part.strip() for part in str(value).split(",") if part.strip())
    # Map-form values ("a=x,b=y") are order-insensitive and path-tolerant — sort
    # entries by repo key and resolve path-like RHS so cosmetic drift does not
    # block a safe resume.
    if key in ("external_stats_map", "dataset_schema"):
        if value is None or value == "":
            return None
        sval = str(value)
        if "=" not in sval:
            return sval.strip()
        entries = []
        for part in sval.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                v = v.strip()
                if key == "external_stats_map":
                    try:
                        v = str(Path(v).resolve())
                    except Exception:
                        pass
                entries.append((k.strip(), v))
            else:
                entries.append((part, ""))
        return tuple(sorted(entries))
    # Resolve path-like fields to absolute before comparison so cosmetic
    # differences ("./foo" vs "/abs/foo", trailing slash) don't trigger false
    # mismatches. Real divergence (different weights/stats file) still surfaces.
    if key in ("external_stats_path", "fast_tokenizer_path",
               "vlm_pretrained_path"):
        if value is None or value == "":
            return None
        try:
            return str(Path(str(value)).resolve())
        except Exception:
            return str(value).rstrip("/")
    return value


def _checkpoint_incomplete_marker(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / ".INCOMPLETE"


def _write_checkpoint_incomplete_marker(checkpoint_dir: str | Path, step: int) -> None:
    marker = _checkpoint_incomplete_marker(checkpoint_dir)
    _json_dump_with_retry(
        {"step": int(step), "status": "checkpoint write in progress"},
        marker,
    )


def _clear_checkpoint_incomplete_marker(checkpoint_dir: str | Path) -> None:
    marker = _checkpoint_incomplete_marker(checkpoint_dir)
    if not marker.exists():
        return
    try:
        # A transient unlink failure must not leave a COMPLETE checkpoint
        # permanently marked incomplete (full resume would then hard-refuse it).
        run_with_storage_retry(marker.unlink, path=marker,
                               description="clear .INCOMPLETE marker")
    except Exception as e:  # noqa: BLE001
        logging.critical(
            "[M99] checkpoint %s is COMPLETE but its .INCOMPLETE marker could "
            "not be removed (%s). Full resume will refuse this checkpoint "
            "until the marker is deleted manually: rm %s",
            checkpoint_dir, e, marker,
        )


def _rotate_checkpoints(output_dir: Path, keep_last: int, *, sticky_every: int = 50000) -> None:
    keep_last = int(keep_last)
    if keep_last <= 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    checkpoints.sort()
    protected = {path for _, path in checkpoints[-keep_last:]}
    for step, path in checkpoints:
        if path in protected:
            continue
        if sticky_every > 0 and step > 0 and step % int(sticky_every) == 0:
            continue
        try:
            shutil.rmtree(path)
        except OSError as e:
            # Claiming success on a failed delete poisons disk-usage and
            # retention forensics.
            logging.warning("Rotation could NOT remove %s: %s", path, e)
            continue
        logging.info("Removed old checkpoint %s (save_total_limit=%d)", path, keep_last)


def _clone_shared_tensors_for_safetensors(state_dict: dict) -> dict:
    """Prepare a state_dict for ``safetensors.save_file``.

    Qwen ties ``lm_head.weight`` and ``language_model.embed_tokens.weight``.
    Deployment strict-loads via ``safetensors.torch.load_model``; saving both
    aliases makes the tied input-embedding alias an unexpected key. Keep the
    canonical lm_head tensor and omit the duplicate embed_tokens alias. Other
    shared tensors are still cloned so safetensors can serialize them.
    """

    tied_lm_head_key = "model.vlm.lm_head.weight"
    tied_embed_key = "model.vlm.model.language_model.embed_tokens.weight"
    tied_group_ptr = None

    lm_head = state_dict.get(tied_lm_head_key)
    embed = state_dict.get(tied_embed_key)
    if (
        isinstance(lm_head, torch.Tensor)
        and isinstance(embed, torch.Tensor)
        and lm_head.data_ptr() == embed.data_ptr()
    ):
        tied_group_ptr = lm_head.data_ptr()

    seen_data_ptrs = {}
    deduped = {}
    for key, tensor in state_dict.items():
        if (
            key == tied_embed_key
            and tied_group_ptr is not None
            and isinstance(tensor, torch.Tensor)
            and tensor.data_ptr() == tied_group_ptr
        ):
            continue
        if isinstance(tensor, torch.Tensor):
            ptr = tensor.data_ptr()
            if ptr in seen_data_ptrs:
                deduped[key] = tensor.clone()
            else:
                seen_data_ptrs[ptr] = key
                deduped[key] = tensor
        else:
            deduped[key] = tensor
    return deduped


def _is_recoverable_checkpoint_save_error(exc: BaseException) -> bool:
    """Only shared-storage transient failures are safe to skip and retry later."""

    return is_transient_storage_error(exc)


def _load_training_state(checkpoint_path: str | Path) -> dict:
    train_state_path = Path(checkpoint_path) / "training_state" / "training_state.pt"
    if not train_state_path.exists():
        return {}
    return _torch_load_with_retry(train_state_path)


def _accelerate_state_dir(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path) / "accelerate_state"


def _path_inside(child, root) -> bool:
    """True iff `child` lies inside (or equals) `root`. Checks BOTH the
    textual normalized form and the symlink-resolved form: a checkpoint whose
    pretrained_model/ is a symlink out of the tree must still count as
    "inside" (resolved test would fail), and a path reached THROUGH a
    symlinked root must too (textual test would fail)."""
    try:
        c_txt = Path(os.path.normpath(str(child)))
        r_txt = Path(os.path.normpath(str(root)))
        if c_txt == r_txt or r_txt in c_txt.parents:
            return True
        c_res = Path(str(child)).resolve()
        r_res = Path(str(root)).resolve()
        return c_res == r_res or r_res in c_res.parents
    except (OSError, ValueError):
        return False


def _validate_full_resume_contract(saved_args: dict, current_args) -> None:
    mismatches = []
    resume_root = getattr(current_args, "resume", None)
    for key in _FULL_RESUME_CONTRACT_KEYS:
        old = _normalize_resume_contract_value(key, saved_args.get(key))
        new = _normalize_resume_contract_value(key, getattr(current_args, key, None))
        if old != new:
            # GPU-resume smoke 2026-06-11: self-contained resume repoints
            # vlm_pretrained_path INTO the checkpoint (bundled vlm_config.json
            # + tokenizer) BEFORE this check runs, while the saved args still
            # hold the launch-time base path — so every bundled checkpoint
            # would fail full resume here. Model identity is enforced by the
            # bundled config + the loaded weights, so skip the comparison iff
            # the current path resolves inside the resume checkpoint. Plain
            # (non-bundled) checkpoints keep the strict path comparison.
            if key == "vlm_pretrained_path" and resume_root and new:
                if _path_inside(new, resume_root):
                    continue
                # Confirm-review NF-1 (2026-06-11): _stabilize_vlm_assets
                # repoints the path to <output_dir>/_vlm_assets/<digest> —
                # OUTSIDE the checkpoint — before this check runs, so the
                # original exemption rejected every stabilized full resume.
                _out_dir = getattr(current_args, "output_dir", None)
                if _out_dir and _path_inside(new, Path(_out_dir) / "_vlm_assets"):
                    # The stable dir must be SELF-VERIFYING — its name is the
                    # content hash of its own vlm_config.json. A foreign / stale
                    # / hand-made dir under _vlm_assets does not pass.
                    try:
                        import hashlib as _hl
                        _cfg = Path(str(new)) / "vlm_config.json"
                        if _cfg.is_file() and Path(str(new)).name == _hl.sha256(
                            _cfg.read_bytes()
                        ).hexdigest()[:16]:
                            continue
                    except OSError:
                        pass
                    # fall through to mismatch (fail-loud)
            mismatches.append((key, old, new))
    if mismatches:
        preview = "; ".join(
            f"{key}: ckpt={old!r} current={new!r}"
            for key, old, new in mismatches[:12]
        )
        extra = "" if len(mismatches) <= 12 else f"; ... +{len(mismatches) - 12} more"
        raise RuntimeError(
            "Full resume contract mismatch. Use --load_weights_only true for "
            "warmstart/finetune, or keep the original training config for exact "
            f"resume. Mismatches: {preview}{extra}"
        )


# (src_name, dest_name) for the small Qwen3-VL assets bundled FLAT into the checkpoint's
# pretrained_model/ dir so it is self-contained (loadable without an external
# vlm_pretrained_path). The arch config is renamed config.json -> vlm_config.json to
# avoid colliding with LabVLA's own config.json. The 8.9GB weight shards and the unused
# generation_config.json are intentionally NOT bundled.
_QWEN_ASSET_FILES = (
    ("config.json", "vlm_config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("vocab.json", "vocab.json"),
    ("merges.txt", "merges.txt"),
    ("chat_template.json", "chat_template.json"),
    ("preprocessor_config.json", "preprocessor_config.json"),
    ("video_preprocessor_config.json", "video_preprocessor_config.json"),
)


# Public: the bundled-asset filenames as they appear INSIDE a checkpoint's
# pretrained_model/ dir (dest names of _QWEN_ASSET_FILES). train.py uses this
# to stabilize the bundle outside the rotation-managed checkpoint dirs.
QWEN_BUNDLED_ASSET_NAMES = tuple(dst for _, dst in _QWEN_ASSET_FILES)


def _bundle_vlm_assets(model_dir: Path, vlm_source: str) -> None:
    """Copy the small Qwen3-VL config/tokenizer/processor files (NOT the 8.9GB weights)
    FLAT into <model_dir> (the checkpoint's pretrained_model/ dir), making the checkpoint
    self-contained so it loads/deploys without an external vlm_pretrained_path. The arch
    config is saved as vlm_config.json (renamed to avoid colliding with LabVLA's own
    config.json). Best-effort: any failure is logged and swallowed — it must never crash
    a training step.
    """
    try:
        import shutil
        src = Path(vlm_source)
        if not src.is_dir():
            logging.warning(
                f"[vlm_assets] vlm_pretrained_path {vlm_source!r} is not a local dir "
                "(likely an HF id); skipping asset bundling (loaders fall back to the "
                "configured vlm_pretrained_path)."
            )
            return
        copied, missing = [], []
        for src_name, dest_name in _QWEN_ASSET_FILES:
            sp = src / src_name
            if sp.exists():
                _atomic_write_with_retry(
                    model_dir / dest_name,
                    lambda tmp_path, _sp=sp: shutil.copyfile(_sp, tmp_path),
                    description=f"bundle vlm asset {dest_name}",
                )
                copied.append(dest_name)
            else:
                missing.append(src_name)
        logging.info(
            f"[vlm_assets] bundled {len(copied)} Qwen3-VL asset files (flat) into {model_dir}"
            + (f"; missing/skipped: {missing}" if missing else "")
        )
    except Exception as exc:
        logging.warning(f"[vlm_assets] asset bundling failed (non-fatal): {exc}")


# Config fields a published checkpoint's config.json does NOT need: not read on any
# deploy path (model build / inference / deploy-transform) — only in the training
# forward/loss, vlm_pretrain-only branches, optimizer/scheduler setup, or overridden
# (vlm_pretrained_path is redirected to the self-contained bundle at load). Skipping them
# keeps published configs minimal. (config.json is read ONLY at deploy — finetune uses
# build_config(args), not the ckpt config; deploy filters via allowed_fields + defaults,
# so a missing field is harmless.) Behaviorally-critical fields like action_mode and
# training_phase are deliberately NOT here (serve reads them off the saved config).
_DEPLOY_INEFFECTIVE_CONFIG_FIELDS = frozenset({
    "CONFIG_NAME", "n_obs_steps", "time_sampling_beta_alpha", "time_sampling_beta_beta",
    "time_sampling_scale", "time_sampling_offset", "empty_cameras", "vlm_pretrained_path",
    "normalization_mapping", "fast_tokenizer_path", "discrete_action_max_length",
    "ki_mse_weight", "discretize_state_in_vlm_pretrain",
})


def save_checkpoint(checkpoint_dir, step, policy, optimizer, scheduler, args,
                    norm_stats=None, schema=None, all_schemas=None,
                    epoch: int = 0, include_optimizer_state: bool = True,
                    include_scheduler_state: bool = True,
                    distributed_state_saved: bool = False,
                    distributed_identity: dict | None = None):
    """Persist a training checkpoint.

    Resume semantics:
      - Restored on full distributed resume: optimizer state, LR scheduler state, RNG state
        (Python / NumPy / PyTorch CPU+CUDA), wandb run id, nominal epoch counter
        (forwarded to PI0Mixture-style datasets via ``set_epoch``). Full
        distributed resume requires a sibling ``accelerate_state/`` written by
        ``accelerator.save_state()``; this helper records whether that
        distributed state was saved by its caller.
      - Restored for current deterministic paths: per-DataLoader sampler
        cursor *within* an epoch. Homogeneous PI0Mixture batches and the
        deterministic single-dataset batch sampler both derive the resume
        offset from the checkpoint step and reset it at the next epoch boundary.

    Backwards compat: older ckpts lack the ``epoch`` field; load/resume paths
    default to 0 when the key is absent.
    """
    # Deployment auto-discovery relies on BOTH norm_stats.json AND labvla_schema.json
    # being written alongside the model. If norm_stats is given but schema is
    # not, deployment falls back to --robot_type + RobotConfig (sub-optimal
    # reproducibility). Fail loudly so callers wire both.
    if norm_stats is not None and schema is None:
        raise ValueError(
            "save_checkpoint: norm_stats provided without schema. "
            "Pass schema=<DatasetSchema> so deployment can reproduce the layout. "
            "Typically: schema=data_schemas[first_repo_id] from build_dataset()."
        )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_dir = checkpoint_dir / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        state_dict = _clone_shared_tensors_for_safetensors(policy.state_dict())
        _atomic_write_with_retry(
            model_dir / "model.safetensors",
            lambda tmp_path: save_file(state_dict, str(tmp_path)),
            description="save model.safetensors",
        )
    except ImportError:
        _torch_save_with_retry(policy.state_dict(), model_dir / "pytorch_model.bin")

    import json
    config_dict = {}
    skip_config_fields = {"input_features", "output_features"} | _DEPLOY_INEFFECTIVE_CONFIG_FIELDS
    for k, v in vars(policy.config).items():
        if k.startswith("_") or k in skip_config_fields:
            continue
        try:
            config_dict[k] = _json_safe_config_value(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"save_checkpoint: LabVLAConfig field {k!r} is not JSON-serializable "
                "by the typed checkpoint serializer. Add an explicit conversion "
                "instead of falling back to repr strings."
            ) from exc
    if hasattr(args, "action_mode"):
        config_dict["action_mode"] = str(args.action_mode)
    _json_dump_with_retry(config_dict, model_dir / "config.json")

    # Bundle the Qwen3-VL config/tokenizer/processor (no weights) so the checkpoint
    # is self-contained (loadable/deployable without an external vlm_pretrained_path).
    _bundle_vlm_assets(model_dir, getattr(policy.config, "vlm_pretrained_path", ""))

    # Save norm_stats alongside the checkpoint so deployment can load them
    # without needing a separate --norm_stats_path argument.
    if norm_stats is not None:
        norm_stats_path = checkpoint_dir / "norm_stats.json"
        _json_dump_with_retry(norm_stats, norm_stats_path)
        logging.info(f"Saved norm_stats to {norm_stats_path}")

    # Save DatasetSchema alongside the checkpoint so deployment can auto-discover
    # the dataset layout without relying on --robot_type CLI arg.
    if schema is not None:
        schema_path = checkpoint_dir / "labvla_schema.json"
        _json_dump_with_retry(
            schema.to_dict() if hasattr(schema, "to_dict") else schema,
            schema_path,
        )
        logging.info(f"Saved schema to {schema_path}")

    # Persist *all* per-repo schemas in multi-repo training so the operator can
    # pick the right one at deploy time. labvla_schema.json (above) keeps the
    # first repo so the existing auto-discover flow still works.
    if all_schemas and len(all_schemas) > 1:
        schemas_path = checkpoint_dir / "labvla_schemas.json"
        dumpable = {}
        for rid, sch in all_schemas.items():
            try:
                dumpable[rid] = sch.to_dict() if hasattr(sch, "to_dict") else sch
            except Exception as _e:
                # A reduced bundle would silently shadow the healthy
                # single-schema fallback at deploy; all-or-nothing.
                raise RuntimeError(
                    f"save_checkpoint: could not serialize schema for repo "
                    f"{rid!r} ({_e}) — refusing to write a PARTIAL "
                    f"labvla_schemas.json (deploy would treat the reduced "
                    f"bundle as authoritative)."
                ) from _e
        _json_dump_with_retry({"_schema": "multi_repo_v1", **dumpable}, schemas_path)
        logging.info(f"Saved all repo schemas to {schemas_path} (n={len(dumpable)})")

    train_state_dir = checkpoint_dir / "training_state"
    train_state_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: Under DeepSpeed ZeRO-2, optimizer.state_dict() only contains the
    # local shard. Restoring optimizer state from this checkpoint across different
    # world sizes will fail. For full optimizer state, use DeepSpeed's
    # optimizer.consolidate_state_dict() or accelerator.save_state().
    # Capture RNG state so resume with --load_weights_only false reproduces the
    # same data shuffling / augmentation sequence as an uninterrupted run.
    # Best-effort: missing keys on load are skipped.
    import random as _random
    import numpy as _np
    rng_state = {
        "python": _random.getstate(),
        "numpy": _np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    # Capture wandb run_id so full-resume can reconnect to the same run.
    _wandb_run_id = None
    try:
        import wandb as _wb
        if _wb.run is not None:
            _wandb_run_id = _wb.run.id
    except Exception:
        pass

    import uuid as _uuid
    _checkpoint_id = f"step{int(step)}-{_uuid.uuid4().hex[:12]}"
    if distributed_state_saved:
        _json_dump_with_retry(
            {"checkpoint_id": _checkpoint_id, "step": int(step)},
            _accelerate_state_dir(checkpoint_dir) / "labvla_checkpoint_id.json",
        )

    train_state = {
        "step": step,
        # Nominal epoch counter; lets resume re-prime PI0Mixture's per-epoch
        # RNG seed (see save_checkpoint docstring).
        "epoch": int(epoch),
        "args": vars(args),
        "rng_state": rng_state,
        "wandb_run_id": _wandb_run_id,
        "accelerate_state": "accelerate_state",
        "full_resume_requires_distributed_state": bool(distributed_state_saved),
        "distributed_state_saved": bool(distributed_state_saved),
        # Bind this sidecar to the sibling accelerate_state/ written in the same
        # save — a hand-assembled or partially-synced checkpoint dir mixing
        # artifacts from different saves must be rejected before optimizer/RNG
        # state is restored.
        "checkpoint_id": _checkpoint_id,
        # Typed lineage (consumers no longer reverse-engineer args).
        "resume_mode": (
            "weights_only" if getattr(args, "load_weights_only", False)
            and getattr(args, "resume", None)
            else "full" if getattr(args, "resume", None) else "fresh"
        ),
        "source_checkpoint": str(getattr(args, "resume", None) or ""),
        # Topology identity — full resume across a different world size / mixed
        # precision cannot restore ZeRO shards correctly.
        "distributed_identity": dict(distributed_identity or {}),
    }
    if include_optimizer_state:
        train_state["optimizer_state_dict"] = optimizer.state_dict()
    if include_scheduler_state:
        train_state["scheduler_state_dict"] = scheduler.state_dict()
    _torch_save_with_retry(train_state, train_state_dir / "training_state.pt")

    logging.info(f"Checkpoint saved to {checkpoint_dir} at step {step}")


def _is_deepspeed_optimizer_state(state: dict) -> bool:
    """Detect DeepSpeed ZeRO optimizer state_dict format.

    Standard PyTorch optimizer state_dict has ['state', 'param_groups']. DS
    ZeRO wraps this under 'base_optimizer_state' and adds 'zero_stage',
    'partition_count', 'loss_scaler', etc. If we see those, the current
    optimizer must be a DS-wrapped optimizer (which happens only AFTER
    accelerator.prepare) for load_state_dict to succeed.
    """
    if not isinstance(state, dict):
        return False
    ds_markers = {"base_optimizer_state", "zero_stage", "partition_count",
                  "single_partition_of_fp32_groups", "ds_version"}
    has_ds_marker = any(k in state for k in ds_markers)
    # Two-factor check: DS markers present AND standard PyTorch optimizer keys
    # absent, to avoid false positives from third-party wrappers reusing names.
    has_pt_keys = ("state" in state and "param_groups" in state)
    return has_ds_marker and not has_pt_keys


def load_checkpoint(checkpoint_path, policy, optimizer=None, scheduler=None,
                    strict: bool | None = None):
    """Load model weights (and optionally optimizer/scheduler/RNG).

    When `optimizer` is passed but the saved optimizer state is in DeepSpeed
    ZeRO format, calling `optimizer.load_state_dict(...)` on a raw PyTorch
    optimizer raises `KeyError: 'param_groups'`. Callers wanting full resume
    with DeepSpeed must invoke this AFTER `accelerator.prepare`, when the
    optimizer is DS-wrapped and accepts DS format. We detect the format and
    either use the provided optimizer if compatible, or skip with a loud warning
    (still returning step so training resumes with weights + step).

    Full resume defaults to strict=True so architectural drift (changed
    dit_num_layers, missing ki_head, renamed projections, etc.) fails loudly
    instead of silently re-initializing missing parameters. Weights-only
    warm-start (optimizer is None) uses strict=False, since it routinely
    transfers a pretrain ckpt into a smaller downstream head with intentionally
    different shapes. Override via the ``strict`` arg for legacy behavior.
    """
    checkpoint_path = Path(checkpoint_path)
    # Legacy checkpoints (pre tied-weight dedup) carry BOTH
    # model.vlm.lm_head.weight and the tied ...embed_tokens.weight; the
    # deploy/generic loader tolerates the alias but this strict training path
    # does not. Normalize the known tied pair before load_state_dict (drop the
    # duplicate embed entry when both names are present — matches the save-side
    # dedup which keeps lm_head).

    weights_only_warmstart = optimizer is None and scheduler is None
    if strict is None:
        strict = not weights_only_warmstart

    model_dir = checkpoint_path / "pretrained_model"
    if model_dir.exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_dir / "model.safetensors"))
            # Normalize the known tied pair on legacy checkpoints that carry
            # BOTH names — keep lm_head (save-side dedup convention) and drop the
            # duplicate embedding entry so strict loads pass.
            for _lm, _emb in (
                ("model.vlm.lm_head.weight",
                 "model.vlm.model.language_model.embed_tokens.weight"),
            ):
                if _lm in state_dict and _emb in state_dict:
                    if torch.equal(state_dict[_lm], state_dict[_emb]):
                        state_dict.pop(_emb)
                        logging.info(
                            "[M22] dropped duplicate tied alias %s from legacy "
                            "checkpoint (kept %s).", _emb, _lm,
                        )
        except (ImportError, FileNotFoundError):
            state_dict = _torch_load_with_retry(model_dir / "pytorch_model.bin")
        load_result = policy.load_state_dict(state_dict, strict=strict)
        if load_result.missing_keys:
            logging.warning(f"Missing keys in checkpoint: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logging.warning(f"Unexpected keys in checkpoint: {load_result.unexpected_keys}")

        train_state_path = checkpoint_path / "training_state" / "training_state.pt"
        if train_state_path.exists():
            train_state = _torch_load_with_retry(train_state_path)
            if optimizer and "optimizer_state_dict" in train_state:
                saved_opt = train_state["optimizer_state_dict"]
                # Check if saved state is DS-format vs optimizer is DS-wrapped.
                is_ds_saved = _is_deepspeed_optimizer_state(saved_opt)
                is_ds_opt = type(optimizer).__module__.startswith("deepspeed")
                if is_ds_saved and not is_ds_opt:
                    logging.warning(
                        "Optimizer state in checkpoint is DeepSpeed ZeRO format "
                        "(keys include 'base_optimizer_state'), but the target "
                        "optimizer is a raw PyTorch optimizer (not yet wrapped "
                        "by accelerator.prepare). Skipping optimizer restore to "
                        "avoid KeyError. To fully resume, call load_checkpoint "
                        "AFTER accelerator.prepare, or restart with "
                        "--load_weights_only true."
                    )
                else:
                    try:
                        optimizer.load_state_dict(saved_opt)
                    except (ValueError, RuntimeError, KeyError) as e:
                        logging.warning(
                            f"Skipping optimizer state restore: {e}. "
                            f"This is expected when resuming across a different "
                            f"world size / DeepSpeed toggle."
                        )
            if scheduler and "scheduler_state_dict" in train_state:
                try:
                    scheduler.load_state_dict(train_state["scheduler_state_dict"])
                except (ValueError, RuntimeError, KeyError) as e:
                    logging.warning(f"Skipping scheduler state restore: {e}")
            # Restore RNG state only on full resume (optimizer present).
            # Weights-only resume intentionally starts with fresh RNG because
            # it's usually a finetune on different data.
            if optimizer and "rng_state" in train_state:
                rng = train_state["rng_state"]
                try:
                    import random as _random
                    import numpy as _np
                    if "python" in rng:
                        _random.setstate(rng["python"])
                    if "numpy" in rng:
                        _np.random.set_state(rng["numpy"])
                    if "torch_cpu" in rng:
                        torch.set_rng_state(rng["torch_cpu"])
                    if "torch_cuda_all" in rng and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
                    logging.info("Restored RNG state from checkpoint")
                except Exception as e:
                    logging.warning(f"Skipping RNG state restore: {e}")
            return train_state.get("step", 0)
        return 0

    if checkpoint_path.is_file():
        ckpt = _torch_load_with_retry(checkpoint_path)
        if "model_state_dict" in ckpt:
            # Legacy bundle resume uses the same strict default as the model_dir
            # branch above (weights-only warmstart gets strict=False via the
            # optimizer-is-None path).
            policy.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if optimizer and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (KeyError, ValueError, RuntimeError) as e:
                logging.warning(f"Skipping optimizer state restore: {e}")
        if scheduler and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (KeyError, ValueError, RuntimeError) as e:
                logging.warning(f"Skipping scheduler state restore: {e}")
        return ckpt.get("step", 0)

    raise FileNotFoundError(f"No valid checkpoint found at {checkpoint_path}")


# ---------------------------------------------------------------------------
# Public API surface.
#
# scripts/train.py (the only external consumer) used to import twelve
# underscore-"private" helpers directly — a private-by-name / public-by-use
# contradiction that left refactors of this module without a visible contract.
# The supported cross-module API is the names below; the underscore originals
# remain as the in-module implementations (and as back-compat aliases for any
# out-of-tree caller).
# ---------------------------------------------------------------------------
accelerate_state_dir = _accelerate_state_dir
checkpoint_incomplete_marker = _checkpoint_incomplete_marker
clear_checkpoint_incomplete_marker = _clear_checkpoint_incomplete_marker
write_checkpoint_incomplete_marker = _write_checkpoint_incomplete_marker
is_recoverable_checkpoint_save_error = _is_recoverable_checkpoint_save_error
load_training_state = _load_training_state
rotate_checkpoints = _rotate_checkpoints
torch_load_with_retry = _torch_load_with_retry
torch_save_with_retry = _torch_save_with_retry
validate_full_resume_contract = _validate_full_resume_contract

__all__ = [
    "FULL_RESUME_CONTRACT_KEYS",
    "save_checkpoint",
    "load_checkpoint",
    "accelerate_state_dir",
    "checkpoint_incomplete_marker",
    "clear_checkpoint_incomplete_marker",
    "write_checkpoint_incomplete_marker",
    "is_recoverable_checkpoint_save_error",
    "load_training_state",
    "rotate_checkpoints",
    "torch_load_with_retry",
    "torch_save_with_retry",
    "validate_full_resume_contract",
]

# Public name for the contract-key tuple as well (read-only by convention).
FULL_RESUME_CONTRACT_KEYS = _FULL_RESUME_CONTRACT_KEYS
